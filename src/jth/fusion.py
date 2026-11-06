"""Ways of combining LR and CF.

- Weighted reciprocal-rank fusion and the history gate (notebook cell 72).
- A: weighted score blend (Burke 2002; convex combination in Bruch et al. 2023).
- B: blend with one weight per history bin (Jahrer et al. 2010, binned linear regression).
- C: feature-weighted linear stacking (Sill et al. 2009), fitted with a pairwise logistic
  loss (linear BPR, Rendle et al. 2009).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

RRF_C = 60.0
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HISTORY_BINS = ("0", "1", "2+")
NORM_EPS = 1e-6


# ------------------------------------------------------------------ rank fusion
def rrf_pool(first: np.ndarray, second: np.ndarray, alpha: float, n_jobs: int, c: float = RRF_C):
    """`alpha/(c+rank_1) + (1-alpha)/(c+rank_2)` over the union. Absent from a list adds zero."""
    if alpha == 1:
        return first, 1 / (c + np.arange(1, len(first) + 1))
    if alpha == 0:
        return second, 1 / (c + np.arange(1, len(second) + 1))
    pool = np.union1d(first, second)
    s1, s2 = np.zeros(n_jobs), np.zeros(n_jobs)
    s1[first] = 1 / (c + np.arange(1, len(first) + 1))
    s2[second] = 1 / (c + np.arange(1, len(second) + 1))
    return pool, alpha * s1[pool] + (1 - alpha) * s2[pool]


# ------------------------------------------------------------------ normalization
def minmax_rows(scores: np.ndarray, masks: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    """Min-max per query over its eligible jobs, into [eps, 1]. A constant row becomes eps.

    So a model with nothing to say about a query (CF on a new candidate) adds a constant
    and cannot change the ranking, and `log` of the result is always defined.
    """
    s = np.where(masks, scores, np.nan)
    low = np.nanmin(s, axis=1, keepdims=True)
    high = np.nanmax(s, axis=1, keepdims=True)
    span = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(span > 0, (scores - low) / span, 0.0)
    return eps + (1 - eps) * np.clip(unit, 0, 1)


def zscore_rows(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """z-score per query over eligible jobs. A constant row becomes zero."""
    s = np.where(masks, scores, np.nan)
    mean = np.nanmean(s, axis=1, keepdims=True)
    std = np.nanstd(s, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > 0, (scores - mean) / std, 0.0)


# ------------------------------------------------------------------ A and B
def history_bins(history: np.ndarray) -> np.ndarray:
    """0: no earlier process. 1: one. 2: two or more."""
    return np.minimum(history, 2)


def blend(norm_a: np.ndarray, norm_b: np.ndarray, alpha) -> np.ndarray:
    """`alpha*a + (1-alpha)*b`. `alpha` may be one value per query."""
    a = np.asarray(alpha, dtype=float)
    a = a[:, None] if a.ndim == 1 else a
    return a * norm_a + (1 - a) * norm_b


def gate_alpha(warm: np.ndarray, alpha: float) -> np.ndarray:
    """Weight of the first model per query: 1 for new candidates, `alpha` otherwise."""
    return np.where(warm, alpha, 1.0)


def best_alpha_per_bin(rr_by_alpha: dict, bins: np.ndarray, mask=None) -> tuple[np.ndarray, dict]:
    """Best alpha per bin. MRR decomposes across queries, so bins can be chosen separately.

    `rr_by_alpha`: alpha -> reciprocal ranks of that global blend. Ties go to the first alpha
    in the given order. Returns the per-query alphas and the chosen alpha per bin.
    """
    mask = np.ones(len(bins), dtype=bool) if mask is None else mask
    chosen = {}
    for b in np.unique(bins):
        selected = (bins == b) & mask
        means = {a: rr[selected].mean() for a, rr in rr_by_alpha.items()}
        chosen[int(b)] = max(means, key=lambda a: (means[a], -list(rr_by_alpha).index(a)))
    alphas = np.array([chosen[int(b)] for b in bins], dtype=float)
    return alphas, chosen


# ------------------------------------------------------------------ C
def fwls_meta(warm: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Candidate-level meta-features: constant, has history, log1p(history)."""
    return np.column_stack([np.ones(len(warm)), warm.astype(float), np.log1p(history)])


def fwls_score(inputs: list, meta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Score per job: `sum_k input_k * (meta @ W_k)`. `weights` has k*m values, by input.

    A candidate-level meta-feature on its own never changes the order inside a query, so it
    only acts through its product with a job-level input. That is why C is FWLS and not a
    plain regression with history as one more input.
    """
    w = np.asarray(weights).reshape(len(inputs), meta.shape[1])
    coef = meta @ w.T  # queries x inputs
    return sum(coef[:, k, None] * inputs[k] for k in range(len(inputs)))


def sample_negatives(gold_idx: np.ndarray, masks: np.ndarray, top_scores: np.ndarray, n_uniform: int,
                     n_hard: int, seed: int = 42) -> list:
    """Per query: uniform eligible jobs plus the first model's top jobs, never the relevant one."""
    rng = np.random.default_rng(seed)
    out = []
    for i, g in enumerate(gold_idx):
        eligible = np.flatnonzero(masks[i])
        eligible = eligible[eligible != g]
        uniform = rng.choice(eligible, size=min(n_uniform, len(eligible)), replace=False)
        order = eligible[np.argsort(-top_scores[i, eligible], kind="stable")[:n_hard]]
        out.append(np.unique(np.concatenate([uniform, order])))
    return out


def fwls_pairs(inputs: list, meta: np.ndarray, gold_idx: np.ndarray, negatives: list, fit_mask: np.ndarray) -> np.ndarray:
    """Rows x_pos - x_neg for the queries in `fit_mask`, without building the full tensor."""
    rows = []
    for i in np.flatnonzero(fit_mask):
        g, neg = gold_idx[i], negatives[i]
        diff = np.stack([inp[i, g] - inp[i, neg] for inp in inputs], axis=1)  # pairs x inputs
        rows.append(np.einsum("pk,m->pkm", diff, meta[i]).reshape(len(neg), -1))
    return np.concatenate(rows)


def fit_pairwise(pairs: np.ndarray, c: float, seed: int = 42) -> np.ndarray:
    """Linear BPR: logistic regression on x_pos - x_neg, no intercept. Pairs mirrored for balance."""
    x = np.concatenate([pairs, -pairs])
    y = np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))])
    model = LogisticRegression(C=c, fit_intercept=False, max_iter=2000, random_state=seed)
    model.fit(x, y)
    return model.coef_[0]
