"""Profile features shared by candidates and jobs."""
from __future__ import annotations

import pandas as pd

RARE = "_rare_skill_"

# Semicolon lists present (with the same vocabulary) on both sides.
# Tokens get a namespace prefix so "data" the area != "data" the skill.
# Sensitive attributes (llm_sex, llm_nationality, llm_age_bucket, ...) are
# deliberately left out.
TOKEN_FIELDS = {
    "skills": "s",
    "llm_hard_skills": "s",
    "llm_programming_languages": "s",
    "llm_tools_technologies": "s",
    "expertise_area": "a",
    "llm_expertise_area": "a",
    "job_category": "c",
    "llm_job_category": "c",
}


def split_list(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [tok.strip().lower() for tok in value.split(";") if tok.strip() and tok.strip() != RARE]


def profile_tokens(table: pd.DataFrame, fields: dict[str, str] = TOKEN_FIELDS) -> list[list[str]]:
    """One bag of namespaced tokens per row (duplicates across fields kept,
    so a skill listed by both the recruiter and the LLM weighs more)."""
    docs: list[list[str]] = [[] for _ in range(len(table))]
    for field, prefix in fields.items():
        if field not in table.columns:
            continue
        for doc, value in zip(docs, table[field]):
            doc.extend(f"{prefix}:{tok}" for tok in split_list(value))
    return docs
