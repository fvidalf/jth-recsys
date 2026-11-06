# Anexo del informe intermedio: formulación de la tarea y procedencia de las cifras

Material para el anexo del H2, en español y con el estilo de `docs/report_style.md`. Las
cifras que aparecen acá salen de `scripts/01_datos_y_validacion.ipynb`, cuyos outputs están
versionados, o de `results/validacion_*.csv`, que genera `scripts/resultados_validacion.py`.
Si una cifra no está en ninguno de esos dos lugares, no debe ir en el informe.

## 1. Formulación de la tarea

El sistema recibe un candidato y un día `t` y devuelve un ordenamiento de las ofertas
disponibles en `t`. La oferta correcta es la del proceso que ese candidato inició ese día, sin
importar la etapa que haya alcanzado después, porque el conjunto registra procesos de
reclutamiento y no contrataciones.

Una consulta es entonces un par candidato-día con al menos un proceso cuyo inicio cae en el
período de test. Hay 8.458 procesos de test y 8.341 consultas, porque cuando un candidato
inicia varios procesos el mismo día el evaluador conserva una sola oferta como relevante.

Cada modelo devuelve a lo más 1.000 ofertas ordenadas por consulta. Las ofertas creadas
después de `t` se eliminan antes de puntuar, de modo que el modelo nunca ve el catálogo
futuro. La métrica principal es el MRR y la secundaria el nDCG@10. Todo resultado se reporta
separado entre candidatos con historial y candidatos nuevos, porque los modelos se comportan
de forma muy distinta en cada grupo.

En el día `t` un modelo puede usar solo interacciones anteriores a `t` y ofertas creadas hasta
`t`. Ningún umbral, peso ni hiperparámetro se elige mirando test. Las decisiones de este
informe se toman en una validación interna, descrita en la sección siguiente.

## 2. Bloques de datos

| Bloque | Desde | Hasta | Procesos | Consultas | Uso |
|---|---|---|---|---|---|
| Entrenamiento oficial | 2018-03-09 | 2024-03-23 | 33.830 | 33.604 | Ajuste y validación |
| Entrenamiento reducido | 2018-03-09 | 2023-07-20 | 29.599 | 29.396 | Ajuste de los modelos |
| Validación | 2023-07-21 | 2024-03-23 | 4.231 | 4.208 | Elegir umbrales y pesos |
| Test oficial | 2024-03-23 | 2025-04-21 | 8.458 | 8.341 | Solo evaluación final |

La validación ocupa el 12,51 % del entrenamiento oficial y está separada por día, sin días
compartidos. El test oficial y el entrenamiento oficial sí comparten el 2024-03-23, porque el
divisor del paper corta por índice de fila y no por fecha. Es una limitación heredada del
código oficial y no se corrige acá.

## 3. Cifras que sostienen el argumento

| Afirmación | Dónde se calcula | Bloque |
|---|---|---|
| La vigencia domina la señal | Figura `fig_antiguedad_positivo.png`, notebook §10.7 | Validación |
| El filtro por actividad sube el MRR de LR | `results/tabla_componentes.tex`, notebook §15 | Validación |
| El peso suave supera al corte duro | `results/tabla_aportes.tex`, notebook §15 | Validación |
| CF y LR son complementarios por historial | `results/tabla_componentes.tex` | Validación |
| El baseline propio de contenido supera a Recencia | `results/tabla_aportes.tex` | Validación |
| Una ventana de 30 días conserva el positivo en la mayoría de las consultas | Figura `fig_filtro_actividad.png`, notebook §13 | Validación |
| Qué proporción de consultas llega sin historial de candidato | notebook §26 | Validación |

Las figuras las extrae `scripts/figuras_notebook.py` y quedan registradas con su celda de
origen y la afirmación que respaldan en `results/figuras_notebook.csv`.

## 4. Lo que todavía no tiene respaldo en el repositorio

El análisis descriptivo del H1 citó tres cifras que este repositorio no reproduce con ningún
script, el 92 % de ofertas correctas con actividad en los 30 días previos y el 95 % frente al
39 % de ofertas y candidatos con historial. La pauta pidió una tabla detrás de esas cifras.
Antes de repetirlas en el H2 hay dos caminos honestos. Recalcularlas sobre el entrenamiento
oficial con un script y reportarlas con su denominador, o atribuirlas explícitamente al
análisis de la propuesta y no presentarlas como resultado de esta etapa.

La cifra de cobertura de la ventana de 30 días que sí está respaldada es la del notebook §13,
que se calcula sobre las 4.208 consultas de validación y no sobre las ofertas del registro
completo. Las dos miden cosas distintas y no deben mezclarse en la misma frase.
