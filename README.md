# Proyecto de Programación para la Ciencia de Datos

![CI](https://github.com/joserodriguezc/grupo_1_programacion_ciencia_datos/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

Repositorio del **Grupo 1** para el proyecto transversal del curso **Programación para la Ciencia de Datos**, correspondiente al Magíster en Ciencia de Datos e Inteligencia Artificial.

## Tema del proyecto

> Posiciones políticas relativas, estabilidad y cohesión legislativa en la tramitación del proyecto de ley sobre protección de datos personales en Chile (boletín 11092-07)

### Objetivo general

Analizar las posiciones políticas relativas y la estabilidad del comportamiento legislativo de las diputadas y los diputados, así como las posiciones políticas relativas y el grado de cohesión de los partidos políticos, a partir de todas las votaciones nominales asociadas al proyecto de ley sobre protección de datos personales en la Cámara de Diputadas y Diputados de Chile.

### Preguntas de investigación

1. ¿Qué posiciones políticas relativas presentan las diputadas y los diputados a partir de sus patrones de votación durante la tramitación del proyecto de ley?
2. ¿Qué grado de estabilidad o variabilidad presenta el comportamiento individual de las diputadas y los diputados a lo largo de las votaciones analizadas?
3. ¿Qué posiciones políticas relativas presentan los partidos políticos a partir del comportamiento de sus integrantes?
4. ¿Qué grado de cohesión o dispersión presentan los partidos políticos durante la tramitación del proyecto de ley?

## Contribuidores

- Yerko Gallardo
- Sebastián Rojas
- José Ignacio Rodríguez

## Estado del proyecto

| Fase | Descripción | Estado |
| ------ | ------------- | -------- |
| **F1** | Planteamiento del problema y diseño del proyecto | ✅ Completa |
| **F2** | Obtención, limpieza y procesamiento de datos | ✅ Completa |
| **F3** | Refactorización modular, equivalencia y rendimiento | ✅ Completa |
| **F4** | Modelamiento, resultados y entrega | ✅ Completa |

## Resultados principales (F4)

Los resultados se integran en [`F4_06_comunicacion.ipynb`](F4/notebooks/F4_06_comunicacion.ipynb). Allí, cada figura dice su hallazgo en el título y va seguida de cuatro frases: qué muestra, qué se infiere, qué límite tiene y cómo aporta al relato. Todos los resultados son **descriptivos**, porque no existe un padrón verificable de elegibilidad por votación.

- **Corpus.** 15 votaciones nominales; 14 de ellas se realizaron en una sola sesión (08-05-2023) y 2 fueron unánimes.
- **P1 · Posiciones individuales.** Dos bloques claramente separados en el eje B-Call (94 diputados en el lado L y 42 en el lado R) y solo 20 posiciones intermedias. El clustering reproduce los mismos grupos (ARI = 1), la primera componente del PCA coincide con d1 (r = 0,9999) y los 25 pivotes elegibles generan el mismo eje.
- **P2 · Estabilidad individual.** El 80 % de los diputados nunca cruza el eje y 27 alternan Sí y No. El corpus no permite evaluar trayectorias en el tiempo.
- **P3 · Posiciones partidarias.** Un bloque compacto de 9 partidos en el lado L, UDI y PREP en el lado R, y una zona central (EVOP, PDG, RN) cuyo valor intermedio refleja votos divididos, no una posición común. IND se calcula, pero no se publica porque no es un partido.
- **P4 · Cohesión.** 10 de 14 partidos votaron siempre unidos; la dispersión se concentra en PDG, RN y UDI.

## Avance por fase

### F1

- ✅ Definición del problema, preguntas, objetivos, mapa conceptual y revisión de literatura (`F1/docs/`, `F1_Definición.ipynb`).

### F2

- ✅ Extracción de votaciones del proyecto de ley (boletín 11092-07), períodos legislativos, diputados y militancias, y detalle nominal de cada votación (`F2/src/01` a `04`, orquestados desde `F2_01_obtencion.ipynb`).
- ✅ XML crudos y CSV intermedios en `F2/data/raw/` y `F2/data/interim/`.
- ✅ Análisis exploratorio (`F2_02_exploracion.ipynb`).
- ✅ Normalización, limpieza y reglas de calidad (`F2_03_procesamiento_validacion.ipynb`), con reporte en `F2/data/processed/reporte_calidad.csv`.
- ✅ Big table analítica, con una fila por diputado y votación (`F2_04_Integración.ipynb` → `F2/data/processed/big_table_analitica.csv`), y diagnóstico en `diagnostico_integracion.csv`.
- ✅ Tests unitarios, contratos de datos y ejecución de notebooks (`F2/test/`).

### F3

F3 refactoriza la lógica de F2 en componentes reutilizables, verificables y medibles, con F2 como referencia.

- ✅ Paquete modular `F3/src/nucleo/`: extracción, normalización, transformación (`TransformadorDataset`), integración (`construir_big_table()`), asignación temporal de militancias y validación.
- ✅ Equivalencia verificada entre las tablas procesadas y la big table de F3 y los productos de F2.
- ✅ Asignación temporal de militancias en versión iterativa y vectorizada, equivalentes sobre los datos reales; la integración usa la vectorizada.
- ✅ Benchmarks reproducibles (escalas `1×` a `100×`, cinco repeticiones y calentamiento), con speedup y crecimiento calculados en el notebook.
- ✅ Tests unitarios y ejecución automatizada de las secciones funcionales de `F3_01_nucleo.ipynb` en CI; el benchmark pesado se ejecuta manualmente.

### F4

F4 estima los modelos sobre el corte analítico de F3, bajo un protocolo metodológico aprobado por el equipo.

- ✅ **Protocolo** (`F4/config/analisis.toml`, v0.8.0, aprobado) y **registro de decisiones** (`F4/data/reports/decisiones_metodologicas.json`), sincronizados mediante un test. Aprobados los universos de cada método y la clasificación de los resultados.
- ✅ **Corte y auditoría** de la entrada F3: manifiesto con commit y SHA-256, y auditoría sin errores.
- ✅ **Codificación y matrices**: vistas nominal, ternaria y binaria, máscara de observación, afiliación por votación y universo por método.
- ✅ **B-Call** (puerto del paquete R `bcall`, con agrupamiento automático y pivote documentado): posición (d1) y variabilidad (d2) individuales.
- ✅ **Posición partidaria** (P_p), **cohesión** (Agreement Index, Rice y entropía), **afinidad** entre pares (Hamming) y **contrastes** con clustering y PCA/SVD.
- ✅ **Sensibilidad** de cada método: retiro de una votación, pivotes, casos frontera, umbrales y bloques de votaciones.
- ✅ **Conciliación** cruzada de conteos y universos, sin pendientes.
- ✅ **Contrato de datos**, **pipeline reproducible** y **manifiesto de entrega** con los hashes de los 98 entregables.
- ✅ Tests unitarios, de integración (la cadena regenerada coincide con lo versionado) y de ejecución de los notebooks F4_01 a F4_06.
- ✅ **Entrega final**: informe técnico, presentación y video de exposición preparados para la entrega.

| Notebook | Contenido |
| -------- | --------- |
| `F4_01_datos_y_cobertura` | Corte, auditoría, codificación, matrices, cobertura y conciliación |
| `F4_02_bcall` | B-Call: posiciones individuales, cotejo externo y sensibilidad |
| `F4_03_metricas_partidos_afinidad` | Posición partidaria, afinidad entre pares y su sensibilidad |
| `F4_04_clustering_pca` | Clustering y PCA como contrastes de B-Call |
| `F4_05_cohesion` | Cohesión partidaria y su sensibilidad |
| `F4_06_comunicacion` | Respuestas a las cuatro preguntas y al objetivo general |

Las convenciones de estructura y visualización de los notebooks están en [`F4/docs/convenciones_notebooks.md`](F4/docs/convenciones_notebooks.md).

## Estructura del repositorio

```text
.
├── .github/workflows/ci.yml   # CI en cada PR hacia main: identidad Git, ruff y tests (rápidos y notebooks)
│
├── F1/
│   ├── docs/                  # Informes, mapa conceptual y papers
│   └── notebooks/F1_Definición.ipynb
│
├── F2/
│   ├── data/
│   │   ├── raw/               # XML originales de la fuente
│   │   ├── interim/           # Productos intermedios de la extracción
│   │   └── processed/         # Tablas limpias, validadas e integradas (big table)
│   ├── notebooks/             # F2_01 obtención · F2_02 exploración · F2_03 procesamiento · F2_04 integración
│   ├── src/                   # Scripts de extracción (01–04) y validaciones
│   └── test/                  # unit/, data_contracts/, notebooks/
│
├── F3/
│   ├── data/                  # interim/ y processed/ (incluye big_table_analitica.csv, entrada de F4)
│   ├── notebooks/F3_01_nucleo.ipynb   # Baseline, equivalencia funcional y benchmarks
│   ├── src/nucleo/            # Extracción, normalización, transformación, integración,
│   │                          # asignación temporal, validación y métricas de rendimiento
│   └── test/                  # unit/, notebooks/
│
├── F4/
│   ├── config/                # analisis.toml (protocolo) y diccionario_votos.toml
│   ├── data/
│   │   ├── processed/         # Votos codificados, matrices y afiliación por votación
│   │   ├── results/           # individual/ (bcall, pca), partidos/, pares/, grupos/, comparacion/
│   │   └── reports/           # Manifiestos, auditoría, universo, sensibilidad, conciliación,
│   │                          # registro de decisiones y manifiesto de entrega
│   ├── docs/                  # Convenciones de notebooks, referencia ideológica externa,
│   │                          # informe, presentación
│   ├── figures/               # Una carpeta por notebook (comunicacion/ para F4_06)
│   ├── notebooks/             # F4_01 a F4_06
│   ├── src/analisis/          # Módulos: corte, auditoria, codificacion, matrices, cobertura,
│   │                          # bcall, orientacion, posicion_partidos, cohesion, afinidad,
│   │                          # agrupamiento, pca_svd, comparacion_metodos, sensibilidad,
│   │                          # universo, conciliacion, contratos, pipeline, exportacion,
│   │                          # visualizacion
│   ├── test/                  # unit/, integration/, notebooks/
│   └── video/
│
├── tests/                     # Chequeo de calidad de todos los notebooks (sin celdas vacías ni sin ejecutar)
├── pyproject.toml
└── uv.lock
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

## Configuración del entorno

Clonar el repositorio e instalar las dependencias bloqueadas:

```bash
git clone <url-del-repo>
cd grupo_1_programacion_ciencia_de_datos
uv sync
```

Con `uv` no es necesario activar el entorno: `uv run` ejecuta cualquier comando dentro del `.venv` que crea `uv sync`.

```bash
uv run jupyter lab
uv run pytest -m "not slow"           # tests rápidos de F2, F3 y F4 (unitarios y contratos)
uv run pytest -m slow                 # ejecución completa de notebooks e integración de F4
uv run pytest tests                   # calidad de todos los notebooks
uv run ruff check F2/src F2/test F3/src F3/test F4/src F4/test
```

### Reproducir los resultados de F4

```bash
# 1. Salidas de B-Call (posiciones individuales y cotejo externo): ejecutar F4_02_bcall.ipynb
# 2. Cadena de módulos: regenera todas las salidas, valida el contrato y escribe el manifiesto
uv run python -m F4.src.analisis.pipeline
# Pasos sueltos, por ejemplo:
uv run python -m F4.src.analisis.pipeline --pasos sensibilidad universo conciliacion
# Validar el contrato de datos o regenerar solo el manifiesto de entrega
uv run python -m F4.src.analisis.contratos
uv run python -m F4.src.analisis.exportacion
```

Si vuelves a ejecutar algún notebook, regenera el manifiesto de entrega: registra el hash de cada notebook y de cada figura.

Para activar el entorno y trabajar sin anteponer `uv run`:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

## Fuente de datos

Servicios de datos abiertos de la Cámara de Diputadas y Diputados de Chile ([datosAbiertos.aspx](https://www.camara.cl/transparencia/datosAbiertos.aspx)), consultados mediante el webservice SOAP para el proyecto de ley identificado con el **boletín 11092-07**. La clasificación ideológica externa de los partidos, usada solo para el cotejo, está en `F4/docs/clasificacion_ideologica_partidos_chilenos.json`.

## Herramientas principales

- **Análisis y modelos**: pandas, numpy, scipy, matplotlib, seaborn
- **Extracción de datos**: zeep (SOAP), lxml
- **Notebooks**: JupyterLab, nbclient (ejecución en tests)
- **Calidad de código**: ruff
- **Testing**: pytest (unitarios, contratos de datos, integración y ejecución de notebooks)
- **Integración continua**: GitHub Actions

### Entregables finales

- 📄 **Informe técnico**: [`F4/docs/informe/f4_s04_evaluacion_entregable_grupo1.pdf`](F4/docs/informe/f4_s04_evaluacion_entregable_grupo1.pdf)
- 📊 **Presentación con guion integrado**: [`F4/docs/presentacion/f4_presentacion_grupo1_con_guion.pptx`](F4/docs/presentacion/f4_presentacion_grupo1_con_guion.pptx)
- 🎥 **Presentación audiovisual**: [`F4/video/Presentacion_grupo_1.mp4`](F4/video/Presentacion_grupo_1.mp4)