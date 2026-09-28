# Proyecto de Programación para la Ciencia de Datos

![CI](https://github.com/joserodriguezc/grupo_1_programacion_ciencia_datos/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

Repositorio del **Grupo 1** para el proyecto transversal del curso **Programación para la Ciencia de Datos**, correspondiente al Magíster en Ciencia de Datos e Inteligencia Artificial.

## Tema del proyecto

> Posicionamiento ideológico y cohesión legislativa en la tramitación del proyecto de ley sobre protección de datos personales en Chile

### Pregunta principal

¿Qué patrones de posicionamiento ideológico relativo y cohesión legislativa se observan entre los diputados y partidos políticos en las votaciones asociadas al boletín 11092-07 del proyecto de Ley Sobre Protección de los Datos Personales?

### Objetivo general

Analizar el posicionamiento ideológico relativo y la cohesión del comportamiento legislativo de los diputados y partidos políticos en las votaciones realizadas sobre el proyecto de Ley Sobre Protección de los Datos Personales.

## Contribuidores

- Yerko Gallardo
- Sebastian Rojas
- José Ignacio Rodríguez

## Estado del proyecto

| Fase | Descripción | Estado |
| ------ | ------------- | -------- |
| **F1** | Planteamiento del problema y diseño del proyecto | ✅ Completa |
| **F2** | Obtención, limpieza y procesamiento de datos | ✅ Extracción, procesamiento e integración completos; tests y CI en funcionamiento |
| **F3** | Análisis exploratorio / modelamiento | ⬜ Pendiente |
| **F4** | Resultados finales y entrega | ⬜ Pendiente |

### Avance de F2

- ✅ Extracción de votaciones del proyecto de ley (boletín 11092-07), períodos legislativos, diputados/militancias y detalle nominal de cada votación (`F2/src/01` a `04`, orquestados desde `F2_01_obtencion.ipynb`).
- ✅ XML crudos y CSV intermedios generados en `F2/data/raw/` e `F2/data/interim/`.
- ✅ Análisis exploratorio (`F2_02_exploracion.ipynb`).
- ✅ Normalización, limpieza y aplicación de reglas de calidad acordadas por el equipo (`F2_03_procesamiento_validacion.ipynb`), con reporte de observaciones en `F2/data/processed/reporte_calidad.csv`.
- ✅ Integración de las tablas procesadas en una big table analítica, una fila por diputado y votación (`F2_04_Integración.ipynb` → `F2/data/processed/big_table_analitica.csv`), con diagnóstico de calidad en `diagnostico_integracion.csv`.
- ✅ Suite de tests automatizados en `F2/test/` (unitarios sobre `F2/src/`, contratos de datos sobre `F2/data/processed/` y ejecución completa de los notebooks de procesamiento/integración).
- ✅ Integración continua con GitHub Actions (`.github/workflows/ci.yml`): lint y tests corren en cada PR hacia `main`.
- ✅ Orden determinista en la detección de diputados sin voto, para que `reporte_calidad.csv` no cambie entre corridas del notebook sin que haya cambios reales de datos.
- ✅  `F2/docs/`

## Avance de F3

F3 se orienta a refactorizar la lógica desarrollada en F2 hacia componentes reutilizables, verificables y medibles, manteniendo como referencia los resultados obtenidos durante la fase anterior.

- ✅ Estructura modular de F3 implementada bajo `F3/src/nucleo/`.
- ✅ Separación de responsabilidades entre transformación, integración, asignación temporal y métricas.
- ✅ Procesamiento encapsulado mediante `TransformadorDataset`.
- ✅ Validación de equivalencia entre las cinco tablas procesadas generadas por F3 y los productos procesados de F2.
- ✅ Integración analítica mediante `construir_big_table()`.
- ✅ Validación de equivalencia entre la big table construida por F3 y `F2/data/processed/big_table_analitica.csv`.
- ✅ Implementación de asignación temporal de militancias mediante una estrategia iterativa.
- ✅ Implementación alternativa vectorizada para la asignación temporal de militancias.
- ✅ La integración productiva de F3 utiliza actualmente la implementación vectorizada.
- ✅ Validación de equivalencia entre las implementaciones iterativa y vectorizada sobre los datos reales.
- ✅ Validación de una única militancia vigente por voto.
- ✅ Cobertura mediante tests unitarios de los estados `OK`, `SIN_MILITANCIA` y `AMBIGUA`.
- ✅ Construcción de un baseline reproducible de F2 para utilizarlo como referencia.
- ✅ Corrección de las mediciones de extracción del baseline para evitar incluir la preparación de fixtures dentro de los tiempos medidos.
- ✅ Uso de artefactos locales de F2 para evitar llamadas de red durante los benchmarks.
- ✅ Redirección de escrituras temporales fuera de F2 durante las mediciones, manteniendo sus XML y CSV como artefactos de solo lectura.
- ✅ Identificación de bloques de notebooks de F2 por secciones en lugar de depender exclusivamente de índices de celdas.
- ✅ Benchmark comparativo entre asignación iterativa y vectorizada.
- ✅ Pruebas de escalabilidad con volúmenes sintéticos `1×`, `5×`, `10×` y `100×`.
- ✅ Cinco repeticiones y una ejecución de calentamiento por escala para obtener medianas más estables.
- ✅ Cálculo de tiempo de ejecución, speedup y crecimiento respecto de la escala base.
- ✅ Comparación descriptiva entre la etapa de integración temporal de F2 y las implementaciones de asignación de F3.
- ✅ Resultados numéricos generados dinámicamente en el notebook para evitar conclusiones desactualizadas.
- ✅ Interpretación de las pruebas como evidencia empírica de rendimiento, sin inferir una complejidad asintótica no demostrada.
- ✅ Prueba automatizada de las secciones funcionales de `F3_01_nucleo.ipynb` en CI.
- ✅ El benchmark pesado de escalabilidad queda fuera de CI y se ejecuta manualmente para el análisis experimental.
- ✅  Documentación y análisis final de resultados de F3 en desarrollo.

## Estructura del repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
│           # CI ejecutado en cada PR hacia main:
│           # identidad Git, lint y tests de F2/F3
│
├── F1/
│   ├── docs/
│   │   ├── informes/
│   │   ├── mapa conceptual/
│   │   └── papers/
│   └── notebooks/
│       └── F1_Definición.ipynb
│
├── F2/
│   ├── data/
│   │   ├── raw/
│   │   │   # Respuestas XML originales obtenidas desde la fuente
│   │   ├── interim/
│   │   │   # Productos intermedios generados durante la extracción
│   │   └── processed/
│   │       # Tablas limpias, validadas e integradas
│   │
│   ├── docs/
│   │
│   ├── notebooks/
│   │   ├── F2_01_obtencion.ipynb
│   │   ├── F2_02_exploracion.ipynb
│   │   ├── F2_03_procesamiento_validacion.ipynb
│   │   └── F2_04_Integración.ipynb
│   │
│   ├── src/
│   │   ├── 01_extraer_votaciones_proyecto.py
│   │   ├── 02_periodos_legislativos.py
│   │   ├── 03_diputados.py
│   │   ├── 04_extraer_detalles_votaciones.py
│   │   └── validaciones.py
│   │
│   └── test/
│       ├── unit/
│       ├── data_contracts/
│       └── notebooks/
│
├── F3/
│   ├── data/
│   │   ├── interim/
│   │   │   # Entradas intermedias utilizadas durante la refactorización
│   │   └── processed/
│   │       # Productos procesados y analíticos utilizados por F3
│   │
│   ├── notebooks/
│   │   └── F3_01_nucleo.ipynb
│   │       # Baseline, equivalencia funcional y benchmarks
│   │
│   ├── src/
│   │   └── nucleo/
│   │       ├── asignadores.py
│   │       │   # Asignación temporal iterativa y vectorizada
│   │       ├── integrador_dataset.py
│   │       │   # Construcción de la big table analítica
│   │       ├── transformador_dataset.py
│   │       │   # Transformación y procesamiento reutilizable
│   │       ├── metricas.py
│   │       │   # Funciones de medición de rendimiento
│   │       └── rendimiento_f2.py
│   │           # Baseline reproducible de F2
│   │
│   └── test/
│       ├── unit/
│       │   # Tests unitarios de componentes de F3
│       └── notebooks/
│           # Ejecución automatizada de secciones funcionales del notebook
│
├── F4/
│   # Por definir
│
├── pyproject.toml
└── uv.lock

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

Con `uv` no es necesario activar el entorno manualmente: `uv run` ejecuta cualquier comando dentro del `.venv` que crea `uv sync`.

```bash
uv run jupyter lab
uv run pytest                    # suite completa (incluye ejecutar F2_03 y F2_04 de punta a punta)
uv run pytest -m "not slow"      # solo tests rápidos, sin ejecutar los notebooks
uv run ruff check F2/src F2/test
```

Si prefieres activar el entorno para trabajar sin anteponer `uv run` a cada comando, también puedes hacerlo:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

## Fuente de datos

Servicios de datos abiertos de la Cámara de Diputadas y Diputados de Chile ([datosAbiertos.aspx](https://www.camara.cl/transparencia/datosAbiertos.aspx)), consultados mediante el webservice SOAP para el proyecto de ley identificado con el **boletín 11092-07**.

## Herramientas principales

- **Análisis**: pandas, numpy, matplotlib, seaborn
- **Extracción de datos**: zeep (SOAP), lxml
- **Notebooks**: JupyterLab
- **Calidad de código**: ruff
- **Testing**: pytest (unitarios, contratos de datos y ejecución de notebooks)
- **Integración continua**: GitHub Actions
