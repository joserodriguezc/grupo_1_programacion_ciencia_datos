# Proyecto de Programación para la Ciencia de Datos

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
| **F2** | Obtención, limpieza y procesamiento de datos | 🚧 Extracción completa; limpieza, integración y validación en curso |
| **F3** | Análisis exploratorio / modelamiento | ⬜ Pendiente |
| **F4** | Resultados finales y entrega | ⬜ Pendiente |

### Avance de F2

- ✅ Extracción de votaciones del proyecto de ley (boletín 11092-07), períodos legislativos, diputados/militancias y detalle nominal de cada votación (`F2/src/01` a `04`, orquestados desde `F2_01_obtencion.ipynb`).
- ✅ XML crudos y CSV intermedios generados en `F2/data/raw/` e `F2/data/interim/`.
- 🚧 Integración temporal de las fuentes (`F2/src/06_integracion.py`, aún sin implementar).
- ⬜ Normalización/limpieza (script 05), validación y construcción de matrices (script 07).
- ⬜ Notebooks `F2_02_exploracion.ipynb` y `F2_03_procesamiento_validacion.ipynb` (creados, aún vacíos).
- ⬜ `F2/data/processed/`, `F2/docs/`, `F2/test/` y manifiestos de extracción, sin contenido todavía.

## Estructura del repositorio

```text
.
├── F1/
│   ├── docs/                 # Informe, mapa conceptual y papers de referencia
│   └── notebooks/
│       └── F1_Definición.ipynb
├── F2/
│   ├── data/
│   │   ├── raw/               # XML crudos: votaciones, diputados, proyecto de ley
│   │   ├── interim/            # CSV intermedios: diputados, militancias, períodos, votaciones
│   │   └── processed/          # Tablas analíticas validadas (pendiente)
│   ├── docs/                  # Pendiente
│   ├── notebooks/
│   │   ├── F2_01_obtencion.ipynb
│   │   ├── F2_02_exploracion.ipynb              # Pendiente
│   │   └── F2_03_procesamiento_validacion.ipynb # Pendiente
│   ├── src/
│   │   ├── 01_extraer_votaciones_proyecto.py
│   │   ├── 02_periodos_legislativos.py
│   │   ├── 03_diputados.py
│   │   ├── 04_extraer_detalles_votaciones.py
│   │   └── 06_integracion.py                    # Pendiente
│   └── test/                  # Pendiente
├── F3/                # Por definir
├── F4/                # Por definir
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

Con `uv` no es necesario activar el entorno manualmente: `uv run` ejecuta cualquier comando dentro del `.venv` que crea `uv sync`.

```bash
uv run jupyter lab
uv run pytest
uv run ruff check .
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
- **Testing**: pytest
