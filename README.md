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
|------|-------------|--------|
| **F1** | Planteamiento del problema y diseño del proyecto | ✅ Completa |
| **F2** | Obtención, limpieza y procesamiento de datos | 🚧 Estructura creada, sin datos aún |
| **F3** | Análisis exploratorio / modelamiento | ⬜ Pendiente |
| **F4** | Resultados finales y entrega | ⬜ Pendiente |

## Estructura del repositorio

```text
.
├── F1/
│   ├── docs/          # Documentos de planteamiento (informe, diagrama, papers de referencia)
│   └── notebooks/
├── F2/
│   ├── data/
│   │   ├── raw/       # Datos sin procesar
│   │   ├── interim/   # Datos intermedios
│   │   └── processed/ # Datos listos para análisis
│   ├── docs/
│   ├── notebooks/
│   ├── src/
│   └── test/
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

Activar el entorno virtual:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

## Herramientas principales

- **Análisis**: pandas, numpy, matplotlib, seaborn
- **Notebooks**: JupyterLab
- **Calidad de código**: ruff
- **Testing**: pytest
