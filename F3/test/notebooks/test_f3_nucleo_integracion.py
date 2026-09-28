"""Prueba las secciones funcionales de F3_01_nucleo sin benchmark pesado."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
F3_DIR = RAIZ_REPOSITORIO / "F3"
RUTA_NOTEBOOK = F3_DIR / "notebooks" / "F3_01_nucleo.ipynb"


def _extraer_secciones_funcionales(notebook):
    """Construye un notebook temporal con las secciones 5 y 6."""

    celdas = []
    incluir = False

    for celda in notebook.cells:
        if celda.cell_type == "markdown":
            texto = celda.source.strip()

            if texto.startswith("## 5."):
                incluir = True

            if texto.startswith("## 7."):
                break

        if incluir:
            celdas.append(celda)

    if not celdas:
        pytest.fail(
            "No se encontraron las secciones 5 y 6 en "
            "F3_01_nucleo.ipynb"
        )

    preparacion = nbformat.v4.new_code_cell(
        """
from pathlib import Path
import sys

import pandas as pd

RAIZ_PROYECTO = Path.cwd().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))
""".strip()
    )

    return nbformat.v4.new_notebook(
        cells=[preparacion, *celdas],
        metadata=notebook.metadata.copy(),
    )


def _salidas_stream(notebook_ejecutado) -> str:
    salidas = []

    for celda in notebook_ejecutado.cells:
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "stream":
                salidas.append(str(salida.get("text", "")))

    return "\n".join(salidas)


@pytest.mark.slow
def test_f3_nucleo_procesamiento_integracion_y_equivalencia():
    """Ejecuta desde cero la parte funcional y excluye 1x/5x/10x/100x."""

    if not RUTA_NOTEBOOK.exists():
        pytest.fail(f"No existe el notebook requerido: {RUTA_NOTEBOOK}")

    notebook = nbformat.read(RUTA_NOTEBOOK, as_version=4)
    notebook_ci = _extraer_secciones_funcionales(notebook)

    cliente = NotebookClient(
        notebook_ci,
        timeout=300,
        kernel_name="python3",
        resources={
            "metadata": {
                "path": str(F3_DIR / "notebooks"),
            }
        },
    )

    ejecutado = cliente.execute()
    evidencia = _salidas_stream(ejecutado)

    assert (
        "OK: contratos básicos de integración F3 satisfechos."
        in evidencia
    )
    assert (
        "OK: el procesamiento F3 reproduce las cinco tablas "
        "procesadas de F2."
        in evidencia
    )
    assert (
        "OK: la integración F3 reproduce la big table analítica de F2."
        in evidencia
    )
    assert "OK: diagnósticos equivalentes." in evidencia
    assert "OK: asignaciones equivalentes." in evidencia
