"""Chequeo estatico: ningun notebook debe tener celdas de codigo vacias
ni celdas con contenido que no se hayan ejecutado antes de comitear.
No re-ejecuta los notebooks (eso ya lo hace F2/test/notebooks/test_notebooks_ejecutan.py
para F2_03 y F2_04); por eso corre rapido y cubre TODOS los notebooks del repo.
"""
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(REPO_ROOT.glob("*/notebooks/*.ipynb"))


def _fuente(celda) -> str:
    fuente = celda.get("source", "")
    if isinstance(fuente, list):
        fuente = "".join(fuente)
    return fuente.strip()


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_notebook_sin_celdas_vacias(ruta: Path):
    nb = nbformat.read(ruta, as_version=4)
    vacias = [
        i for i, c in enumerate(nb.cells)
        if c.get("cell_type") == "code" and not _fuente(c)
    ]
    assert not vacias, (
        f"{ruta.name} tiene celdas de codigo vacias en las posiciones {vacias}. "
        "Eliminalas antes de comitear."
    )


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_notebook_celdas_ejecutadas(ruta: Path):
    nb = nbformat.read(ruta, as_version=4)
    sin_ejecutar = [
        i for i, c in enumerate(nb.cells)
        if c.get("cell_type") == "code" and _fuente(c) and c.get("execution_count") is None
    ]
    assert not sin_ejecutar, (
        f"{ruta.name} tiene celdas de codigo con contenido pero sin ejecutar "
        f"(execution_count vacio) en las posiciones {sin_ejecutar}. "
        "Ejecuta 'Run All' y vuelve a guardar el notebook antes de comitear."
    )
