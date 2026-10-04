"""Ejecuta los notebooks F4_01 a F4_05 en una copia temporal del repositorio.

Cada notebook compara lo que recalcula con las salidas versionadas (assert) y termina
imprimiendo F4_0X_APROBADO=True. Se ejecutan en una copia para no modificar el árbol de
trabajo: F4_02 y F4_04 reescriben salidas de módulos y todos guardan figuras. Al final se
comprueba que esas salidas reescritas sigan idénticas a las versionadas.
"""

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[3]
NOTEBOOKS = [
    "F4_01_datos_y_cobertura.ipynb",
    "F4_02_bcall.ipynb",
    "F4_03_metricas_partidos_afinidad.ipynb",
    "F4_04_clustering_pca.ipynb",
    "F4_05_cohesion.ipynb",
]
IGNORAR = shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints")


@pytest.fixture(scope="module")
def copia(tmp_path_factory) -> Path:
    raiz = tmp_path_factory.mktemp("repo")
    shutil.copy2(RAIZ / "pyproject.toml", raiz / "pyproject.toml")
    shutil.copytree(RAIZ / "F3/data", raiz / "F3/data", ignore=IGNORAR)
    for carpeta in ("config", "data", "docs", "notebooks", "src"):
        shutil.copytree(RAIZ / "F4" / carpeta, raiz / "F4" / carpeta, ignore=IGNORAR)
    return raiz


def ejecutar_notebook(ruta: Path) -> str:
    notebook = nbformat.read(ruta, as_version=4)
    NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ruta.parent)}},
    ).execute()
    return "\n".join(
        str(salida.get("text", ""))
        for celda in notebook.cells
        for salida in celda.get("outputs", [])
        if salida.get("output_type") == "stream"
    )


@pytest.mark.slow
@pytest.mark.parametrize("nombre", NOTEBOOKS)
def test_notebook_ejecuta_y_aprueba(copia: Path, nombre: str) -> None:
    evidencia = ejecutar_notebook(copia / "F4/notebooks" / nombre)
    indicador = f"{nombre[:5]}_APROBADO=True"
    assert indicador in evidencia, f"{nombre} terminó sin emitir {indicador}"


@pytest.mark.slow
def test_notebooks_no_alteran_las_salidas_versionadas(copia: Path) -> None:
    # Después de los cinco notebooks (los JSON de ejecución llevan fecha y commit).
    distintas = [
        ruta.relative_to(copia).as_posix()
        for ruta in sorted((copia / "F4/data").rglob("*.csv"))
        if ruta.read_bytes() != (RAIZ / ruta.relative_to(copia)).read_bytes()
    ]
    assert not distintas, f"Los notebooks cambiaron salidas versionadas: {distintas}"
