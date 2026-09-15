from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

F2_DIR = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    (
        F2_DIR / "notebooks" / "F2_03_procesamiento_validacion.ipynb",
        "F2_03_APROBADO=True",
    ),
    (
        F2_DIR / "notebooks" / "F2_04_Integración.ipynb",
        "F2_04_APROBADO=True",
    ),
]


def ejecutar_notebook(ruta: Path) -> str:
    if not ruta.exists():
        pytest.fail(f"No existe el notebook requerido: {ruta}")

    notebook = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(F2_DIR / "notebooks")}},
    )
    ejecutado = cliente.execute()

    salidas = []
    for celda in ejecutado.cells:
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "stream":
                salidas.append(str(salida.get("text", "")))

    return "\n".join(salidas)


@pytest.mark.slow
def test_notebooks_f2_03_y_f2_04_ejecutan_y_aprueban():
    for ruta, indicador in NOTEBOOKS:
        evidencia = ejecutar_notebook(ruta)
        assert indicador in evidencia, (
            f"{ruta.name} terminó sin emitir el indicador estable esperado: {indicador}"
        )