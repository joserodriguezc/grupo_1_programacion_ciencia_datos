from pathlib import Path

import pandas as pd
import pytest

F2_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = F2_DIR / "data" / "processed"


def cargar_csv(ruta: Path) -> pd.DataFrame:
    """Carga un producto procesado obligatorio para los contratos de datos."""
    if not ruta.exists():
        pytest.fail(f"No existe el producto procesado requerido: {ruta}")

    return pd.read_csv(ruta)


@pytest.fixture
def detalle_procesado() -> pd.DataFrame:
    return cargar_csv(PROCESSED_DIR / "detalle_votaciones_procesado.csv")


@pytest.fixture
def militancias_analiticas() -> pd.DataFrame:
    return cargar_csv(PROCESSED_DIR / "militancias_analiticas.csv")


@pytest.fixture
def reporte_calidad() -> pd.DataFrame:
    return cargar_csv(PROCESSED_DIR / "reporte_calidad.csv")


@pytest.fixture
def big_table_analitica() -> pd.DataFrame:
    return cargar_csv(PROCESSED_DIR / "big_table_analitica.csv")
