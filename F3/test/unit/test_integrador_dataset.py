from pathlib import Path

import pandas as pd

from F3.src.nucleo.integrador_dataset import construir_big_table

RAIZ_REPOSITORIO = (
    Path(__file__).resolve().parents[3]
)
F3_PROCESSED = (
    RAIZ_REPOSITORIO / "F3" / "data" / "processed"
)
F2_PROCESSED = (
    RAIZ_REPOSITORIO / "F2" / "data" / "processed"
)


def test_big_table_f3_reproduce_la_integracion_f2():
    detalle = pd.read_csv(
        F3_PROCESSED / "detalle_votaciones_procesado.csv",
        dtype={
            "diputado_id": "string",
            "votacion_id": "string",
        },
    )
    proyecto = pd.read_csv(
        F3_PROCESSED / "proyecto_ley_procesado.csv",
        dtype={"Id": "string"},
    )
    diputados = pd.read_csv(
        F3_PROCESSED / "diputados_procesados.csv",
        dtype={"diputado_id": "string"},
    )
    militancias = pd.read_csv(
        F3_PROCESSED / "militancias_analiticas.csv",
        dtype={
            "diputado_id": "string",
            "partido_id": "string",
        },
    )

    big_table_f3, diagnosticos = construir_big_table(
        detalle,
        proyecto,
        diputados,
        militancias,
    )

    assert big_table_f3.shape == (1993, 34)
    assert len(diagnosticos) == 1993
    assert diagnosticos["estado"].eq("OK").all()
    assert not big_table_f3.duplicated(
        ["votacion_id", "diputado_id"]
    ).any()

    # Comparamos los valores tal como quedarían en ambos CSV.
    # Esto evita diferencias irrelevantes de tipos al leerlos.
    big_table_f2 = pd.read_csv(
        F2_PROCESSED / "big_table_analitica.csv",
        dtype="string",
    )

    big_table_f3_comparable = (
        big_table_f3
        .astype("string")
        .fillna("<NULO>")
    )
    big_table_f2_comparable = (
        big_table_f2
        .fillna("<NULO>")
    )

    pd.testing.assert_frame_equal(
        big_table_f3_comparable,
        big_table_f2_comparable,
    )