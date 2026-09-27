from pathlib import Path

import pandas as pd
import pytest

from F3.src.nucleo.asignadores import (
    asignar_iterativo,
    asignar_vectorizado,
    exigir_asignacion_unica,
    preparar_entradas,
)

RAIZ_F3 = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ_F3 / "data" / "processed"


def _datos_de_prueba():
    votos = pd.DataFrame(
        {
            "fila_voto_id": [
                "inicio",
                "termino",
                "fuera",
                "abierta",
                "ambigua",
                "sin",
            ],
            "diputado_id": ["1", "1", "1", "2", "3", "4"],
            "fecha": [
                "2024-01-01",
                "2024-01-31",
                "2024-02-01",
                "2030-01-01",
                "2024-01-15",
                "2024-01-15",
            ],
        }
    )

    militancias = pd.DataFrame(
        {
            "diputado_id": ["1", "2", "3", "3"],
            "partido_id": ["A", "B", "C", "D"],
            "partido_nombre": [
                "Partido A",
                "Partido B",
                "Partido C",
                "Partido D",
            ],
            "partido_alias": ["A", "B", "C", "D"],
            "fecha_inicio": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-10",
            ],
            "fecha_termino": [
                "2024-01-31",
                None,
                "2024-01-31",
                None,
            ],
        }
    )

    return preparar_entradas(votos, militancias)


@pytest.mark.parametrize(
    "asignar",
    [asignar_iterativo, asignar_vectorizado],
)
def test_limites_vigencia_abierta_y_ambiguedad(asignar):
    votos, militancias = _datos_de_prueba()

    diagnosticos, asignaciones = asignar(votos, militancias)

    assert diagnosticos["estado"].tolist() == [
        "OK",
        "OK",
        "SIN_MILITANCIA",
        "OK",
        "AMBIGUA",
        "SIN_MILITANCIA",
    ]
    assert diagnosticos["militancias_vigentes"].tolist() == [
        1, 1, 0, 1, 2, 0
    ]
    assert asignaciones["partido_id"].tolist()[:2] == ["A", "A"]
    assert asignaciones.loc[3, "partido_id"] == "B"
    assert asignaciones.loc[
        [2, 4, 5], "partido_id"
    ].isna().all()


def test_integracion_rechaza_diagnosticos_invalidos():
    votos, militancias = _datos_de_prueba()
    diagnosticos, _ = asignar_iterativo(votos, militancias)

    with pytest.raises(
        ValueError,
        match="3 votos no tienen exactamente",
    ):
        exigir_asignacion_unica(diagnosticos)


def test_ambos_asignadores_coinciden_en_los_1993_votos_de_f3():
    votos = pd.read_csv(
        PROCESSED / "detalle_votaciones_procesado.csv",
        dtype={
            "diputado_id": "string",
            "votacion_id": "string",
        },
    )
    militancias = pd.read_csv(
        PROCESSED / "militancias_analiticas.csv",
        dtype={
            "diputado_id": "string",
            "partido_id": "string",
        },
    )

    votos, militancias = preparar_entradas(votos, militancias)

    diagnostico_iterativo, asignacion_iterativa = (
        asignar_iterativo(votos, militancias)
    )
    diagnostico_vectorizado, asignacion_vectorizada = (
        asignar_vectorizado(votos, militancias)
    )

    pd.testing.assert_frame_equal(
        diagnostico_iterativo.reset_index(drop=True),
        diagnostico_vectorizado.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        asignacion_iterativa.reset_index(drop=True),
        asignacion_vectorizada.reset_index(drop=True),
        check_dtype=False,
    )

    assert len(diagnostico_iterativo) == 1993
    assert diagnostico_iterativo["estado"].eq("OK").all()
    assert asignacion_iterativa["partido_id"].notna().all()

    exigir_asignacion_unica(diagnostico_iterativo)
    exigir_asignacion_unica(diagnostico_vectorizado)