"""Casos límite de la regla temporal utilizada por F2 y F3."""

import pandas as pd
import pytest

from F2.src.validaciones import detectar_solapamientos_vigencia


def _militancias(inicio_2: str, termino_1: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diputado_id": [1096, 1096],
            "fecha_inicio": ["2022-03-11 00:00:00", inicio_2],
            "fecha_termino": [termino_1, "2026-03-10 23:59:59"],
        }
    )


def test_termino_igual_a_inicio_es_solapamiento() -> None:
    df = _militancias("2024-03-11 00:00:00", "2024-03-11 00:00:00")

    conflictos = detectar_solapamientos_vigencia(df)

    assert len(conflictos) == 1
    assert conflictos.loc[0, "indice_a"] == 0
    assert conflictos.loc[0, "indice_b"] == 1


def test_periodos_consecutivos_sin_solapamiento() -> None:
    df = _militancias("2024-03-11 00:00:00", "2024-03-10 23:59:59")

    assert detectar_solapamientos_vigencia(df).empty


def test_termino_abierto_solapa_con_inicio_posterior() -> None:
    df = _militancias("2024-03-11 00:00:00", "")

    assert len(detectar_solapamientos_vigencia(df)) == 1


@pytest.mark.parametrize("termino", ["fecha-invalida", "2024-13-11"])
def test_fecha_invalida_no_se_interpreta_como_vigencia_abierta(termino: str) -> None:
    df = _militancias("2024-03-11 00:00:00", termino)

    with pytest.raises(ValueError, match="fecha_termino"):
        detectar_solapamientos_vigencia(df)


def test_inicio_posterior_al_termino_es_error() -> None:
    df = _militancias("2024-03-11 00:00:00", "2020-03-10 23:59:59")

    with pytest.raises(ValueError, match="fecha_termino"):
        detectar_solapamientos_vigencia(df)
