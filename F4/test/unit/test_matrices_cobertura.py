from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis.matrices import (
    ConstructorMatrices,
    ErrorMatrices,
    validar_contra_auditoria,
)


RAIZ = Path(__file__).resolve().parents[3]


def _tabla_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diputado_id": [10, 10, 11, 11],
            "votacion_id": [100, 101, 100, 101],
            "fecha": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "partido_id": [1, 2, 3, pd.NA],
            "partido_nombre": ["Partido A", "Partido B", "Independientes", pd.NA],
            "partido_alias": ["PA", "PB", "IND", pd.NA],
            "voto_binario": [1, pd.NA, 0, 1],
            "voto_ternario": [1, 0, -1, 1],
            "voto_nominal": ["Sí", "Abstención", "No", "Sí"],
            "observado": [True, True, True, True],
        }
    )


def test_construye_tres_matrices_y_mascara() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    assert resultado.binaria.shape == (2, 3)
    assert resultado.ternaria.shape == (2, 3)
    assert resultado.nominal.shape == (2, 3)
    assert resultado.mascara.shape == (2, 3)

    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")
    nominal = resultado.nominal.set_index("diputado_id")

    assert binaria.loc[10, 100] == 1
    assert pd.isna(binaria.loc[10, 101])
    assert ternaria.loc[10, 101] == 0
    assert nominal.loc[10, 101] == "Abstención"


def test_fila_faltante_no_se_interpreta_como_ausencia() -> None:
    tabla = _tabla_base().drop(index=3).reset_index(drop=True)
    resultado = ConstructorMatrices().construir(tabla)

    mascara = resultado.mascara.set_index("diputado_id")
    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")
    nominal = resultado.nominal.set_index("diputado_id")

    assert not bool(mascara.loc[11, 101])
    assert pd.isna(binaria.loc[11, 101])
    assert pd.isna(ternaria.loc[11, 101])
    assert pd.isna(nominal.loc[11, 101])


def test_abstencion_es_observada_y_ternaria_cero() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    mascara = resultado.mascara.set_index("diputado_id")
    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")

    assert bool(mascara.loc[10, 101])
    assert pd.isna(binaria.loc[10, 101])
    assert ternaria.loc[10, 101] == 0


def test_rechaza_clave_duplicada() -> None:
    tabla = pd.concat(
        [_tabla_base(), _tabla_base().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ErrorMatrices, match="debe ser única"):
        ConstructorMatrices().construir(tabla)


def test_afiliacion_extrae_solo_filas_existentes() -> None:
    tabla = _tabla_base().drop(index=3).reset_index(drop=True)
    resultado = ConstructorMatrices().construir(tabla)

    assert len(resultado.afiliacion) == len(tabla)

    clave = resultado.afiliacion[["diputado_id", "votacion_id"]]
    assert not clave.duplicated().any()

    # No se inventa la combinación 11 × 101 eliminada de la entrada.
    assert not (
        resultado.afiliacion["diputado_id"].eq(11)
        & resultado.afiliacion["votacion_id"].eq(101)
    ).any()


def test_afiliacion_conserva_cambio_historico() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    diputado = resultado.afiliacion[
        resultado.afiliacion["diputado_id"].eq(10)
    ].sort_values("votacion_id")

    assert diputado["partido_id"].tolist() == [1, 2]
    assert diputado["partido_nombre"].tolist() == ["Partido A", "Partido B"]


def test_afiliacion_conserva_nulos_e_independientes() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    independiente = resultado.afiliacion[
        (resultado.afiliacion["diputado_id"] == 11)
        & (resultado.afiliacion["votacion_id"] == 100)
    ].iloc[0]
    desconocida = resultado.afiliacion[
        (resultado.afiliacion["diputado_id"] == 11)
        & (resultado.afiliacion["votacion_id"] == 101)
    ].iloc[0]

    assert independiente["partido_nombre"] == "Independientes"
    assert pd.isna(desconocida["partido_id"])
    assert pd.isna(desconocida["partido_nombre"])
    assert pd.isna(desconocida["partido_alias"])


def test_corte_real_conserva_conteos_y_afiliaciones() -> None:
    votos = pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv")
    resultado = ConstructorMatrices().construir(votos)

    assert resultado.n_diputados == 151
    assert resultado.n_votaciones == 15
    assert resultado.n_decisiones_observadas == 1993
    assert resultado.n_celdas_sin_registro == 272

    columnas = [c for c in resultado.mascara.columns if c != "diputado_id"]

    assert int(resultado.binaria[columnas].notna().sum().sum()) == 1925
    assert int(resultado.ternaria[columnas].notna().sum().sum()) == 1993
    assert int(resultado.nominal[columnas].notna().sum().sum()) == 1993

    assert len(resultado.afiliacion) == len(votos)
    assert not resultado.afiliacion.duplicated(
        ["diputado_id", "votacion_id"]
    ).any()


def test_concilia_con_auditoria_real() -> None:
    votos = pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv")

    validar_contra_auditoria(
        votos,
        RAIZ / "F4/data/reports/auditoria_entrada.json",
    )