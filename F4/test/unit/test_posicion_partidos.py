import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.posicion_partidos import (
    ErrorPosicionPartido,
    PosicionPartido,
)


def _ejemplo():
    # Votos ya estandarizados y orientados del ejemplo guía del anexo.
    u = pd.DataFrame(
        {
            "j1": [-0.816, -0.816, -0.816, 1.225, 1.225],
            "j3": [-1.225, -1.225, 0.816, 0.816, 0.816],
            "j4": [-1.508, -0.302, np.nan, 0.905, 0.905],
        },
        index=["A", "B", "C", "D", "E"],
    )

    afiliacion = pd.DataFrame(
        [
            ["A", "j1", "2026-01-01", "p1", "Partido 1", "P1"],
            ["B", "j1", "2026-01-01", "p1", "Partido 1", "P1"],
            ["C", "j1", "2026-01-01", "p1", "Partido 1", "P1"],
            ["D", "j1", "2026-01-01", "p2", "Partido 2", "P2"],
            ["E", "j1", "2026-01-01", "p2", "Partido 2", "P2"],
            ["A", "j3", "2026-01-03", "p1", "Partido 1", "P1"],
            ["B", "j3", "2026-01-03", "p1", "Partido 1", "P1"],
            ["C", "j3", "2026-01-03", "p1", "Partido 1", "P1"],
            ["D", "j3", "2026-01-03", "p2", "Partido 2", "P2"],
            ["E", "j3", "2026-01-03", "p2", "Partido 2", "P2"],
            ["A", "j4", "2026-01-04", "p1", "Partido 1", "P1"],
            ["B", "j4", "2026-01-04", "p1", "Partido 1", "P1"],
            ["D", "j4", "2026-01-04", "p2", "Partido 2", "P2"],
            ["E", "j4", "2026-01-04", "p2", "Partido 2", "P2"],
        ],
        columns=[
            "diputado_id",
            "votacion_id",
            "fecha",
            "partido_id",
            "partido_nombre",
            "partido_alias",
        ],
    )

    diputados = pd.DataFrame(
        {
            "d1": [-1.183, -0.781, 0.0, 0.982, 0.982],
            "d2": [0.284, 0.378, 0.816, 0.175, 0.175],
        },
        index=["A", "B", "C", "D", "E"],
    )
    return u, afiliacion, diputados


def test_promedia_integrantes_y_luego_votaciones() -> None:
    u, afiliacion, diputados = _ejemplo()
    r = PosicionPartido().calcular(u, afiliacion, diputados)

    p1 = r.por_votacion[r.por_votacion["partido_id"].eq("p1")].set_index(
        "votacion_id"
    )

    assert p1.loc["j1", "b_pj"] == pytest.approx(-0.816, abs=1e-3)
    assert p1.loc["j3", "b_pj"] == pytest.approx((-1.225 - 1.225 + 0.816) / 3)
    assert p1.loc["j4", "b_pj"] == pytest.approx((-1.508 - 0.302) / 2)

    perfil = r.resumen.set_index("partido_id")
    esperado = p1.loc[["j1", "j3", "j4"], "b_pj"].mean()
    assert perfil.loc["p1", "P_p"] == pytest.approx(esperado)


def test_cada_votacion_pesa_igual_no_cada_persona() -> None:
    u, afiliacion, diputados = _ejemplo()
    r = PosicionPartido().calcular(u, afiliacion, diputados)
    perfil = r.resumen.set_index("partido_id")

    media_simple_d1 = diputados.loc[["A", "B", "C"], "d1"].mean()

    assert perfil.loc["p1", "P_p"] != pytest.approx(media_simple_d1)


def test_cambio_de_militancia_se_respeta_por_votacion() -> None:
    u, afiliacion, diputados = _ejemplo()

    # B cambia de p1 a p2 solo en j4.
    mascara = (
        afiliacion["diputado_id"].eq("B")
        & afiliacion["votacion_id"].eq("j4")
    )
    afiliacion.loc[mascara, ["partido_id", "partido_nombre", "partido_alias"]] = [
        "p2",
        "Partido 2",
        "P2",
    ]

    r = PosicionPartido(
        min_decisiones_partido_votacion=1,
        min_votaciones_partido=1,
    ).calcular(u, afiliacion, diputados)

    j4 = r.por_votacion[r.por_votacion["votacion_id"].eq("j4")]
    p1 = j4[j4["partido_id"].eq("p1")].iloc[0]
    p2 = j4[j4["partido_id"].eq("p2")].iloc[0]

    assert p1["n_decisiones"] == 1
    assert p1["b_pj"] == pytest.approx(-1.508)
    assert p2["n_decisiones"] == 3
    assert p2["b_pj"] == pytest.approx((-0.302 + 0.905 + 0.905) / 3)


def test_partido_de_un_integrante_queda_descriptivo_si_minimo_es_dos() -> None:
    u, afiliacion, diputados = _ejemplo()

    nueva = pd.DataFrame(
        [["A", "j1", "2026-01-01", "p3", "Partido 3", "P3"]],
        columns=afiliacion.columns,
    )
    afiliacion = pd.concat(
        [
            afiliacion[
                ~(
                    afiliacion["diputado_id"].eq("A")
                    & afiliacion["votacion_id"].eq("j1")
                )
            ],
            nueva,
        ],
        ignore_index=True,
    )

    r = PosicionPartido().calcular(u, afiliacion, diputados)
    fila = r.por_votacion[
        r.por_votacion["partido_id"].eq("p3")
    ].iloc[0]

    assert fila["n_decisiones"] == 1
    assert not bool(fila["incluido"])
    assert fila["razon_exclusion"] == "DECISIONES_INSUFICIENTES_PARTIDO_VOTACION"


def test_afiliacion_desconocida_no_se_asigna_a_partido() -> None:
    u, afiliacion, diputados = _ejemplo()
    mascara = (
        afiliacion["diputado_id"].eq("C")
        & afiliacion["votacion_id"].eq("j3")
    )
    afiliacion.loc[
        mascara, ["partido_id", "partido_nombre", "partido_alias"]
    ] = pd.NA

    r = PosicionPartido().calcular(u, afiliacion, diputados)

    fila = r.por_votacion[
        r.por_votacion["partido_id"].eq("p1")
        & r.por_votacion["votacion_id"].eq("j3")
    ].iloc[0]

    assert fila["n_decisiones"] == 2
    assert fila["b_pj"] == pytest.approx((-1.225 - 1.225) / 2)


def test_heterogeneidad_usa_diputados_unicos_por_partido() -> None:
    u, afiliacion, diputados = _ejemplo()
    r = PosicionPartido().calcular(u, afiliacion, diputados)
    perfil = r.resumen.set_index("partido_id")

    esperado = diputados.loc[["A", "B", "C"], "d1"]
    assert perfil.loc["p1", "mediana_d1"] == pytest.approx(esperado.median())
    assert perfil.loc["p1", "iqr_d1"] == pytest.approx(
        esperado.quantile(0.75) - esperado.quantile(0.25)
    )
    assert perfil.loc["p1", "n_diputados"] == 3


def test_falta_afiliacion_para_un_voto_orientado_es_error() -> None:
    u, afiliacion, diputados = _ejemplo()
    afiliacion = afiliacion[
        ~(
            afiliacion["diputado_id"].eq("A")
            & afiliacion["votacion_id"].eq("j1")
        )
    ]

    with pytest.raises(ErrorPosicionPartido, match="Faltan filas de afiliación"):
        PosicionPartido().calcular(u, afiliacion, diputados)


def test_afiliacion_duplicada_es_error() -> None:
    u, afiliacion, diputados = _ejemplo()
    afiliacion = pd.concat([afiliacion, afiliacion.iloc[[0]]], ignore_index=True)

    with pytest.raises(ErrorPosicionPartido, match="fila única"):
        PosicionPartido().calcular(u, afiliacion, diputados)

def test_independientes_se_calculan_pero_no_se_publican() -> None:
    # p2 hace de IND (grupo no partidario); su P_p se calcula como el de cualquier grupo.
    u, afiliacion, diputados = _ejemplo()
    r = PosicionPartido(grupos_no_partidarios=("p2",)).calcular(u, afiliacion, diputados)
    perfil = r.resumen.set_index("partido_id")

    assert perfil.loc["p2", "P_p"] == pytest.approx(
        PosicionPartido().calcular(u, afiliacion, diputados)
        .resumen.set_index("partido_id").loc["p2", "P_p"])
    assert perfil.loc["p2", "tipo_grupo"] == "independientes"
    assert not perfil.loc["p2", "publicable"]
    assert perfil.loc["p2", "razon_no_publicable"] == "GRUPO_INDEPENDIENTES"
    assert perfil.loc["p1", "publicable"] and pd.isna(perfil.loc["p1", "razon_no_publicable"])


def test_publicable_exige_perfil_incluido() -> None:
    u, afiliacion, diputados = _ejemplo()
    r = PosicionPartido(min_votaciones_partido=4).calcular(u, afiliacion, diputados)
    perfil = r.resumen.set_index("partido_id")

    assert not perfil["publicable"].any()
    assert set(perfil["razon_no_publicable"]) == {
        "VOTACIONES_INSUFICIENTES_PARA_PERFIL_PARTIDARIO"}
