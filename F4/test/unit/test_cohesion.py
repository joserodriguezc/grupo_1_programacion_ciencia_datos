"""Índices de cohesión: valores de referencia y casos límite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.cohesion import (
    ErrorCohesion,
    IndicesCohesion,
    ParametrosCohesion,
    agreement_index,
    cohesion_entropica,
    desde_repositorio,
    rice,
)

RAIZ = Path(__file__).resolve().parents[3]

# Valores de referencia: (Sí, No, Abst.) -> (Rice, AI, C^H), a tres decimales.
VALORES_REFERENCIA = [
    ((10, 0, 0), (1.000, 1.000, 1.000)),
    ((8, 1, 1), (0.778, 0.700, 0.418)),
    ((5, 5, 0), (0.000, 0.250, 0.369)),
    ((1, 0, 1), (1.000, 0.250, 0.369)),
    ((1, 2, 0), (0.333, 0.500, 0.421)),
    ((3, 3, 3), (0.000, 0.000, 0.000)),
    ((0, 0, 3), (np.nan, 1.000, 1.000)),
]


@pytest.mark.parametrize(("conteo", "esperado"), VALORES_REFERENCIA)
def test_valores_de_referencia(conteo, esperado):
    y, n, a = ([v] for v in conteo)
    obtenido = (rice(y, n)[0], agreement_index(y, n, a)[0], cohesion_entropica(y, n, a)[0])
    np.testing.assert_allclose(obtenido, esperado, atol=5e-4, equal_nan=True)


def test_t_cero_es_na_en_los_tres_indices():
    assert np.isnan([rice([0], [0])[0], agreement_index([0], [0], [0])[0],
                     cohesion_entropica([0], [0], [0])[0]]).all()


@pytest.mark.parametrize("conteo", [(8, 1, 1), (1, 2, 0), (5, 5, 0)])
def test_permutar_categorias_no_cambia_ai_ni_entropia(conteo):
    base = [[c] for c in conteo]
    for perm in [(1, 0, 2), (2, 1, 0), (0, 2, 1)]:
        otro = [base[i] for i in perm]
        assert agreement_index(*otro)[0] == pytest.approx(agreement_index(*base)[0])
        assert cohesion_entropica(*otro)[0] == pytest.approx(cohesion_entropica(*base)[0])


# Ejemplo de referencia: A, B, C en p1; D, E en p2; C sin voto en j4; B se abstiene en j4.
GUIA = {"A": ("p1", ["No", "Sí", "No", "Sí"]), "B": ("p1", ["No", "Sí", "No", "Abstención"]),
        "C": ("p1", ["No", "Sí", "Sí", None]), "D": ("p2", ["Sí", "Sí", "Sí", "No"]),
        "E": ("p2", ["Sí", "Sí", "Sí", "No"])}
FECHAS = {"j1": "2023-01-01 10:00:00", "j2": "2023-01-01 11:00:00",
          "j3": "2023-01-01 12:00:00", "j4": "2023-01-02 10:00:00"}


def matrices_guia(partidos: dict | None = None):
    """Misma forma que matriz_nominal.csv y afiliacion_por_votacion.csv."""
    nominal = pd.DataFrame([{"diputado_id": d, **dict(zip(FECHAS, v))}
                            for d, (_, v) in GUIA.items()], dtype="string")
    afil = pd.DataFrame([{"diputado_id": d, "votacion_id": j, "fecha": f,
                          "partido_id": (partidos or {}).get(d, p),
                          "partido_nombre": p, "partido_alias": (partidos or {}).get(d, p)}
                         for d, (p, v) in GUIA.items() for (j, f), x in zip(FECHAS.items(), v)
                         if x is not None], dtype="string")
    return nominal, afil


@pytest.fixture
def guia():
    modelo = IndicesCohesion()
    return modelo.calcular(modelo.conteos(IndicesCohesion.votos_largos(*matrices_guia())))


def test_ejemplo_de_referencia_p1_por_votacion(guia):
    p1 = guia[guia.partido_id == "p1"].set_index("votacion_id")
    np.testing.assert_allclose(p1["agreement_index"], [1, 1, 0.5, 0.25], atol=5e-4)
    np.testing.assert_allclose(p1["rice"], [1, 1, 0.333, 1], atol=5e-4)
    np.testing.assert_allclose(p1["cohesion_entropica"], [1, 1, 0.421, 0.369], atol=5e-4)
    assert p1.loc["j4", ["Y", "N", "A", "T"]].tolist() == [1, 0, 1, 2]  # C ausente no cuenta


def test_cada_fila_muestra_y_n_a_t(guia):
    assert (guia["T"] == guia[["Y", "N", "A"]].sum(axis=1)).all()
    assert int(guia["T"].sum()) == 19  # 20 celdas menos la de C en j4


def test_rice_no_publicable_con_un_solo_voto_binario(guia):
    fila = guia.set_index(["partido_id", "votacion_id"]).loc[("p1", "j4")]
    assert fila["rice"] == 1 and not fila["publicable_rice"]  # Rice = 1 con un solo voto
    assert fila["publicable_ai_entropia"]  # T = 2 alcanza el mínimo
    assert fila["razon_no_publicable"] == "binarios_bajo_minimo_rice"


def test_partido_con_un_integrante_no_se_publica():
    modelo = IndicesCohesion()
    nominal, afil = matrices_guia({"C": "p3"})
    t = modelo.calcular(modelo.conteos(IndicesCohesion.votos_largos(nominal, afil)))
    p3 = t[t.partido_id == "p3"]
    assert (p3["agreement_index"] == 1).all()  # se calcula
    assert not p3["publicable_ai_entropia"].any()
    assert set(p3["razon_no_publicable"]) == {"T_bajo_minimo"}


def test_independientes_se_calculan_pero_no_como_cohesion_partidaria():
    modelo = IndicesCohesion()
    nominal, afil = matrices_guia({"D": "IND", "E": "IND"})
    t = modelo.calcular(modelo.conteos(IndicesCohesion.votos_largos(nominal, afil)))
    ind = t[t.partido_id == "IND"]
    assert set(ind["tipo_grupo"]) == {"independientes"}
    assert ind["agreement_index"].notna().all()
    assert not ind["publicable_ai_entropia"].any() and not ind["publicable_rice"].any()


def test_votacion_unanime_se_conserva_y_se_marca(guia):
    unanime = guia[guia.votacion_id == "j2"]
    assert unanime["votacion_unanime_sala"].all() and (unanime["agreement_index"] == 1).all()
    assert not guia[guia.votacion_id != "j2"]["votacion_unanime_sala"].any()


def test_razon_na_explicita_y_nunca_cero():
    t = IndicesCohesion().calcular(pd.DataFrame({
        "partido_id": ["p"], "partido_alias": ["p"], "votacion_id": ["j"],
        "fecha": ["2023-01-01"], "Y": [0], "N": [0], "A": [3], "T": [3]}))
    assert pd.isna(t.loc[0, "rice"]) and t.loc[0, "razon_na_rice"] == "sin_si_ni_no"
    assert t.loc[0, "agreement_index"] == 1 and pd.isna(t.loc[0, "razon_na_ai_entropia"])


def test_voto_sin_afiliacion_es_error():
    nominal, afil = matrices_guia()
    with pytest.raises(ErrorCohesion):
        IndicesCohesion.votos_largos(nominal, afil.iloc[1:])


def test_parametros_desde_analisis_toml():
    p = ParametrosCohesion.desde_toml(RAIZ / "F4/config/analisis.toml")
    assert (p.min_decisiones_publicacion, p.min_binarios_publicacion_rice,
            p.categorias_entropia) == (2, 2, 3)


def test_corte_real_conserva_todas_las_decisiones():
    t = desde_repositorio(RAIZ)
    assert int(t["T"].sum()) == 1993
    assert not t.duplicated(["partido_id", "votacion_id"]).any()
    assert set(t.loc[t["votacion_unanime_sala"], "votacion_id"]) == {"20627", "20628"}
    assert t[["agreement_index", "cohesion_entropica"]].notna().all(axis=None)
