import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.bcall import ModeloBCall
from F4.src.analisis.orientacion import (
    clasificacion_partidos,
    cotejar_asignacion,
    elegir_pivote,
)


@pytest.fixture
def datos():
    x = pd.DataFrame({"v1": [-1, -1, 1, 1], "v2": [1, 1, -1, -1]},
                     index=[1, 2, 815, 1100], dtype=float)
    a = pd.DataFrame({"diputado_id": [1, 2, 815, 1100], "votacion_id": ["v1"] * 4,
                      "partido_nombre": ["Izquierda", "Independientes",
                                         "Unión Demócrata Independiente", "Partido Republicano"]})
    j = {"registros": [{"partido_nombre": n, "codigo_binario": c} for n, c in [
        ("Izquierda", 0), ("Independientes", None),
        ("Unión Demócrata Independiente", 1), ("Partido Republicano", 1)]]}
    return x, a, j


def test_auto_oraculo_bloques_sin_json(datos):
    x, _, _ = datos
    r = ModeloBCall().calcular_auto(x, pivot=815)
    assert r.clustering_auto.to_dict() == {1: "left", 2: "left", 815: "right", 1100: "right"}
    np.testing.assert_allclose(r.diputados.d1, [-np.sqrt(.75)]*2+[np.sqrt(.75)]*2)
    np.testing.assert_allclose(r.diputados.d2, 0)


def test_independiente_tiene_cluster_pero_no_etiqueta_externa(datos):
    x, a, j = datos
    r = ModeloBCall().calcular_auto(x, pivot=815)
    cotejo = cotejar_asignacion(r.diputados.grupo_bcall, a, j).set_index("diputado_id")
    assert cotejo.loc[2, "grupo_bcall"] == "L"
    assert pd.isna(cotejo.loc[2, "grupo_externo"])
    assert pd.isna(cotejo.loc[2, "coincide"])
    assert not cotejo.loc[2, "comparable"]
    assert r.diputados.loc[2, "m_i"] == 2


def test_json_no_modifica_clusters_y_divergencia_no_es_error(datos):
    x, a, j = datos
    r = ModeloBCall().calcular_auto(x, pivot=815)
    j["registros"][0]["codigo_binario"] = 1
    cotejo = cotejar_asignacion(r.diputados.grupo_bcall, a, j)
    assert not cotejo.loc[0, "coincide"]
    assert r.diputados.loc[1, "grupo_bcall"] == "L"


def test_elegir_pivote_por_participacion_y_menor_id(datos):
    x, a, j = datos
    assert elegir_pivote(x, a, j) == 815
    x.loc[815, "v2"] = np.nan
    assert elegir_pivote(x, a, j) == 1100


def test_pivote_mismo_grupo_no_cambia_resultado_sin_empates(datos):
    x, _, _ = datos
    a = ModeloBCall().calcular_auto(x, pivot=815)
    b = ModeloBCall().calcular_auto(x, pivot=1100)
    pd.testing.assert_frame_equal(a.diputados, b.diputados)


def test_invertir_pivote_en_bloques_separados_invierte_eje(datos):
    x, _, _ = datos
    a = ModeloBCall().calcular_auto(x, pivot=815)
    b = ModeloBCall().calcular_auto(x, pivot=1)
    np.testing.assert_allclose(a.diputados.d1, -b.diputados.d1)
    np.testing.assert_allclose(a.diputados.d2, b.diputados.d2)


@pytest.mark.parametrize("metodo, esperado", [(1, .75), (2, np.sqrt(5)/np.sqrt(8))])
def test_distancias_normalizadas_y_sin_covotos(metodo, esperado):
    a = np.array([[-1., 0., np.nan]])
    b = np.array([[1., 1., 1.], [np.nan, np.nan, 1.]])
    d = ModeloBCall._distancias(a, b, metodo)
    assert d[0, 0] == pytest.approx(esperado)
    assert d[0, 1] == 0  # comportamiento original; no es ausencia imputada.


def test_filtro_se_aplica_despues_del_clustering(datos):
    x, _, _ = datos
    x.loc[3] = np.nan
    r = ModeloBCall().calcular_auto(x, pivot=815)
    assert 3 in r.clustering_auto.index
    assert 3 not in r.diputados.index
    assert r.seleccion.loc[3, "razon_exclusion"] == "PARTICIPACION_NO_SUPERA_UMBRAL"


def test_empate_reclasificacion_va_a_right():
    x = pd.DataFrame({"v": [-1., 1., 0.]}, index=[1, 815, 2])
    r = ModeloBCall().calcular_auto(x, pivot=815)
    assert r.clustering_auto.loc[2] == "right"


def test_sin_separacion_lanza_error(datos):
    x, _, _ = datos
    with pytest.raises(ValueError, match="separación"):
        ModeloBCall().calcular_auto(x * 0, pivot=815)


@pytest.mark.parametrize("metodo", [0, 3, True])
def test_metodo_invalido(datos, metodo):
    x, _, _ = datos
    with pytest.raises(ValueError):
        ModeloBCall().calcular_auto(x, pivot=815, distance_method=metodo)


def test_afiliacion_por_fecha_sin_forzar_etiqueta_independiente(datos):
    x, a, j = datos
    a = pd.concat([a, pd.DataFrame({"diputado_id": [815], "votacion_id": ["v2"],
                                   "partido_nombre": ["Independientes"]})], ignore_index=True)
    grupos = pd.Series("R", index=x.index)
    c = cotejar_asignacion(grupos, a, j)
    assert pd.isna(c.iloc[-1].codigo_externo)
    assert not c.iloc[-1].comparable
    assert elegir_pivote(x, a, j) == 1100


def test_no_imputar_partido_desconocido(datos):
    x, a, j = datos
    a.loc[0, "partido_nombre"] = "Partido no catalogado"
    c = cotejar_asignacion(pd.Series("R", index=x.index), a, j)
    assert pd.isna(c.loc[0, "codigo_externo"])
    assert not c.loc[0, "comparable"]


def test_binaria_texto_y_codigo_deben_coincidir():
    j = {"registros": [{"partido_nombre": "Partido de la Gente",
                        "binaria_izq_der": "Derecha", "codigo_binario": None}]}
    with pytest.raises(ValueError, match="no coinciden"):
        clasificacion_partidos(j)


def test_referencia_real_es_consistente():
    ruta = Path("F4/docs/clasificacion_ideologica_partidos_chilenos.json")
    mapa = clasificacion_partidos(json.loads(ruta.read_text(encoding="utf-8")))
    assert mapa["Partido de la Gente"] == 1
    assert mapa["Partido Demócratas Chile"] == 1
    assert pd.isna(mapa["Independientes"])
