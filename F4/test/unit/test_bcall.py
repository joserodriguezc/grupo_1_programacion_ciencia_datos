"""Oráculo manual y reglas del núcleo bcall() R; no requiere R en ejecución."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from F4.src.analisis.bcall import ModeloBCall


@pytest.fixture
def ejemplo():
    x = pd.DataFrame(
        {"j1": [-1, -1, -1, 1, 1], "j2": [1, 1, 1, 1, 1],
         "j3": [-1, -1, 1, 1, 1], "j4": [1, 0, np.nan, -1, -1]},
        index=list("ABCDE"), dtype=float,
    )
    return x, pd.Series(["L", "L", "L", "R", "R"], index=x.index)


def run(x, grupos, **kwargs):
    return ModeloBCall().calcular(x, grupos, pivot="D", **kwargs)


def test_oraculo_muestral_con_todos_asignados(ejemplo):
    x, g = ejemplo
    r = run(x, g)
    # Cinco personas: S=sqrt(1.2) en j1/j3. Cuatro: S=sqrt(11/12) en j4.
    s, t = np.sqrt(1.2), np.sqrt(11 / 12)
    manual = np.array([
        [-.8/s, -1.2/s, -1.25/t], [-.8/s, -1.2/s, -.25/t],
        [-.8/s, .8/s, np.nan], [1.2/s, .8/s, .75/t], [1.2/s, .8/s, .75/t],
    ])
    np.testing.assert_allclose(r.votos_orientados, manual)
    np.testing.assert_allclose(r.diputados.d1, np.nanmean(manual, axis=1))
    np.testing.assert_allclose(r.diputados.d2, np.nanstd(manual, axis=1, ddof=1))
    assert r.diputados.m_i.tolist() == [3, 3, 2, 3, 3]
    assert r.diputados.loc["C", "votaciones_usadas"] == ["j1", "j3"]
    assert "j2" not in r.votos_orientados


def test_empate_tiene_direccion_menos_uno_y_se_conserva(ejemplo):
    x, g = ejemplo
    # L=(-1,0,1), R=(-1,1): ambas medias=0; varianza positiva.
    x["empate"] = [-1, 0, 1, -1, 1]
    r = run(x, g)
    assert r.votaciones.loc["empate", "orientacion"] == -1
    assert "empate" in r.votos_orientados
    np.testing.assert_allclose(r.votos_orientados.empate, -x.empate/x.empate.std(ddof=1))


def test_filtro_estricto_diez_por_ciento_y_antes_de_medias():
    # X=1/10, Y=2/10; con >10% solo X se excluye.
    x = pd.DataFrame(np.tile([-1., 1., 1., 1.], (10, 1)).T,
                     index=["A", "D", "X", "Y"])
    x.loc["X", 1:] = np.nan
    x.loc["Y", 2:] = np.nan
    g = pd.Series(["L", "R", "L", "L"], index=x.index)
    r = run(x, g)
    assert r.seleccion.loc["X", "participacion"] == .1
    assert not r.seleccion.loc["X", "incluido"]
    assert r.seleccion.loc["Y", "incluido"]
    assert r.votaciones.loc[0, "n_observados"] == 3
    assert r.votaciones.loc[0, "media"] == pytest.approx(1/3)
    assert r.votaciones.loc[0, "media_L"] == 0
    assert x[0].mean() == .5  # demuestra que el excluido no contribuye.


def test_default_equivale_a_threshold_punto_uno(ejemplo):
    x, g = ejemplo
    assert_frame_equal(run(x, g).diputados, run(x, g, threshold=.1).diputados)


def test_abstencion_cuenta_como_participacion_y_ausencia_no(ejemplo):
    x, g = ejemplo
    r = run(x, g)
    assert r.seleccion.loc["B", "participacion"] == 1
    assert r.seleccion.loc["C", "participacion"] == .75
    assert pd.isna(r.votos_orientados.loc["C", "j4"])
    assert pd.notna(r.votos_orientados.loc["B", "j4"])


def test_grupo_sin_observaciones_propaga_nan(ejemplo):
    x, g = ejemplo
    x.loc[["D", "E"], "j3"] = np.nan
    r = run(x, g)
    assert r.votaciones.loc["j3", "incluida_por_varianza"]
    assert pd.isna(r.votaciones.loc["j3", "orientacion"])
    assert r.votos_orientados.j3.isna().all()


def test_una_votacion_d2_no_estimable(ejemplo):
    x, g = ejemplo
    r = run(x[["j1"]], g)
    assert r.diputados.d1.notna().all()
    assert r.diputados.d2.isna().all()


def test_sin_varianza_lanza_error_como_R(ejemplo):
    x, g = ejemplo
    with pytest.raises(ValueError, match="varianza"):
        run(x[["j2"]], g)


def test_pivot_excluido_lanza_error(ejemplo):
    x, g = ejemplo
    x.loc["D"] = np.nan
    with pytest.raises(ValueError, match="pivote"):
        run(x, g)


def test_grupo_opuesto_excluido_lanza_error(ejemplo):
    x, g = ejemplo
    x.loc[["A", "B", "C"]] = np.nan
    with pytest.raises(ValueError, match="otro grupo"):
        run(x, g)


def test_persona_sin_cluster_se_excluye_antes_del_calculo(ejemplo):
    x, g = ejemplo
    r = run(x, g.drop("C"))
    assert r.seleccion.loc["C", "razon_exclusion"] == "SIN_CLUSTER"
    assert r.votaciones.loc["j1", "n_observados"] == 4
    assert r.votaciones.loc["j1", "media"] == 0


def test_dataframe_cluster_y_nullable(ejemplo):
    x, g = ejemplo
    a = run(x, g)
    b = run(x.astype("Int64"), g.to_frame("cluster"))
    assert_frame_equal(a.diputados, b.diputados)


def test_no_modifica_entradas_y_es_repetible(ejemplo):
    x, g = ejemplo
    copia = x.copy(deep=True)
    a, b = run(x, g), run(x, g)
    assert_frame_equal(x, copia)
    assert_frame_equal(a.diputados, b.diputados)
    assert_frame_equal(a.diputados.sort_index(), run(x.iloc[::-1], g).diputados.sort_index())


@pytest.mark.parametrize("v", [2, -2, np.inf, -np.inf, "1", True, 1j])
def test_valores_invalidos(ejemplo, v):
    x, g = ejemplo
    if isinstance(v, (str, bool, complex)):
        x = x.astype(object)
    x.loc["A", "j1"] = v
    with pytest.raises(ValueError):
        run(x, g)


@pytest.mark.parametrize("t", [-.1, 1.1, np.nan, np.inf, True, "0.1"])
def test_threshold_invalido(ejemplo, t):
    x, g = ejemplo
    with pytest.raises(ValueError):
        run(x, g, threshold=t)


def test_grupos_y_pivot_invalidos(ejemplo):
    x, g = ejemplo
    with pytest.raises(ValueError):
        run(x, pd.Series("L", index=x.index))
    with pytest.raises(ValueError):
        ModeloBCall().calcular(x, g, "Z")
    with pytest.raises(ValueError):
        run(x, pd.concat([g, g]))
    with pytest.raises(ValueError):
        run(x, pd.concat([g, g], axis=1))


def test_identificadores_invalidos(ejemplo):
    x, g = ejemplo
    x.index = ["A", "A", "C", "D", "E"]
    with pytest.raises(ValueError):
        run(x, g)
    with pytest.raises(ValueError):
        run(pd.DataFrame(), g)
