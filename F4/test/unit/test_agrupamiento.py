import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.agrupamiento import (
    AgrupamientoDiputados,
    ErrorAgrupamiento,
    guardar_resultados,
)


def _dos_bloques() -> pd.DataFrame:
    """Dos bloques claros, una votación unánime y un diputado con baja cobertura."""
    return pd.DataFrame(
        {
            "j1": ["Sí", "Sí", "Sí", "No", "No", "No", "Sí"],
            "j2": ["Sí", "Sí", "Sí", "No", "No", "No", np.nan],
            "j3": ["Sí", "Sí", "No", "No", "No", "Sí", np.nan],
            "j4": ["No", "No", "No", "Sí", "Sí", "Sí", np.nan],
            "unanime": ["Sí"] * 7,
        },
        index=["A1", "A2", "A3", "B1", "B2", "B3", "P"],
    )


def _clusters(resultado) -> pd.Series:
    return resultado.diputados.set_index("diputado_id")["cluster"]


def test_separa_dos_bloques_con_etiquetas_neutrales() -> None:
    resultado = AgrupamientoDiputados().calcular(_dos_bloques())
    clusters = _clusters(resultado)

    assert set(clusters.unique()) == {1, 2}
    assert clusters.loc[["A1", "A2", "A3"]].nunique() == 1
    assert clusters.loc[["B1", "B2", "B3"]].nunique() == 1
    assert clusters["A1"] != clusters["B1"]
    assert resultado.diputados["estado"].eq("DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS").all()


def test_excluye_votaciones_sin_variacion_antes_de_filtrar_personas() -> None:
    resultado = AgrupamientoDiputados().calcular(_dos_bloques())

    assert resultado.votaciones_incluidas == ("j1", "j2", "j3", "j4")
    assert resultado.votaciones_excluidas == ("unanime",)
    assert resultado.diputados["n_votaciones_informativas"].eq(4).all()


def test_excluye_por_cobertura_con_razon() -> None:
    resultado = AgrupamientoDiputados(cobertura_minima=0.80).calcular(_dos_bloques())

    # ceil(0,8 · 4) = 4 votos exigidos; P solo registra uno.
    assert resultado.min_covotos == 4
    assert "P" not in set(resultado.diputados["diputado_id"])
    exclusion = resultado.exclusiones.set_index("diputado_id").loc["P"]
    assert exclusion["razon_exclusion"] == "COBERTURA_INSUFICIENTE"
    assert exclusion["n_votos_observados"] == 1
    assert exclusion["cobertura_corpus"] == pytest.approx(0.25)


def test_minimo_tolera_error_de_coma_flotante() -> None:
    # 0,75 · 4 = 3 exactos: no debe redondearse hacia arriba a 4.
    matriz = _dos_bloques().drop(index="P")
    resultado = AgrupamientoDiputados(cobertura_minima=0.75).calcular(matriz)

    assert resultado.min_covotos == 3


def test_matriz_hamming_y_enlace_consistentes() -> None:
    resultado = AgrupamientoDiputados().calcular(_dos_bloques())
    distancias = resultado.matriz_hamming.to_numpy(dtype=float)

    np.testing.assert_allclose(distancias, distancias.T)
    np.testing.assert_allclose(np.diag(distancias), 0)
    assert ((distancias >= 0) & (distancias <= 1)).all()
    # A1 y A2 votan igual; A1 y B1 discrepan en las cuatro votaciones informativas.
    assert resultado.matriz_hamming.loc["A1", "A2"] == pytest.approx(0)
    assert resultado.matriz_hamming.loc["A1", "B1"] == pytest.approx(1)
    assert resultado.enlace.shape == (len(resultado.diputados) - 1, 4)


def test_no_imputa_pares_sin_covotos_suficientes() -> None:
    matriz = pd.DataFrame(
        {
            "j1": ["Sí", "No", "Sí", np.nan],
            "j2": ["Sí", "No", "No", np.nan],
            "j3": ["No", "Sí", np.nan, "Sí"],
            "j4": ["No", "Sí", np.nan, "No"],
        },
        index=["A", "B", "X", "Y"],
    )

    # Con cobertura 0,5 se exigen dos votos; X e Y no comparten ninguno.
    with pytest.raises(ErrorAgrupamiento, match="co-votos"):
        AgrupamientoDiputados(cobertura_minima=0.5).calcular(matriz)


def test_requiere_dos_votaciones_informativas() -> None:
    matriz = pd.DataFrame(
        {"j1": ["Sí", "No", "Sí"], "j2": ["Sí", "Sí", "Sí"]},
        index=["A", "B", "C"],
    )

    with pytest.raises(ErrorAgrupamiento, match="dos votaciones informativas"):
        AgrupamientoDiputados().calcular(matriz)


def test_requiere_diputados_suficientes_para_los_clusters() -> None:
    with pytest.raises(ErrorAgrupamiento, match="clusters"):
        AgrupamientoDiputados(n_clusters=7).calcular(_dos_bloques())


@pytest.mark.parametrize(
    "matriz, mensaje",
    [
        (pd.DataFrame(), "vacía"),
        (pd.DataFrame({"j1": ["Sí", "Quizás"], "j2": ["No", "Sí"]}), "Categorías inválidas"),
        (pd.DataFrame({"j1": ["Sí", "No"], "j2": ["No", "Sí"]}, index=["A", "A"]), "únicos"),
    ],
)
def test_valida_la_matriz_nominal(matriz, mensaje) -> None:
    with pytest.raises(ErrorAgrupamiento, match=mensaje):
        AgrupamientoDiputados().calcular(matriz)


def test_rechaza_entrada_que_no_es_dataframe() -> None:
    with pytest.raises(TypeError):
        AgrupamientoDiputados().calcular([["Sí", "No"]])


@pytest.mark.parametrize("cobertura", [0, -0.1, 1.1, np.nan, True])
def test_valida_cobertura_minima(cobertura) -> None:
    with pytest.raises(ValueError, match="cobertura_minima"):
        AgrupamientoDiputados(cobertura_minima=cobertura)


@pytest.mark.parametrize("n_clusters", [1, 2.0, True])
def test_valida_numero_de_clusters(n_clusters) -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        AgrupamientoDiputados(n_clusters=n_clusters)


def test_guardar_resultados_escribe_todas_las_tablas(tmp_path) -> None:
    resultado = AgrupamientoDiputados().calcular(_dos_bloques())
    guardar_resultados(resultado, tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {
        "clusters_diputados.csv",
        "exclusiones_clustering.csv",
        "matriz_hamming_clustering.csv",
        "enlace_clustering.csv",
        "seleccion_votaciones_clustering.csv",
    }
    seleccion = pd.read_csv(tmp_path / "seleccion_votaciones_clustering.csv")
    assert seleccion.set_index("votacion_id").loc["unanime", "razon"] == (
        "SIN_VARIACION_OBSERVADA"
    )
