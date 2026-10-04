import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.pca_svd import PCASVD, ErrorPCA, guardar_resultados


def _ternaria() -> pd.DataFrame:
    """Cinco diputados completos, uno incompleto y una votación unánime."""
    return pd.DataFrame(
        {
            "v1": [1, 1, -1, -1, 0, 1],
            "v2": [1, 1, -1, 1, -1, np.nan],
            "v3": [1, -1, -1, -1, 1, -1],
            "v4": [0, 1, 0, -1, 1, 0],
            "unanime": [1, 1, 1, 1, 1, 1],
        },
        index=pd.Index([10, 11, 12, 13, 14, 15], name="diputado_id"),
    )


def test_coordenadas_son_proyeccion_de_datos_centrados() -> None:
    resultado = PCASVD(n_componentes=2).calcular(_ternaria())

    x = _ternaria().drop(columns="unanime").drop(index=15)
    cargas = resultado.cargas.set_index("votacion_id")[["PC1", "PC2"]]
    esperado = (x - x.mean()) @ cargas

    coordenadas = resultado.coordenadas.set_index("diputado_id")[["PC1", "PC2"]]
    np.testing.assert_allclose(coordenadas.to_numpy(), esperado.to_numpy())
    pd.testing.assert_series_equal(resultado.medias, x.mean())


def test_cargas_son_ortonormales_y_con_signo_reproducible() -> None:
    resultado = PCASVD(n_componentes=2).calcular(_ternaria())
    cargas = resultado.cargas[["PC1", "PC2"]].to_numpy()

    np.testing.assert_allclose(cargas.T @ cargas, np.eye(2), atol=1e-12)
    for k in range(2):
        assert cargas[np.argmax(np.abs(cargas[:, k])), k] > 0


def test_varianza_explicada_coincide_con_las_coordenadas() -> None:
    resultado = PCASVD(n_componentes=2).calcular(_ternaria())
    varianza = resultado.varianza

    assert varianza["proporcion_varianza"].sum() == pytest.approx(1)
    assert varianza["proporcion_acumulada"].iloc[-1] == pytest.approx(1)
    assert varianza["proporcion_varianza"].is_monotonic_decreasing
    # El autovalor es la varianza muestral de las coordenadas del componente.
    for componente in ["PC1", "PC2"]:
        autovalor = varianza.set_index("componente").loc[componente, "autovalor"]
        assert resultado.coordenadas[componente].var(ddof=1) == pytest.approx(autovalor)
    assert varianza["coordenadas_exportadas"].tolist()[:3] == [True, True, False]


def test_no_imputa_casos_incompletos() -> None:
    resultado = PCASVD().calcular(_ternaria())

    assert 15 not in set(resultado.coordenadas["diputado_id"])
    exclusion = resultado.exclusiones.set_index("diputado_id").loc[15]
    assert exclusion["razon_exclusion"] == "CASO_INCOMPLETO"
    assert exclusion["n_faltantes"] == 1


def test_excluye_votaciones_sin_variacion() -> None:
    resultado = PCASVD().calcular(_ternaria())
    seleccion = resultado.seleccion_votaciones.set_index("votacion_id")

    assert not seleccion.loc["unanime", "incluida"]
    assert seleccion.loc["unanime", "razon"] == "SIN_VARIACION_OBSERVADA"
    assert "unanime" not in set(resultado.cargas["votacion_id"])


def test_datos_unidimensionales_tienen_rango_uno() -> None:
    matriz = pd.DataFrame({"v1": [1, 1, -1, -1], "v2": [1, 1, -1, -1]})

    resultado = PCASVD(n_componentes=1).calcular(matriz)
    assert resultado.rango == 1
    assert resultado.varianza["proporcion_varianza"].iloc[0] == pytest.approx(1)

    with pytest.raises(ErrorPCA, match="rango efectivo es 1"):
        PCASVD(n_componentes=2).calcular(matriz)


def test_requiere_dos_diputados_completos() -> None:
    # Ambas votaciones varían, pero solo el segundo diputado vota en las dos.
    matriz = pd.DataFrame({"v1": [1, -1, np.nan], "v2": [np.nan, 1, -1]})

    with pytest.raises(ErrorPCA, match="dos diputados completos"):
        PCASVD(n_componentes=1).calcular(matriz)


def test_requiere_votaciones_con_variacion() -> None:
    with pytest.raises(ErrorPCA, match="variación"):
        PCASVD().calcular(pd.DataFrame({"v1": [1, 1], "v2": [-1, -1]}))


@pytest.mark.parametrize(
    "matriz, mensaje",
    [
        (pd.DataFrame(), "vacía"),
        (pd.DataFrame({"v1": [1, 2], "v2": [0, 1]}), "-1, 0, 1"),
        (pd.DataFrame({"v1": [True, False], "v2": [1, 0]}), "numéricos"),
        (pd.DataFrame({"v1": ["Sí", "No"], "v2": [1, 0]}), "numéricos"),
        (pd.DataFrame({"v1": [1, -1], "v2": [1, 0]}, index=[1, 1]), "únicos"),
    ],
)
def test_valida_la_matriz_ternaria(matriz, mensaje) -> None:
    with pytest.raises(ErrorPCA, match=mensaje):
        PCASVD().calcular(matriz)


@pytest.mark.parametrize("n_componentes", [0, 1.0, True])
def test_valida_numero_de_componentes(n_componentes) -> None:
    with pytest.raises(ValueError, match="n_componentes"):
        PCASVD(n_componentes=n_componentes)


def test_guardar_resultados_escribe_todas_las_tablas(tmp_path) -> None:
    guardar_resultados(PCASVD().calcular(_ternaria()), tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {
        "coordenadas_diputados.csv",
        "cargas_votaciones.csv",
        "varianza_explicada.csv",
        "exclusiones_pca.csv",
        "seleccion_votaciones_pca.csv",
        "medias_votaciones.csv",
    }
