import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.universo import (
    COLUMNAS,
    ENTRADAS,
    RAIZ_REPOSITORIO,
    UniversoPorMetodo,
    cobertura_diputados,
    contar_cobertura,
    desde_repositorio,
    motivos,
    partidos_pequenos,
    votaciones_informativas,
)


# ------------------------------------------------------------------ funciones auxiliares
def test_motivos_cuenta_por_frecuencia_y_omite_nulos() -> None:
    serie = pd.Series(["B", None, "A", "B", np.nan])

    assert motivos(serie) == "B:2|A:1"
    assert motivos(pd.Series([None, np.nan])) is None


def test_contar_cobertura_incluye_el_umbral_exacto() -> None:
    # 0,6 calculado como 3/5 no debe quedar fuera por coma flotante.
    cobertura = pd.Series([0.39, 0.4, 3 / 5, 0.79, 1.0])

    assert contar_cobertura(cobertura) == {
        "n_cobertura_040": 4, "n_cobertura_060": 3, "n_cobertura_080": 1,
    }


def test_cobertura_usa_solo_votaciones_informativas() -> None:
    ternaria = pd.DataFrame(
        {"unanime": [1, 1, 1], "v1": [1, -1, np.nan], "v2": [0, 1, np.nan]},
        index=["a", "b", "c"],
    )

    assert votaciones_informativas(ternaria) == ["v1", "v2"]
    pd.testing.assert_series_equal(
        cobertura_diputados(ternaria), pd.Series([1.0, 1.0, 0.0], index=["a", "b", "c"])
    )


def test_partidos_pequenos_excluye_grupos_no_partidarios() -> None:
    afiliacion = pd.DataFrame({
        "diputado_id": ["1", "2", "3", "4", "5", "6"],
        "partido_id": ["A", "IND", "IND", "B", "B", "B"],
    })

    assert partidos_pequenos(afiliacion, maximo=2).to_dict() == {"A": 1}


def test_requiere_todas_las_tablas() -> None:
    with pytest.raises(ValueError, match="Faltan tablas"):
        UniversoPorMetodo({"ternaria": pd.DataFrame()})


# ------------------------------------------------------- acta sobre las salidas reales
def _salidas_disponibles() -> bool:
    return all((RAIZ_REPOSITORIO / ruta).is_file()
               and (RAIZ_REPOSITORIO / ruta).stat().st_size > 0 for ruta in ENTRADAS.values())


requiere_salidas = pytest.mark.skipif(not _salidas_disponibles(),
                                      reason="faltan salidas de los métodos")


@pytest.fixture(scope="module")
def universo() -> pd.DataFrame:
    return desde_repositorio().calcular().set_index("metodo")


@requiere_salidas
def test_una_fila_por_metodo_con_columnas_del_contrato(universo) -> None:
    assert list(universo.reset_index().columns) == COLUMNAS
    assert universo.index.is_unique
    assert {"bcall_automatico", "posicion_partidaria_P_p", "cohesion_ai_entropia",
            "afinidad_pares", "clustering", "pca"} <= set(universo.index)


@requiere_salidas
def test_cada_exclusion_tiene_motivo(universo) -> None:
    for metodo, fila in universo.iterrows():
        assert fila.n_incluidos + fila.n_excluidos == fila.n_corpus, metodo
        total = 0 if pd.isna(fila.motivos_exclusion) else sum(
            int(parte.rsplit(":", 1)[1]) for parte in fila.motivos_exclusion.split("|"))
        assert total == fila.n_excluidos, metodo


@requiere_salidas
def test_universo_coincide_con_las_salidas_de_cada_metodo(universo) -> None:
    seleccion = pd.read_csv(RAIZ_REPOSITORIO / ENTRADAS["bcall_seleccion"])
    clusters = pd.read_csv(RAIZ_REPOSITORIO / ENTRADAS["clusters"])
    pca = pd.read_csv(RAIZ_REPOSITORIO / ENTRADAS["pca"])

    assert universo.loc["bcall_automatico", "n_incluidos"] == seleccion["incluido"].sum()
    assert universo.loc["clustering", "n_incluidos"] == len(clusters)
    assert universo.loc["pca", "n_incluidos"] == len(pca)
    # El clustering filtra con cobertura 0,80: su universo es el de esa cobertura.
    assert universo.loc["clustering", "n_cobertura_080"] == len(clusters)
