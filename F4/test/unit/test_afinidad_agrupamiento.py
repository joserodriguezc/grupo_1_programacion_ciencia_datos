import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.afinidad import AfinidadPares, ErrorAfinidad, hamming_entre_partidos


def _ejemplo_guia() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "j1": ["No", "No", "No", "Sí", "Sí"],
            "j2": ["Sí", "Sí", "Sí", "Sí", "Sí"],
            "j3": ["No", "No", "Sí", "Sí", "Sí"],
            "j4": ["Sí", "Abstención", np.nan, "No", "No"],
        },
        index=["A", "B", "C", "D", "E"],
    )


def _fila(resultado, i, j):
    return resultado.pares[
        resultado.pares["diputado_i"].eq(i)
        & resultado.pares["diputado_j"].eq(j)
    ].iloc[0]


def test_reproduce_ejemplo_guia() -> None:
    resultado = AfinidadPares(min_covotos=1).calcular(_ejemplo_guia())

    de = _fila(resultado, "D", "E")
    ab = _fila(resultado, "A", "B")
    ac = _fila(resultado, "A", "C")
    ad = _fila(resultado, "A", "D")

    assert de["n_covotos"] == 4
    assert de["acuerdo"] == pytest.approx(1.0)
    assert de["hamming"] == pytest.approx(0.0)

    assert ab["n_covotos"] == 4
    assert ab["acuerdo"] == pytest.approx(0.75)
    assert ab["hamming"] == pytest.approx(0.25)

    assert ac["n_covotos"] == 3
    assert ac["cobertura_corpus"] == pytest.approx(0.75)
    assert ac["acuerdo"] == pytest.approx(2 / 3)
    assert ac["hamming"] == pytest.approx(1 / 3)

    assert ad["n_covotos"] == 4
    assert ad["acuerdo"] == pytest.approx(0.25)
    assert ad["hamming"] == pytest.approx(0.75)


def test_dos_ausencias_no_cuentan_como_acuerdo() -> None:
    matriz = pd.DataFrame(
        {
            "j1": [np.nan, np.nan],
            "j2": ["Sí", "Sí"],
            "j3": ["No", np.nan],
        },
        index=[1, 2],
    )

    resultado = AfinidadPares(min_covotos=1).calcular(matriz)
    fila = resultado.pares.iloc[0]

    assert fila["n_covotos"] == 1
    assert fila["coincidencias"] == 1
    assert fila["acuerdo"] == pytest.approx(1.0)
    assert fila["hamming"] == pytest.approx(0.0)


def test_sin_covotos_produce_na_y_razon() -> None:
    matriz = pd.DataFrame(
        {"j1": ["Sí", np.nan], "j2": [np.nan, "No"]},
        index=[1, 2],
    )

    resultado = AfinidadPares(min_covotos=1).calcular(matriz)
    fila = resultado.pares.iloc[0]

    assert fila["n_covotos"] == 0
    assert pd.isna(fila["acuerdo"])
    assert pd.isna(fila["hamming"])
    assert not bool(fila["incluido"])
    assert fila["razon_exclusion"] == "SIN_COVOTOS"


def test_minimo_covotos_controla_inclusion() -> None:
    matriz = pd.DataFrame(
        {
            "j1": ["Sí", "Sí"],
            "j2": ["No", np.nan],
            "j3": ["Abstención", np.nan],
        },
        index=[1, 2],
    )

    resultado = AfinidadPares(min_covotos=2).calcular(matriz)
    fila = resultado.pares.iloc[0]

    assert fila["n_covotos"] == 1
    assert fila["acuerdo"] == pytest.approx(1.0)
    assert not bool(fila["incluido"])
    assert fila["razon_exclusion"] == "COVOTOS_INSUFICIENTES"
    assert pd.isna(resultado.matriz_hamming.loc[1, 2])


def test_abstencion_es_categoria_nominal_no_punto_medio() -> None:
    matriz = pd.DataFrame(
        {"j1": ["Sí", "Abstención"], "j2": ["No", "No"]},
        index=[1, 2],
    )

    resultado = AfinidadPares(min_covotos=1).calcular(matriz)
    fila = resultado.pares.iloc[0]

    assert fila["n_covotos"] == 2
    assert fila["coincidencias"] == 1
    assert fila["acuerdo"] == pytest.approx(0.5)
    assert fila["hamming"] == pytest.approx(0.5)


def test_contraste_sin_votaciones_unanimes() -> None:
    resultado = AfinidadPares(min_covotos=1).calcular(_ejemplo_guia())
    ad = _fila(resultado, "A", "D")

    # j2 es unánime y es la única coincidencia entre A y D.
    assert ad["n_covotos_sin_unanimes"] == 3
    assert ad["acuerdo_sin_unanimes"] == pytest.approx(0.0)
    assert ad["hamming_sin_unanimes"] == pytest.approx(1.0)


def test_matriz_hamming_es_simetrica_y_diagonal_cero() -> None:
    resultado = AfinidadPares(min_covotos=1).calcular(_ejemplo_guia())
    matriz = resultado.matriz_hamming

    assert np.allclose(matriz.to_numpy(), matriz.to_numpy().T, equal_nan=True)
    assert np.allclose(np.diag(matriz.to_numpy()), 0.0)


def test_rechaza_categoria_nominal_desconocida() -> None:
    matriz = pd.DataFrame({"j1": ["Sí", "Ausente"]}, index=[1, 2])

    with pytest.raises(ErrorAfinidad, match="categorías no reconocidas"):
        AfinidadPares().calcular(matriz)


def _afiliacion_guia(cambios=None) -> pd.DataFrame:
    """A, B y C en p1; D y E en p2. cambios: {(diputado, votación): partido}."""
    partido = {"A": "p1", "B": "p1", "C": "p1", "D": "p2", "E": "p2"}
    return pd.DataFrame(
        [
            {"diputado_id": d, "votacion_id": j,
             "partido_alias": (cambios or {}).get((d, j), partido[d])}
            for d in partido
            for j in ("j1", "j2", "j3", "j4")
        ]
    )


def test_hamming_entre_partidos_reproduce_ejemplo_guia() -> None:
    resultado = hamming_entre_partidos(_ejemplo_guia(), _afiliacion_guia())
    tabla = resultado.set_index(["partido_a", "partido_b"])

    # p1 × p1: 10 comparaciones (3 + 3 + 3 + 1, C sin voto en j4) y 3 distintas.
    assert tabla.loc[("p1", "p1"), "n_comparaciones"] == 10
    assert tabla.loc[("p1", "p1"), "hamming"] == pytest.approx(0.3)
    # p1 × p2: 6 + 6 + 6 + 4 comparaciones; distintas 6 + 0 + 4 + 4.
    assert tabla.loc[("p1", "p2"), "n_comparaciones"] == 22
    assert tabla.loc[("p1", "p2"), "hamming"] == pytest.approx(14 / 22)
    assert tabla.loc[("p2", "p2"), "hamming"] == 0.0
    assert set(tabla["n_votaciones"]) == {4}


def test_hamming_entre_partidos_usa_el_partido_vigente_en_cada_votacion() -> None:
    # A pasa a p2 solo en j1: en esa votación se compara con D y E como p2 × p2.
    resultado = hamming_entre_partidos(
        _ejemplo_guia(), _afiliacion_guia({("A", "j1"): "p2"})
    ).set_index(["partido_a", "partido_b"])

    assert resultado.loc[("p2", "p2"), "n_comparaciones"] == 4 + 2
    assert resultado.loc[("p2", "p2"), "hamming"] == pytest.approx(2 / 6)
    assert resultado.loc[("p1", "p1"), "n_comparaciones"] == 10 - 2


def test_hamming_entre_partidos_omite_diputados_sin_afiliacion() -> None:
    afiliacion = _afiliacion_guia()
    afiliacion = afiliacion[afiliacion["diputado_id"].ne("E")]
    resultado = hamming_entre_partidos(_ejemplo_guia(), afiliacion)
    tabla = resultado.set_index(["partido_a", "partido_b"])

    assert ("p2", "p2") not in tabla.index  # solo D queda en p2: no hay pares
    assert tabla.loc[("p1", "p2"), "n_comparaciones"] == 3 + 3 + 3 + 2
