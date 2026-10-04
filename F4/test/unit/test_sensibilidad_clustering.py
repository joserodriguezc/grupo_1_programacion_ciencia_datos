import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.sensibilidad import (
    ErrorSensibilidad,
    SensibilidadClustering,
    _comparar_clusters,
)


def _bloques_con_parcial() -> pd.DataFrame:
    """Dos bloques y P, que vota solo en tres de cinco votaciones (cobertura 0,6)."""
    return pd.DataFrame(
        {
            1: ["Sí", "Sí", "Sí", "No", "No", "No", "Sí"],
            2: ["Sí", "Sí", "Sí", "No", "No", "No", "Sí"],
            3: ["Sí", "Sí", "No", "No", "No", "Sí", "Sí"],
            4: ["No", "No", "No", "Sí", "Sí", "Sí", np.nan],
            5: ["Sí", "Sí", "Sí", "No", "No", "No", np.nan],
        },
        index=["A1", "A2", "A3", "B1", "B2", "B3", "P"],
    )


def _escenario(resultado, nombre):
    filas = resultado.resumen.loc[resultado.resumen["escenario"].eq(nombre)]
    assert len(filas) == 1
    return filas.iloc[0]


# ------------------------------------------------------------- _comparar_clusters
def test_ari_es_uno_con_etiquetas_intercambiadas() -> None:
    base = pd.Series(["L", "L", "R", "R"])
    alt = pd.Series([2, 2, 1, 1])

    ari, mapa, ambiguo = _comparar_clusters(base, alt)

    assert ari == pytest.approx(1)
    assert mapa == {1: "R", 2: "L"}
    assert not ambiguo


def test_ari_reproduce_valor_calculado_a_mano() -> None:
    # Contingencia [[2, 0], [1, 1]]: índice 1, esperado 1, máximo 2,5 → ARI 0.
    ari, _, _ = _comparar_clusters(pd.Series(["a", "a", "b", "b"]), pd.Series([1, 1, 1, 2]))

    assert ari == pytest.approx(0)


def test_alineacion_empatada_se_marca_como_ambigua() -> None:
    _, _, ambiguo = _comparar_clusters(
        pd.Series(["a", "a", "b", "b"]), pd.Series([1, 2, 1, 2])
    )

    assert ambiguo


def test_comparar_clusters_exige_indices_alineados() -> None:
    with pytest.raises(ValueError, match="alineados"):
        _comparar_clusters(pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[1, 2]))


# ------------------------------------------------------------- SensibilidadClustering
def test_genera_escenarios_de_votacion_bloque_y_cobertura() -> None:
    resultado = SensibilidadClustering(umbrales_cobertura=(0.4, 0.8)).calcular(
        _bloques_con_parcial(), bloques={"primeras": ["1", "2"]}
    )

    assert resultado.resumen["escenario"].tolist() == [
        "base",
        "retiro_votacion:1",
        "retiro_votacion:2",
        "retiro_votacion:3",
        "retiro_votacion:4",
        "retiro_votacion:5",
        "retiro_bloque:primeras",
        "cobertura:0.4",
    ]
    # Los IDs del bloque se normalizan: "1" y "2" corresponden a las columnas enteras.
    assert _escenario(resultado, "retiro_bloque:primeras")["votaciones_retiradas"] == "1|2"


def test_escenario_base_es_identico() -> None:
    resultado = SensibilidadClustering().calcular(_bloques_con_parcial())
    base = _escenario(resultado, "base")

    assert base["ari"] == pytest.approx(1)
    assert base["n_cambios_grupo"] == 0
    assert base["n_entran"] == 0 and base["n_salen"] == 0
    # P queda fuera con cobertura 0,8 (exige 4 de 5 votos).
    assert base["n_diputados_base"] == 6


def test_bloques_claros_son_estables_al_retirar_una_votacion() -> None:
    resultado = SensibilidadClustering().calcular(_bloques_con_parcial())
    retiros = resultado.resumen.loc[
        resultado.resumen["escenario"].str.startswith("retiro_votacion")
    ]

    assert retiros["estado"].eq("DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS").all()
    np.testing.assert_allclose(retiros["ari"].astype(float), 1)
    assert retiros["n_cambios_grupo"].eq(0).all()


def test_cobertura_menor_registra_diputados_que_entran() -> None:
    resultado = SensibilidadClustering(umbrales_cobertura=(0.4,)).calcular(
        _bloques_con_parcial()
    )
    fila = _escenario(resultado, "cobertura:0.4")
    detalle = resultado.detalle.loc[resultado.detalle["escenario"].eq("cobertura:0.4")]
    p = detalle.set_index("diputado_id").loc["P"]

    assert fila["n_entran"] == 1
    assert fila["n_comunes"] == 6
    # El ARI usa solo los diputados comunes; P se informa aparte.
    assert fila["ari"] == pytest.approx(1)
    assert p["estado_universo"] == "ENTRA"
    assert not p["incluido_base"] and p["incluido_alternativo"]
    assert pd.isna(p["cambio_grupo"])


def test_escenario_no_estimable_no_se_reporta_como_ari_cero() -> None:
    matriz = pd.DataFrame(
        {
            "j1": ["Sí", "No", "Sí", "No"],
            "j2": ["Sí", "No", "No", "Sí"],
            "unanime": ["Sí"] * 4,
        },
        index=["A", "B", "C", "D"],
    )

    resultado = SensibilidadClustering(umbrales_cobertura=(0.8,)).calcular(matriz)
    fila = _escenario(resultado, "retiro_votacion:j1")

    assert fila["estado"] == "ESCENARIO_NO_ESTIMABLE"
    assert np.isnan(fila["ari"])
    assert "dos votaciones informativas" in fila["razon_NA"]
    assert resultado.detalle["escenario"].ne("retiro_votacion:j1").all()
    # Retirar la votación unánime no cambia el universo informativo.
    assert _escenario(resultado, "retiro_votacion:unanime")["ari"] == pytest.approx(1)


@pytest.mark.parametrize(
    "bloques",
    [{"vacio": []}, {"": ["1"]}, {"inexistente": ["99"]}],
)
def test_rechaza_bloques_invalidos(bloques) -> None:
    with pytest.raises(ErrorSensibilidad, match="Bloque inválido"):
        SensibilidadClustering().calcular(_bloques_con_parcial(), bloques=bloques)


def test_valida_parametros() -> None:
    with pytest.raises(ValueError, match="umbral alternativo"):
        SensibilidadClustering(umbrales_cobertura=())
    with pytest.raises(ValueError, match="cobertura_minima"):
        SensibilidadClustering(umbrales_cobertura=(0.4, 1.5))
    with pytest.raises(ValueError, match="n_clusters"):
        SensibilidadClustering(n_clusters=1)
