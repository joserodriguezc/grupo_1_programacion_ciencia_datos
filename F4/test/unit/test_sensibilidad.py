import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.cohesion import IndicesCohesion
from F4.src.analisis.sensibilidad import (
    AnalisisRobustez,
    SensibilidadCohesion,
    _orden_relativo,
)


def _datos_sinteticos():
    matriz = pd.DataFrame(
        {
            1: [-1, -1, 1, 1],
            2: [-1, 1, 1, 1],
            3: [-1, -1, 1, -1],
            4: [1, -1, 1, -1],
        },
        index=[10, 11, 20, 21],
        dtype=float,
    )

    filas = []
    partidos = {
        10: ("A", "Partido A", "PA"),
        11: ("A", "Partido A", "PA"),
        20: ("B", "Partido B", "PB"),
        21: ("B", "Partido B", "PB"),
    }
    for diputado in matriz.index:
        for votacion in matriz.columns:
            pid, nombre, alias = partidos[diputado]
            filas.append(
                {
                    "diputado_id": diputado,
                    "votacion_id": votacion,
                    "fecha": f"2026-01-0{votacion}",
                    "partido_id": pid,
                    "partido_nombre": nombre,
                    "partido_alias": alias,
                }
            )
    return matriz, pd.DataFrame(filas)


def _analisis():
    return AnalisisRobustez(
        pivot=20,
        metodo_distancia=1,
        threshold_bcall=0.1,
        min_decisiones_partido_votacion=1,
        min_votaciones_partido=1,
        umbrales_cobertura=(0.40, 0.60, 0.80),
    )


def test_retirar_votacion_recalcula_modelos_completos() -> None:
    matriz, afiliacion = _datos_sinteticos()
    resultado = _analisis().calcular(matriz, afiliacion)

    fila = resultado[
        (resultado["metodo"] == "bcall_d1")
        & (resultado["id"].astype(str) == "10")
        & (resultado["votacion_retirada"] == 4)
        & (resultado["umbral_cobertura"] == 0.60)
    ].iloc[0]

    # La alternativa se obtiene de un nuevo B-Call, no de restar un valor
    # preexistente del promedio final.
    assert pd.notna(fila["valor_base"])
    assert pd.notna(fila["valor_alternativo"])
    assert fila["n_efectivo_alternativo"] <= fila["n_efectivo_base"]


def test_genera_un_escenario_por_cada_votacion() -> None:
    matriz, afiliacion = _datos_sinteticos()
    resultado = _analisis().calcular(matriz, afiliacion)

    assert set(resultado["votacion_retirada"].unique()) == {1, 2, 3, 4}


def test_repite_cada_comparacion_en_umbrales_40_60_80() -> None:
    matriz, afiliacion = _datos_sinteticos()
    resultado = _analisis().calcular(matriz, afiliacion)

    subconjunto = resultado[
        (resultado["metodo"] == "bcall_d1")
        & (resultado["id"].astype(str) == "10")
        & (resultado["votacion_retirada"] == 1)
    ]

    assert set(subconjunto["umbral_cobertura"]) == {0.4, 0.6, 0.8}


def test_incluye_d1_d2_y_posicion_partidaria() -> None:
    matriz, afiliacion = _datos_sinteticos()
    resultado = _analisis().calcular(matriz, afiliacion)

    assert {"bcall_d1", "bcall_d2", "perfil_partidario_P_p"}.issubset(
        set(resultado["metodo"])
    )


def test_no_interpreta_cobertura_observada_como_asistencia() -> None:
    matriz, afiliacion = _datos_sinteticos()
    matriz.loc[10, 4] = np.nan
    afiliacion = afiliacion[
        ~(
            afiliacion["diputado_id"].eq(10)
            & afiliacion["votacion_id"].eq(4)
        )
    ]

    resultado = _analisis().calcular(matriz, afiliacion)

    assert (resultado["estado"] == "DESCRIPTIVO_SIN_PADRON").all()


def test_rechaza_afiliacion_duplicada() -> None:
    matriz, afiliacion = _datos_sinteticos()
    afiliacion = pd.concat([afiliacion, afiliacion.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="única"):
        _analisis().calcular(matriz, afiliacion)


# --------------------------------------------------------------- cohesión partidaria
# p1: AI por votación (1; 1; 0,5; 0,25); j2 es unánime en la sala; j4 de p1 tiene T = 2.
CONTEOS = [("p1", "j1", 0, 3, 0), ("p1", "j2", 3, 0, 0), ("p1", "j3", 1, 2, 0),
           ("p1", "j4", 1, 0, 1), ("p2", "j1", 2, 0, 0), ("p2", "j2", 2, 0, 0),
           ("p2", "j3", 2, 0, 0), ("p2", "j4", 0, 2, 0)]


@pytest.fixture
def sensibilidad_cohesion():
    conteos = pd.DataFrame(
        [{"partido_id": p, "partido_alias": p, "votacion_id": j, "fecha": f"2023-01-0{j[1]}",
          "Y": y, "N": n, "A": a, "T": y + n + a} for p, j, y, n, a in CONTEOS])
    por_votacion = IndicesCohesion().calcular(conteos)
    return SensibilidadCohesion(umbrales_cobertura=(0.6,)).calcular(por_votacion)


def _fila(tabla, escenario, metodo, partido, retirada=None):
    filas = tabla[(tabla["escenario"] == escenario) & (tabla["metodo"] == metodo)
                  & (tabla["id"] == partido)]
    if retirada is not None:
        filas = filas[filas["votacion_retirada"] == retirada]
    assert len(filas) == 1
    return filas.iloc[0]


def test_cohesion_retiro_de_una_votacion(sensibilidad_cohesion):
    fila = _fila(sensibilidad_cohesion, "retiro_una_votacion", "mediana_agreement_index",
                 "p1", "j1")
    assert fila["valor_base"] == pytest.approx(0.75)
    assert fila["valor_alternativo"] == pytest.approx(0.5)  # (1; 0,5; 0,25)
    assert fila["diferencia"] == pytest.approx(-0.25)


def test_cohesion_sin_votaciones_unanimes(sensibilidad_cohesion):
    fila = _fila(sensibilidad_cohesion, "sin_votaciones_unanimes",
                 "mediana_agreement_index", "p1")
    assert fila["votacion_retirada"] == "j2"
    assert fila["valor_alternativo"] == pytest.approx(0.5)


def test_cohesion_minimo_de_decisiones_alternativo(sensibilidad_cohesion):
    p1 = _fila(sensibilidad_cohesion, "minimo_decisiones_3", "mediana_agreement_index", "p1")
    assert (p1["n_efectivo_base"], p1["n_efectivo_alternativo"]) == (4, 3)  # sale j4
    assert p1["valor_alternativo"] == pytest.approx(1.0)
    p2 = _fila(sensibilidad_cohesion, "minimo_decisiones_3", "mediana_agreement_index", "p2")
    assert p2["publicable_base"] and not p2["publicable_alternativo"]
    assert p2["cambio_publicabilidad"]


def test_cohesion_contraste_entre_indices(sensibilidad_cohesion):
    fila = _fila(sensibilidad_cohesion, "indice_alternativo", "contraste_rice_vs_ai", "p1")
    assert fila["valor_base"] == pytest.approx(0.75)  # AI
    assert fila["valor_alternativo"] == pytest.approx(1.0)  # Rice: sin j4 (un voto binario)


def test_cohesion_usa_el_mismo_formato_que_la_posicion(sensibilidad_cohesion):
    matriz, afiliacion = _datos_sinteticos()
    posicion = _analisis().calcular(matriz, afiliacion)
    assert list(sensibilidad_cohesion.columns) == list(posicion.columns)
    assert set(sensibilidad_cohesion["familia"]) == {"cohesion_partidaria"}
    assert set(sensibilidad_cohesion["orientacion_escenario"]) == {"no_aplica"}


def test_orden_relativo_ignora_ruido_de_punto_flotante() -> None:
    base = pd.Series([-0.7147086138910308, -0.7147086138910308, 0.5], index=[1, 2, 3])
    alternativo = pd.Series([-0.7147086138910307, -0.7147086138910308, 0.5], index=[1, 2, 3])

    pd.testing.assert_series_equal(_orden_relativo(base), _orden_relativo(alternativo))
    assert _orden_relativo(base).tolist() == [1.5, 1.5, 3.0]
