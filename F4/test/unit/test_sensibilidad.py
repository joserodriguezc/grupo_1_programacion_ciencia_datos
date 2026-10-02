import numpy as np
import pandas as pd
import pytest

from F4.src.analisis.sensibilidad import AnalisisRobustez


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
