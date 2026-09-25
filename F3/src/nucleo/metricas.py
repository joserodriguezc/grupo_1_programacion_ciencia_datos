"""Utilidades para medición de rendimiento de F3.

Permite medir tiempo de ejecución y memoria Python de funciones de forma
reproducible para comparar la línea base F2 con las implementaciones de F3.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from statistics import median, stdev
from time import perf_counter
from typing import Any


def medir_tiempo(
    funcion: Callable[..., Any],
    *args,
    repeticiones: int = 5,
    calentamiento: int = 1,
    **kwargs,
) -> tuple[Any, dict[str, float | int]]:
    """Mide el tiempo de ejecución de una función.

    Se realizan ejecuciones de calentamiento que no forman parte
    de las estadísticas finales.

    Parameters
    ----------
    funcion:
        Función que será medida.
    *args:
        Argumentos posicionales para la función.
    repeticiones:
        Número de ejecuciones utilizadas para calcular las métricas.
    calentamiento:
        Número de ejecuciones previas que no serán medidas.
    **kwargs:
        Argumentos nombrados para la función.

    Returns
    -------
    tuple
        Resultado de la última ejecución y métricas temporales.
    """

    if repeticiones < 1:
        raise ValueError("repeticiones debe ser mayor o igual a 1")

    if calentamiento < 0:
        raise ValueError("calentamiento no puede ser negativo")

    # Ejecuciones previas para estabilizar cachés/importaciones.
    for _ in range(calentamiento):
        funcion(*args, **kwargs)

    tiempos: list[float] = []
    resultado: Any = None

    for _ in range(repeticiones):
        inicio = perf_counter()

        resultado = funcion(*args, **kwargs)

        tiempo = perf_counter() - inicio

        tiempos.append(tiempo)

    return resultado, {
        "repeticiones": repeticiones,
        "calentamiento": calentamiento,
        "tiempo_mediana_s": median(tiempos),
        "tiempo_min_s": min(tiempos),
        "tiempo_max_s": max(tiempos),
        "tiempo_std_s": (
            stdev(tiempos)
            if len(tiempos) > 1
            else 0.0
        ),
    }


def medir_memoria(
    funcion: Callable[..., Any],
    *args,
    repeticiones: int = 5,
    **kwargs,
) -> tuple[Any, dict[str, float | int]]:
    """Mide el pico de memoria Python utilizado por una función.

    La medición utiliza tracemalloc, por lo que representa principalmente
    asignaciones administradas por Python. No necesariamente captura toda
    la memoria nativa utilizada por bibliotecas como NumPy o pandas.

    Parameters
    ----------
    funcion:
        Función que será medida.
    *args:
        Argumentos posicionales para la función.
    repeticiones:
        Número de ejecuciones.
    **kwargs:
        Argumentos nombrados para la función.

    Returns
    -------
    tuple
        Resultado de la última ejecución y métricas de memoria.
    """

    if repeticiones < 1:
        raise ValueError("repeticiones debe ser mayor o igual a 1")

    picos_memoria: list[float] = []
    resultado: Any = None

    for _ in range(repeticiones):
        tracemalloc.start()

        try:
            resultado = funcion(*args, **kwargs)

            _, memoria_pico = tracemalloc.get_traced_memory()

        finally:
            tracemalloc.stop()

        picos_memoria.append(
            memoria_pico / (1024**2)
        )

    return resultado, {
        "repeticiones": repeticiones,
        "memoria_pico_mediana_mb": median(picos_memoria),
        "memoria_pico_min_mb": min(picos_memoria),
        "memoria_pico_max_mb": max(picos_memoria),
        "memoria_pico_std_mb": (
            stdev(picos_memoria)
            if len(picos_memoria) > 1
            else 0.0
        ),
    }


def benchmark(
    funcion: Callable[..., Any],
    *args,
    repeticiones: int = 5,
    calentamiento: int = 1,
    **kwargs,
) -> tuple[Any, dict[str, float | int]]:
    """Ejecuta un benchmark completo de tiempo y memoria.

    El tiempo y la memoria se miden por separado para evitar que
    tracemalloc altere significativamente la medición temporal.

    Parameters
    ----------
    funcion:
        Función que será medida.
    *args:
        Argumentos posicionales.
    repeticiones:
        Número de repeticiones para cada medición.
    calentamiento:
        Número de ejecuciones previas a la medición temporal.
    **kwargs:
        Argumentos nombrados.

    Returns
    -------
    tuple
        Resultado de la función y diccionario con todas las métricas.
    """

    resultado, metricas_tiempo = medir_tiempo(
        funcion,
        *args,
        repeticiones=repeticiones,
        calentamiento=calentamiento,
        **kwargs,
    )

    _, metricas_memoria = medir_memoria(
        funcion,
        *args,
        repeticiones=repeticiones,
        **kwargs,
    )

    return resultado, {
        **metricas_tiempo,
        **{
            clave: valor
            for clave, valor in metricas_memoria.items()
            if clave != "repeticiones"
        },
    }