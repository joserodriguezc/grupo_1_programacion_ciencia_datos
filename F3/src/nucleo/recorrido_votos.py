"""Comparación de estrategias de recorrido sobre los dicts que entrega zeep.

ExtractorDetalleVotaciones._parsear() recorre la lista "Votos" -> "Voto" de
forma iterativa (un for). Este módulo aísla esa misma conversión de nodo a
fila en una función compartida y la enfrenta a una variante recursiva, para
medir con metricas.py cuál conviene como predeterminada (plan de trabajo,
Yerko: "recorrido iterativo frente a recursivo sobre los dicts que entrega
zeep").

La recursión NO es "una llamada por voto": con más de ~1.000 votos eso supera
el límite de recursión por defecto de Python (RecursionError) incluso en el
tamaño real (1.993 votos). Se usa recursión por división: la lista se parte
a la mitad en cada llamada, así la profundidad es log2(n) en vez de n
(para 19.930 votos, ~15 niveles, no ~19.930).
"""

from __future__ import annotations

from typing import Any

from .extractores import _codigo_valor, _texto_valor

UMBRAL_DIVISION = 32
"""Bajo este tamaño, la recursión por división deja de partir y arma las
filas directo: evita miles de llamadas extra para segmentos triviales."""


def _fila_desde_voto(voto: dict[str, Any], votacion: dict[str, Any]) -> dict:
    """Conversión de un nodo Voto a la fila que espera VotoNominal.COLUMNAS.

    Compartida por ambas estrategias: lo único que cambia entre ellas es
    cómo se llega a cada `voto`, no cómo se convierte."""
    diputado = voto["Diputado"]
    opcion = voto["OpcionVoto"]
    return {
        "diputado_id": diputado["Id"],
        "nombre": diputado["Nombre"],
        "nombre2": diputado["Nombre2"],
        "apellido_paterno": diputado["ApellidoPaterno"],
        "apellido_materno": diputado["ApellidoMaterno"],
        "opcion_codigo": _codigo_valor(opcion),
        "opcion_voto": _texto_valor(opcion),
        "votacion_id": votacion["Id"],
        "descripcion": votacion["Descripcion"],
        "fecha": votacion["Fecha"],
        "total_si": votacion["TotalSi"],
        "total_no": votacion["TotalNo"],
        "total_abstencion": votacion["TotalAbstencion"],
        "total_dispensado": votacion["TotalDispensado"],
        "quorum_codigo": _codigo_valor(votacion["Quorum"]),
        "quorum": _texto_valor(votacion["Quorum"]),
        "resultado_codigo": _codigo_valor(votacion["Resultado"]),
        "resultado": _texto_valor(votacion["Resultado"]),
        "tipo_codigo": _codigo_valor(votacion["Tipo"]),
        "tipo": _texto_valor(votacion["Tipo"]),
    }


def recorrer_iterativo(
    votos: list[dict[str, Any]], votacion: dict[str, Any]
) -> list[dict]:
    """Recorrido iterativo: el mismo for que usa hoy ExtractorDetalleVotaciones."""
    return [_fila_desde_voto(voto, votacion) for voto in votos]


def recorrer_recursivo(
    votos: list[dict[str, Any]], votacion: dict[str, Any]
) -> list[dict]:
    """Recorrido recursivo por división: profundidad log2(n), no n."""
    if len(votos) <= UMBRAL_DIVISION:
        return [_fila_desde_voto(voto, votacion) for voto in votos]

    mitad = len(votos) // 2
    izquierda = recorrer_recursivo(votos[:mitad], votacion)
    derecha = recorrer_recursivo(votos[mitad:], votacion)
    return izquierda + derecha


def construir_dataset_sintetico(
    votos_reales: list[dict[str, Any]], factor: int
) -> list[dict[str, Any]]:
    """Replica los votos reales `factor` veces para escalar el volumen
    (1x, 5x, 10x). No inventa datos: repite las mismas filas ya obtenidas
    de la API, dejando explícito que es una réplica sintética para medir
    rendimiento, no una muestra real más grande."""
    if factor < 1:
        raise ValueError("factor debe ser >= 1")
    return list(votos_reales) * factor
