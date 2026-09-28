"""Compara recorrido iterativo vs. recursivo sobre los votos parseados desde
el XML crudo (ElementTree), la misma conversión de nodo <Voto> a fila que usa
ExtractorDetalleVotaciones._parsear(). Antes se comparaba sobre dicts de zeep;
ahora ambas estrategias reciben los mismos nodos que el extractor real.

La recursión es por división (no una llamada por voto): con más de ~1.000
votos, una llamada por voto supera el límite de recursión de Python. Partir
la lista a la mitad en cada llamada mantiene la profundidad en log2(n).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .extractores import _codigo_atributo, _hijo, _texto, _valor_texto

UMBRAL_DIVISION = 32
"""Bajo este tamaño, la recursión por división deja de partir y arma las
filas directo: evita miles de llamadas extra para segmentos triviales."""


def _fila_desde_voto(voto: ET.Element, votacion: ET.Element) -> dict:
    """Conversión de un nodo <Voto> a la fila que espera VotoNominal.COLUMNAS.

    Es la misma conversión que ExtractorDetalleVotaciones._parsear() hace en
    su for; se aísla aquí para que ambas estrategias de recorrido compartan
    exactamente la misma lógica de fila y solo difieran en cómo llegan a
    cada `voto`."""
    diputado = _hijo(voto, "Diputado")
    opcion = _hijo(voto, "OpcionVoto")
    return {
        "diputado_id": _texto(diputado, "Id"),
        "nombre": _texto(diputado, "Nombre"),
        "nombre2": _texto(diputado, "Nombre2"),
        "apellido_paterno": _texto(diputado, "ApellidoPaterno"),
        "apellido_materno": _texto(diputado, "ApellidoMaterno"),
        "opcion_codigo": _codigo_atributo(opcion),
        "opcion_voto": _valor_texto(opcion),
        "votacion_id": _texto(votacion, "Id"),
        "descripcion": _texto(votacion, "Descripcion"),
        "fecha": _texto(votacion, "Fecha"),
        "total_si": _texto(votacion, "TotalSi"),
        "total_no": _texto(votacion, "TotalNo"),
        "total_abstencion": _texto(votacion, "TotalAbstencion"),
        "total_dispensado": _texto(votacion, "TotalDispensado"),
        "quorum_codigo": _codigo_atributo(_hijo(votacion, "Quorum")),
        "quorum": _valor_texto(_hijo(votacion, "Quorum")),
        "resultado_codigo": _codigo_atributo(_hijo(votacion, "Resultado")),
        "resultado": _valor_texto(_hijo(votacion, "Resultado")),
        "tipo_codigo": _codigo_atributo(_hijo(votacion, "Tipo")),
        "tipo": _valor_texto(_hijo(votacion, "Tipo")),
    }


def recorrer_iterativo(votos: list[ET.Element], votacion: ET.Element) -> list[dict]:
    """Recorrido iterativo: el mismo for que usa hoy ExtractorDetalleVotaciones."""
    return [_fila_desde_voto(voto, votacion) for voto in votos]


def recorrer_recursivo(votos: list[ET.Element], votacion: ET.Element) -> list[dict]:
    """Recorrido recursivo por división: profundidad log2(n), no n."""
    if len(votos) <= UMBRAL_DIVISION:
        return [_fila_desde_voto(voto, votacion) for voto in votos]

    mitad = len(votos) // 2
    izquierda = recorrer_recursivo(votos[:mitad], votacion)
    derecha = recorrer_recursivo(votos[mitad:], votacion)
    return izquierda + derecha


def construir_dataset_sintetico(
    votos_reales: list[ET.Element], factor: int
) -> list[ET.Element]:
    """Replica los votos reales `factor` veces para escalar el volumen
    (1x, 5x, 10x). No inventa datos: repite los mismos nodos <Voto> ya
    parseados desde el XML real, dejando explícito que es una réplica
    sintética para medir rendimiento, no una muestra real más grande."""
    if factor < 1:
        raise ValueError("factor debe ser >= 1")
    return list(votos_reales) * factor
