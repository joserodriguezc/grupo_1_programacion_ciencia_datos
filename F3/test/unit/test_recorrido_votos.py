# F3/test/unit/test_recorrido_votos.py
"""Pruebas de recorrido_votos.py: ambas estrategias reciben los mismos
nodos <Voto> de ElementTree que entrega el XML crudo real (ver
F3/data/raw/votaciones/*.xml), igual que
ExtractorDetalleVotaciones._parsear()."""

import xml.etree.ElementTree as ET

import pytest

from F3.src.nucleo.extractores import _hijo, _hijos
from F3.src.nucleo.recorrido_votos import (
    UMBRAL_DIVISION,
    construir_dataset_sintetico,
    recorrer_iterativo,
    recorrer_recursivo,
)

NS_V1 = "http://opendata.camara.cl/camaradiputados/v1"


def _voto_xml(diputado_id: int) -> str:
    return (
        f"<Voto><Diputado><Id>{diputado_id}</Id><Nombre>Nombre{diputado_id}</Nombre>"
        "<Nombre2/>"
        f"<ApellidoPaterno>Apellido{diputado_id}</ApellidoPaterno>"
        "<ApellidoMaterno>Soto</ApellidoMaterno></Diputado>"
        '<OpcionVoto Valor="1">A favor</OpcionVoto></Voto>'
    )


def _votacion(votos_xml: str) -> ET.Element:
    """Arma un nodo <Votacion> con la misma forma real (namespace v1),
    incluyendo su lista de <Votos>."""
    xml = (
        f'<Votacion xmlns="{NS_V1}">'
        "<Id>20627</Id><Descripcion>Votación de prueba</Descripcion>"
        "<Fecha>2024-01-01T00:00:00</Fecha><TotalSi>10</TotalSi><TotalNo>5</TotalNo>"
        "<TotalAbstencion>0</TotalAbstencion><TotalDispensado>0</TotalDispensado>"
        '<Quorum Valor="1">Simple</Quorum>'
        '<Resultado Valor="1">Aprobado</Resultado>'
        '<Tipo Valor="1">Normal</Tipo>'
        f"<Votos>{votos_xml}</Votos>"
        "</Votacion>"
    )
    return ET.fromstring(xml)


def _votos_de(n: int) -> tuple[ET.Element, list[ET.Element]]:
    votacion = _votacion("".join(_voto_xml(i) for i in range(n)))
    return votacion, _hijos(_hijo(votacion, "Votos"), "Voto")


# ---------------------------------------------------------------------------
# 1. Igualdad exacta iterativo vs. recursivo
# ---------------------------------------------------------------------------

def test_igualdad_lista_vacia():
    votacion, votos = _votos_de(0)
    assert recorrer_iterativo(votos, votacion) == []
    assert recorrer_recursivo(votos, votacion) == []


def test_igualdad_un_solo_voto():
    votacion, votos = _votos_de(1)
    assert recorrer_iterativo(votos, votacion) == recorrer_recursivo(votos, votacion)


def test_igualdad_bajo_el_umbral_de_division():
    votacion, votos = _votos_de(UMBRAL_DIVISION - 1)
    assert recorrer_iterativo(votos, votacion) == recorrer_recursivo(votos, votacion)


def test_igualdad_justo_en_el_umbral_de_division():
    votacion, votos = _votos_de(UMBRAL_DIVISION)
    assert recorrer_iterativo(votos, votacion) == recorrer_recursivo(votos, votacion)


def test_igualdad_justo_sobre_el_umbral_dispara_una_division():
    votacion, votos = _votos_de(UMBRAL_DIVISION + 1)
    assert recorrer_iterativo(votos, votacion) == recorrer_recursivo(votos, votacion)


def test_igualdad_y_orden_se_preservan_con_muchas_divisiones():
    votacion, votos = _votos_de(501)
    iterativo = recorrer_iterativo(votos, votacion)
    recursivo = recorrer_recursivo(votos, votacion)
    assert iterativo == recursivo
    assert [fila["diputado_id"] for fila in recursivo] == [str(i) for i in range(501)]


def test_recursivo_no_revienta_el_limite_de_recursion_en_volumen_10x():
    votacion, votos = _votos_de(1)
    votos = construir_dataset_sintetico(votos, factor=19_930)
    resultado = recorrer_recursivo(votos, votacion)
    assert len(resultado) == 19_930
    assert resultado == recorrer_iterativo(votos, votacion)


# ---------------------------------------------------------------------------
# 2. Decodificación de enums
# ---------------------------------------------------------------------------

def test_ambas_estrategias_decodifican_los_mismos_enums():
    votacion, votos = _votos_de(1)
    fila_it = recorrer_iterativo(votos, votacion)[0]
    fila_rec = recorrer_recursivo(votos, votacion)[0]
    for fila in (fila_it, fila_rec):
        assert fila["quorum"] == "Simple"
        assert fila["resultado"] == "Aprobado"
        assert fila["opcion_voto"] == "A favor"
        assert fila["opcion_codigo"] == "1"


# ---------------------------------------------------------------------------
# 3. construir_dataset_sintetico()
# ---------------------------------------------------------------------------

def test_construir_dataset_sintetico_replica_el_factor_pedido():
    _, votos = _votos_de(2)
    assert construir_dataset_sintetico(votos, factor=1) == votos
    sintetico = construir_dataset_sintetico(votos, factor=5)
    assert len(sintetico) == 10
    assert sintetico == votos * 5


def test_construir_dataset_sintetico_rechaza_factor_invalido():
    _, votos = _votos_de(1)
    with pytest.raises(ValueError):
        construir_dataset_sintetico(votos, factor=0)
    with pytest.raises(ValueError):
        construir_dataset_sintetico(votos, factor=-3)
