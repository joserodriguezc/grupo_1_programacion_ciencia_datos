# F3/test/unit/test_recorrido_votos.py

import pytest

from F3.src.nucleo.recorrido_votos import (
    UMBRAL_DIVISION,
    construir_dataset_sintetico,
    recorrer_iterativo,
    recorrer_recursivo,
)

VOTACION = {
    "Id": "20627",
    "Descripcion": "Votación de prueba",
    "Fecha": "2024-01-01T00:00:00",
    "TotalSi": "10",
    "TotalNo": "5",
    "TotalAbstencion": "0",
    "TotalDispensado": "0",
    "Quorum": {"_value_1": "Simple", "Valor": 1},
    "Resultado": {"_value_1": "Aprobado", "Valor": 1},
    "Tipo": {"_value_1": "Normal", "Valor": 1},
}


def _voto(diputado_id: int) -> dict:
    return {
        "Diputado": {
            "Id": diputado_id,
            "Nombre": f"Nombre{diputado_id}",
            "Nombre2": None,
            "ApellidoPaterno": f"Apellido{diputado_id}",
            "ApellidoMaterno": "Soto",
        },
        "OpcionVoto": {"_value_1": "A favor", "Valor": 1},
    }


# ---------------------------------------------------------------------------
# 1. Igualdad exacta iterativo vs. recursivo (plan de trabajo: "igualdad
#    exacta de resultados antes de medir")
# ---------------------------------------------------------------------------

def test_igualdad_lista_vacia():
    assert recorrer_iterativo([], VOTACION) == []
    assert recorrer_recursivo([], VOTACION) == []


def test_igualdad_un_solo_voto():
    votos = [_voto(1)]
    assert recorrer_iterativo(votos, VOTACION) == recorrer_recursivo(votos, VOTACION)


def test_igualdad_bajo_el_umbral_de_division():
    votos = [_voto(i) for i in range(UMBRAL_DIVISION - 1)]
    assert recorrer_iterativo(votos, VOTACION) == recorrer_recursivo(votos, VOTACION)


def test_igualdad_justo_en_el_umbral_de_division():
    votos = [_voto(i) for i in range(UMBRAL_DIVISION)]
    assert recorrer_iterativo(votos, VOTACION) == recorrer_recursivo(votos, VOTACION)


def test_igualdad_justo_sobre_el_umbral_dispara_una_division():
    votos = [_voto(i) for i in range(UMBRAL_DIVISION + 1)]
    assert recorrer_iterativo(votos, VOTACION) == recorrer_recursivo(votos, VOTACION)


def test_igualdad_y_orden_se_preservan_con_muchas_divisiones():
    # 501 votos fuerza varios niveles de partición recursiva.
    votos = [_voto(i) for i in range(501)]
    iterativo = recorrer_iterativo(votos, VOTACION)
    recursivo = recorrer_recursivo(votos, VOTACION)
    assert iterativo == recursivo
    # el orden debe conservarse: no solo "mismo contenido", mismo orden.
    assert [fila["diputado_id"] for fila in recursivo] == list(range(501))


def test_recursivo_no_revienta_el_limite_de_recursion_en_volumen_10x():
    # 19.930 = 10x del dataset real (1.993 votos). Una recursión ingenua
    # (una llamada por voto) superaría el límite por defecto de Python
    # (1000) y lanzaría RecursionError; la partición por mitades no.
    votos = [_voto(1)] * 19_930
    resultado = recorrer_recursivo(votos, VOTACION)
    assert len(resultado) == 19_930
    assert resultado == recorrer_iterativo(votos, VOTACION)


# ---------------------------------------------------------------------------
# 2. Decodificación de enums (Quorum/Resultado/Tipo/OpcionVoto) igual en
#    ambas estrategias, porque comparten _fila_desde_voto()
# ---------------------------------------------------------------------------

def test_ambas_estrategias_decodifican_los_mismos_enums():
    votos = [_voto(1)]
    fila_it = recorrer_iterativo(votos, VOTACION)[0]
    fila_rec = recorrer_recursivo(votos, VOTACION)[0]
    for fila in (fila_it, fila_rec):
        assert fila["quorum"] == "Simple"
        assert fila["resultado"] == "Aprobado"
        assert fila["opcion_voto"] == "A favor"
        assert fila["opcion_codigo"] == 1


# ---------------------------------------------------------------------------
# 3. construir_dataset_sintetico()
# ---------------------------------------------------------------------------

def test_construir_dataset_sintetico_replica_el_factor_pedido():
    votos = [_voto(1), _voto(2)]
    assert construir_dataset_sintetico(votos, factor=1) == votos
    sintetico = construir_dataset_sintetico(votos, factor=5)
    assert len(sintetico) == 10
    assert sintetico == votos * 5


def test_construir_dataset_sintetico_rechaza_factor_invalido():
    with pytest.raises(ValueError):
        construir_dataset_sintetico([_voto(1)], factor=0)
    with pytest.raises(ValueError):
        construir_dataset_sintetico([_voto(1)], factor=-3)
