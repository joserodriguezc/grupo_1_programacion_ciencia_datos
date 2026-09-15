import importlib.util
from pathlib import Path

import pandas as pd

SRC_FILE = Path(__file__).resolve().parents[2] / "src" / "04_extraer_detalles_votaciones.py"

COLUMNAS_ESPERADAS = [
    "diputado_id",
    "nombre",
    "nombre2",
    "apellido_paterno",
    "apellido_materno",
    "opcion_codigo",
    "opcion_voto",
    "votacion_id",
    "descripcion",
    "fecha",
    "total_si",
    "total_no",
    "total_abstencion",
    "total_dispensado",
    "quorum_codigo",
    "quorum",
    "resultado_codigo",
    "resultado",
    "tipo_codigo",
    "tipo",
]


def cargar_modulo():
    spec = importlib.util.spec_from_file_location("detalle_votaciones", SRC_FILE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def xml_votacion() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<Votacion xmlns="http://opendata.camara.cl/camaradiputados/v1">
  <Id>20627</Id>
  <Descripcion>Boletín N° 11092-07</Descripcion>
  <Fecha>2023-05-08 16:00:00</Fecha>
  <TotalSi>1</TotalSi>
  <TotalNo>0</TotalNo>
  <TotalAbstencion>0</TotalAbstencion>
  <TotalDispensado>0</TotalDispensado>
  <Quorum Valor="1">Quórum Simple</Quorum>
  <Resultado Valor="1">Aprobado</Resultado>
  <Tipo Valor="1">Proyecto de Ley</Tipo>
  <Votos>
    <Voto>
      <Diputado>
        <Id>1001</Id>
        <Nombre>Ana</Nombre>
        <Nombre2 />
        <ApellidoPaterno>Pérez</ApellidoPaterno>
        <ApellidoMaterno>Soto</ApellidoMaterno>
      </Diputado>
      <OpcionVoto Valor="1">Afirmativo</OpcionVoto>
    </Voto>
  </Votos>
</Votacion>
"""


def test_leer_xml_votacion_mapea_detalle_nominal(tmp_path):
    modulo = cargar_modulo()
    ruta = tmp_path / "votacion.xml"
    ruta.write_text(xml_votacion(), encoding="utf-8")

    resultado = modulo.leer_xml_votacion(ruta)

    assert list(resultado.columns) == COLUMNAS_ESPERADAS
    assert len(resultado) == 1
    assert resultado.loc[0, "votacion_id"] == "20627"
    assert resultado.loc[0, "diputado_id"] == "1001"
    assert resultado.loc[0, "opcion_codigo"] == "1"
    assert resultado.loc[0, "opcion_voto"] == "Afirmativo"


def test_obtener_ids_elimina_duplicados(tmp_path):
    modulo = cargar_modulo()
    ruta = tmp_path / "proyecto_ley.csv"
    pd.DataFrame({"Id": [20627, 20627, 20628, None]}).to_csv(ruta, index=False)

    ids = modulo.obtener_ids_votaciones(ruta)

    assert ids == [20627, 20628]


def test_guardar_votos_csv_respeta_columnas_y_crea_directorio(tmp_path):
    modulo = cargar_modulo()
    xml_path = tmp_path / "votacion.xml"
    xml_path.write_text(xml_votacion(), encoding="utf-8")
    detalle = modulo.leer_xml_votacion(xml_path)

    salida = tmp_path / "interim" / "votaciones" / "detalle_votaciones.csv"
    modulo.guardar_votos_csv(detalle, salida)

    assert salida.exists()
    guardado = pd.read_csv(salida)
    assert list(guardado.columns) == COLUMNAS_ESPERADAS
    assert len(guardado) == 1
