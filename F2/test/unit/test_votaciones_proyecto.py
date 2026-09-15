import importlib.util
from pathlib import Path

import pandas as pd

SRC_FILE = Path(__file__).resolve().parents[2] / "src" / "01_extraer_votaciones_proyecto.py"

COLUMNAS_ESPERADAS = [
    "numero_boletin",
    "Id",
    "Descripcion",
    "Fecha",
    "TotalSi",
    "TotalNo",
    "TotalAbstencion",
    "TotalDispensado",
    "Quorum",
    "Resultado",
    "Tipo",
    "TipoVotacionProyectoLey",
    "Articulo",
    "TramiteConstitucional",
    "TramiteReglamentario",
]


def cargar_modulo():
    spec = importlib.util.spec_from_file_location("extraer_votaciones_proyecto", SRC_FILE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class RespuestaFalsa:
    content = b"<xml>respuesta simulada</xml>"

    @staticmethod
    def raise_for_status():
        return None


def test_votacion_valida_nulo_permitido_y_escritura_csv(tmp_path, monkeypatch):
    modulo = cargar_modulo()

    columnas_api = COLUMNAS_ESPERADAS[1:]
    fila = {
        "Id": 20627,
        "Descripcion": "Boletín N° 11092-07",
        "Fecha": "2023-05-08 16:00:00",
        "TotalSi": 122,
        "TotalNo": 0,
        "TotalAbstencion": 0,
        "TotalDispensado": 0,
        "Quorum": "Quórum Simple",
        "Resultado": "Aprobado",
        "Tipo": "Proyecto de Ley",
        "TipoVotacionProyectoLey": "Particular",
        "Articulo": None,
        "TramiteConstitucional": "Segundo Trámite",
        "TramiteReglamentario": "Primer Informe",
    }
    df_api = pd.DataFrame([fila], columns=columnas_api)

    xml_salida = tmp_path / "raw" / "proyecto_ley.xml"
    csv_salida = tmp_path / "interim" / "proyecto_ley.csv"

    monkeypatch.setattr(modulo, "XML_FILE", xml_salida)
    monkeypatch.setattr(modulo, "CSV_FILE", csv_salida)
    monkeypatch.setattr(modulo.requests, "get", lambda *args, **kwargs: RespuestaFalsa())
    monkeypatch.setattr(modulo.pd, "read_xml", lambda *args, **kwargs: df_api.copy())

    resultado = modulo.extraer_votaciones("11092-07")

    assert list(resultado.columns) == COLUMNAS_ESPERADAS
    assert resultado.loc[0, "numero_boletin"] == "11092-07"
    assert pd.isna(resultado.loc[0, "Articulo"])
    assert xml_salida.exists()
    assert csv_salida.exists()

    guardado = pd.read_csv(csv_salida)
    assert list(guardado.columns) == COLUMNAS_ESPERADAS
    assert len(guardado) == 1
