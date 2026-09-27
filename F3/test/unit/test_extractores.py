# F3/test/unit/test_extractores.py
"""Pruebas de ExtractorBase y sus 4 subclases sobre el flujo
descargar()/procesar(): _parsear() recibe el XML crudo (ElementTree), no
los objetos que entregaba zeep. Se construyen sobres SOAP mínimos con la
misma forma real que devuelve la API (namespace v1), en vez de simular la
forma de zeep."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from F3.src.nucleo.extractores import (
    ExtractorDetalleVotaciones,
    ExtractorDiputados,
    ExtractorPeriodosLegislativos,
    ExtractorVotacionesProyecto,
    RespuestaCruda,
    construir_dataframe_interim,
)

NS_SOAP = "http://schemas.xmlsoap.org/soap/envelope/"
NS_V1 = "http://opendata.camara.cl/camaradiputados/v1"


def _envelope(metodo: str, cuerpo_interno: str) -> str:
    """Arma un sobre SOAP mínimo, con la misma forma que guarda el
    extractor real (ver F3/data/raw/*.xml)."""
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        f'<soap:Envelope xmlns:soap="{NS_SOAP}">'
        "<soap:Body>"
        f'<{metodo}Response xmlns="{NS_V1}">'
        f"<{metodo}Result>{cuerpo_interno}</{metodo}Result>"
        f"</{metodo}Response>"
        "</soap:Body>"
        "</soap:Envelope>"
    )


def _raiz(metodo: str, cuerpo_interno: str) -> ET.Element:
    return ET.fromstring(_envelope(metodo, cuerpo_interno))


# ---------------------------------------------------------------------------
# 1. ExtractorPeriodosLegislativos
# ---------------------------------------------------------------------------

def test_periodos_parsear_construye_filas_esperadas():
    extractor = ExtractorPeriodosLegislativos(base_dir="no-se-usa")

    raiz = _raiz(
        "retornarPeriodosLegislativos",
        "<PeriodoLegislativo><Id>55</Id><Nombre>Período 2018-2022</Nombre>"
        "<FechaInicio>2018-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2022-03-11T00:00:00</FechaTermino></PeriodoLegislativo>"
        "<PeriodoLegislativo><Id>56</Id><Nombre>Período 2022-2026</Nombre>"
        "<FechaInicio>2022-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2026-03-11T00:00:00</FechaTermino></PeriodoLegislativo>",
    )

    filas = extractor._parsear(raiz)

    assert filas == [
        {"periodo_id": "55", "nombre": "Período 2018-2022",
         "fecha_inicio": "2018-03-11T00:00:00", "fecha_termino": "2022-03-11T00:00:00"},
        {"periodo_id": "56", "nombre": "Período 2022-2026",
         "fecha_inicio": "2022-03-11T00:00:00", "fecha_termino": "2026-03-11T00:00:00"},
    ]


def test_periodos_extraer_guarda_crudo_y_csv_ordenado(tmp_path, monkeypatch):
    extractor = ExtractorPeriodosLegislativos(base_dir=tmp_path)

    xml_crudo = _envelope(
        "retornarPeriodosLegislativos",
        "<PeriodoLegislativo><Id>56</Id><Nombre>Período 2022-2026</Nombre>"
        "<FechaInicio>2022-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2026-03-10T00:00:00</FechaTermino></PeriodoLegislativo>"
        "<PeriodoLegislativo><Id>55</Id><Nombre>Período 2018-2022</Nombre>"
        "<FechaInicio>2018-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2022-03-10T00:00:00</FechaTermino></PeriodoLegislativo>",
    )
    monkeypatch.setattr(
        extractor, "_consultar", lambda: RespuestaCruda(datos=None, xml_crudo=xml_crudo)
    )

    df = extractor.extraer()

    ruta_xml = tmp_path / "data" / "raw" / "periodos_legislativos.xml"
    assert ruta_xml.read_text(encoding="utf-8") == xml_crudo

    ruta_csv = tmp_path / "data" / "interim" / "periodos.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorPeriodosLegislativos.MODELO.COLUMNAS)

    # -- quedó ordenado por fecha_inicio, aunque llegó desordenado --
    assert df["periodo_id"].tolist() == ["55", "56"]


def test_periodos_extraer_falla_con_fila_invalida(tmp_path, monkeypatch):
    extractor = ExtractorPeriodosLegislativos(base_dir=tmp_path)

    xml_crudo = _envelope(
        "retornarPeriodosLegislativos",
        "<PeriodoLegislativo><Id>55</Id><Nombre>Período válido</Nombre>"
        "<FechaInicio>2018-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2022-03-11T00:00:00</FechaTermino></PeriodoLegislativo>"
        "<PeriodoLegislativo><Id>no-es-numero</Id><Nombre>Período roto</Nombre>"
        "<FechaInicio>2022-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2026-03-11T00:00:00</FechaTermino></PeriodoLegislativo>",
    )
    monkeypatch.setattr(
        extractor, "_consultar", lambda: RespuestaCruda(datos=None, xml_crudo=xml_crudo)
    )

    with pytest.raises(ValueError, match="fila 2"):
        extractor.extraer()


# ---------------------------------------------------------------------------
# 2. ExtractorVotacionesProyecto
# ---------------------------------------------------------------------------

_VOTACION_PROYECTO_XML = (
    "<Id>42724</Id><Descripcion>Boletín N°11092-07</Descripcion>"
    "<Fecha>2024-08-26T19:03:25</Fecha><TotalSi>65</TotalSi><TotalNo>22</TotalNo>"
    "<TotalAbstencion>36</TotalAbstencion><TotalDispensado>0</TotalDispensado>"
    '<Quorum Valor="2">Quórum Calificado</Quorum>'
    '<Resultado Valor="1">Aprobado</Resultado>'
    '<Tipo Valor="1">Proyecto de Ley</Tipo>'
    '<TipoVotacionProyectoLey Valor="6">Única</TipoVotacionProyectoLey>'
    "<Articulo>Texto del artículo.</Articulo>"
    '<TramiteConstitucional Valor="4">Comisión Mixta</TramiteConstitucional>'
    '<TramiteReglamentario Valor="7">Sin Informe</TramiteReglamentario>'
)


def test_votaciones_proyecto_parsear_extrae_texto_legible_de_los_enums():
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")

    raiz = _raiz(
        "retornarVotacionesXProyectoLey",
        f"<Votaciones><VotacionProyectoLey>{_VOTACION_PROYECTO_XML}</VotacionProyectoLey>"
        f"<VotacionProyectoLey><Id>21219</Id><Descripcion>Otra</Descripcion>"
        "<Fecha>2024-08-26T19:03:25</Fecha><TotalSi>1</TotalSi><TotalNo>1</TotalNo>"
        "<TotalAbstencion>0</TotalAbstencion><TotalDispensado>0</TotalDispensado>"
        '<Quorum Valor="1">Quórum Simple</Quorum><Resultado Valor="1">Aprobado</Resultado>'
        '<Tipo Valor="1">Proyecto de Ley</Tipo>'
        '<TipoVotacionProyectoLey Valor="6">Única</TipoVotacionProyectoLey>'
        "<Articulo/>"
        '<TramiteConstitucional Valor="4">Comisión Mixta</TramiteConstitucional>'
        '<TramiteReglamentario Valor="7">Sin Informe</TramiteReglamentario>'
        "</VotacionProyectoLey></Votaciones>",
    )

    filas = extractor._parsear(raiz)

    assert len(filas) == 2
    assert filas[0]["numero_boletin"] == "11092-07"
    assert filas[0]["Quorum"] == "Quórum Calificado"
    assert filas[0]["Resultado"] == "Aprobado"
    assert filas[0]["TramiteConstitucional"] == "Comisión Mixta"
    assert filas[1]["Id"] == "21219"
    assert filas[1]["Articulo"] is None


def test_votaciones_proyecto_parsear_sin_votaciones_devuelve_lista_vacia():
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")

    raiz = _raiz("retornarVotacionesXProyectoLey", "<Votaciones/>")

    assert extractor._parsear(raiz) == []


def test_votaciones_proyecto_extraer_guarda_crudo_y_csv(tmp_path, monkeypatch):
    extractor = ExtractorVotacionesProyecto(base_dir=tmp_path, numero_boletin="11092-07")

    xml_crudo = _envelope(
        "retornarVotacionesXProyectoLey",
        f"<Votaciones><VotacionProyectoLey>{_VOTACION_PROYECTO_XML}</VotacionProyectoLey></Votaciones>",
    )
    monkeypatch.setattr(
        extractor, "_consultar", lambda: RespuestaCruda(datos=None, xml_crudo=xml_crudo)
    )

    df = extractor.extraer()

    ruta_xml = tmp_path / "data" / "raw" / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"
    assert ruta_xml.read_text(encoding="utf-8") == xml_crudo

    ruta_csv = tmp_path / "data" / "interim" / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorVotacionesProyecto.MODELO.COLUMNAS)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# 3. ExtractorDetalleVotaciones: N descargas -> 1 CSV consolidado
# ---------------------------------------------------------------------------

_VOTO_XML = (
    "<Voto><Diputado><Id>803</Id><Nombre>René</Nombre><Nombre2/>"
    "<ApellidoPaterno>Alinco</ApellidoPaterno><ApellidoMaterno>Bustos</ApellidoMaterno>"
    '</Diputado><OpcionVoto Valor="0">En Contra</OpcionVoto></Voto>'
)


def _votacion_detalle_xml(votacion_id: int, votos_xml: str) -> str:
    return (
        f"<Id>{votacion_id}</Id><Descripcion>Boletín N° 11092-07</Descripcion>"
        "<Fecha>2024-08-26T19:03:25</Fecha><TotalSi>65</TotalSi><TotalNo>22</TotalNo>"
        "<TotalAbstencion>36</TotalAbstencion><TotalDispensado>0</TotalDispensado>"
        '<Quorum Valor="2">Quórum Calificado</Quorum>'
        '<Resultado Valor="1">Aprobado</Resultado>'
        '<Tipo Valor="1">Proyecto de Ley</Tipo>'
        f"<Votos>{votos_xml}</Votos>"
    )


def test_detalle_votaciones_parsear_extrae_un_voto_por_diputado():
    extractor = ExtractorDetalleVotaciones(base_dir="no-se-usa", ids_votaciones=[42724])

    otro_voto = (
        "<Voto><Diputado><Id>872</Id><Nombre>Jaime</Nombre><Nombre2/>"
        "<ApellidoPaterno>Mulet</ApellidoPaterno><ApellidoMaterno>Martínez</ApellidoMaterno>"
        '</Diputado><OpcionVoto Valor="1">Afirmativo</OpcionVoto></Voto>'
    )
    raiz = _raiz(
        "retornarVotacionDetalle", _votacion_detalle_xml(42724, _VOTO_XML + otro_voto)
    )

    filas = extractor._parsear(raiz)

    assert len(filas) == 2
    assert filas[0]["diputado_id"] == "803"
    assert filas[0]["opcion_voto"] == "En Contra"
    assert filas[0]["opcion_codigo"] == "0"
    assert filas[0]["quorum"] == "Quórum Calificado"
    assert filas[1]["diputado_id"] == "872"
    assert filas[1]["opcion_voto"] == "Afirmativo"


def test_detalle_votaciones_parsear_sin_votos_devuelve_lista_vacia():
    extractor = ExtractorDetalleVotaciones(base_dir="no-se-usa", ids_votaciones=[42724])

    raiz = _raiz("retornarVotacionDetalle", _votacion_detalle_xml(42724, ""))

    assert extractor._parsear(raiz) == []


def test_detalle_votaciones_extraer_guarda_un_xml_por_id_y_un_solo_csv(tmp_path, monkeypatch):
    extractor = ExtractorDetalleVotaciones(base_dir=tmp_path, ids_votaciones=[42724, 21219])

    xml_por_id = {
        42724: _envelope("retornarVotacionDetalle", _votacion_detalle_xml(42724, _VOTO_XML)),
        21219: _envelope("retornarVotacionDetalle", _votacion_detalle_xml(21219, _VOTO_XML)),
    }
    monkeypatch.setattr(
        extractor,
        "_consultar_una",
        lambda votacion_id: RespuestaCruda(datos=None, xml_crudo=xml_por_id[votacion_id]),
    )

    df = extractor.extraer()

    ruta_xml_1 = tmp_path / "data" / "raw" / "votaciones" / "votacion_42724.xml"
    ruta_xml_2 = tmp_path / "data" / "raw" / "votaciones" / "votacion_21219.xml"
    assert ruta_xml_1.read_text(encoding="utf-8") == xml_por_id[42724]
    assert ruta_xml_2.read_text(encoding="utf-8") == xml_por_id[21219]

    ruta_csv = tmp_path / "data" / "interim" / "votaciones" / "detalle_votaciones.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorDetalleVotaciones.MODELO.COLUMNAS)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 4. ExtractorDiputados: una descarga -> dos salidas (diputados + militancias)
# ---------------------------------------------------------------------------

def _diputado_periodo_xml(diputado_id: int, militancias_xml: str = "") -> str:
    return (
        "<DiputadoPeriodo><FechaInicio>2022-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2026-03-10T23:59:59</FechaTermino>"
        f"<Diputado><Id>{diputado_id}</Id><Nombre>María Candelaria</Nombre><Nombre2/>"
        "<ApellidoPaterno>Acevedo</ApellidoPaterno><ApellidoMaterno>Sáez</ApellidoMaterno>"
        "<FechaNacimiento>1958-09-12T00:00:00</FechaNacimiento><RUT/><RUTDV/>"
        '<Sexo Valor="0">Femenino</Sexo>'
        f"<Militancias>{militancias_xml}</Militancias></Diputado></DiputadoPeriodo>"
    )


_MILITANCIA_XML = (
    "<Militancia><FechaInicio>2022-03-11T00:00:00</FechaInicio>"
    "<FechaTermino>2026-03-10T23:59:59</FechaTermino>"
    "<Partido><Id>PC</Id><Nombre>Partido Comunista</Nombre><Alias>PC</Alias></Partido>"
    "</Militancia>"
)


def test_diputados_parsear_mapea_columnas_y_decodifica_sexo():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    raiz = _raiz("retornarDiputadosXPeriodo", _diputado_periodo_xml(1096))

    filas = extractor._parsear(raiz)

    assert len(filas) == 1
    assert filas[0]["diputado_id"] == "1096"
    assert filas[0]["periodo_id"] == "10"
    assert filas[0]["sexo_valor"] == "0"
    assert filas[0]["sexo_desc"] == "Femenino"
    assert filas[0]["rut"] is None  # <RUT/> vacío -> texto_opcional lo vuelve ""


def test_diputados_militancias_admite_coleccion_vacia():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    raiz = _raiz("retornarDiputadosXPeriodo", _diputado_periodo_xml(1096))

    assert extractor._parsear_militancias(raiz) == []


def test_diputados_militancias_admite_varios_partidos():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    militancia_dc = (
        "<Militancia><FechaInicio>2022-03-11T00:00:00</FechaInicio>"
        "<FechaTermino>2026-03-10T23:59:59</FechaTermino>"
        "<Partido><Id>DC</Id><Nombre>Partido DC</Nombre><Alias>DC</Alias></Partido></Militancia>"
    )
    raiz = _raiz(
        "retornarDiputadosXPeriodo",
        _diputado_periodo_xml(1096, _MILITANCIA_XML + militancia_dc),
    )

    filas = extractor._parsear_militancias(raiz)

    assert [fila["partido_alias"] for fila in filas] == ["PC", "DC"]
    assert filas[0]["diputado_id"] == "1096"


def test_diputados_extraer_guarda_un_xml_y_dos_csv(tmp_path, monkeypatch):
    extractor = ExtractorDiputados(base_dir=tmp_path, periodo_id=10)

    xml_crudo = _envelope(
        "retornarDiputadosXPeriodo",
        _diputado_periodo_xml(1096, _MILITANCIA_XML) + _diputado_periodo_xml(1097),
    )
    monkeypatch.setattr(
        extractor, "_consultar", lambda: RespuestaCruda(datos=None, xml_crudo=xml_crudo)
    )

    df_diputados, df_militancias = extractor.extraer()

    ruta_xml = tmp_path / "data" / "raw" / "diputados" / "diputados_periodo_10.xml"
    assert ruta_xml.read_text(encoding="utf-8") == xml_crudo

    ruta_diputados = tmp_path / "data" / "interim" / "diputados.csv"
    ruta_militancias = tmp_path / "data" / "interim" / "militancias.csv"
    assert ruta_diputados.exists()
    assert ruta_militancias.exists()

    df_diputados_leido = pd.read_csv(ruta_diputados, dtype=str)
    df_militancias_leido = pd.read_csv(ruta_militancias, dtype=str)
    assert list(df_diputados_leido.columns) == list(ExtractorDiputados.MODELO.COLUMNAS)
    assert list(df_militancias_leido.columns) == list(
        ExtractorDiputados.MODELO_MILITANCIA.COLUMNAS
    )
    assert len(df_diputados) == 2
    assert len(df_militancias) == 1


# ---------------------------------------------------------------------------
# 5. Regresión: el parseo nuevo reproduce EXACTO lo que ya hay en el repo
#    (data/raw/*.xml -> mismo contenido que data/interim/*.csv de hoy)
# ---------------------------------------------------------------------------

F3_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = F3_DIR / "data"


def _leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype=str).fillna("")


def _df_desde_filas(filas, modelo) -> pd.DataFrame:
    """Pasa las filas por el mismo camino que procesar(): construir_dataframe_interim()
    (valida con el dataclass del modelo, ahí es donde fecha_iso() normaliza el
    separador de fecha), y las deja como texto para comparar contra el CSV."""
    return construir_dataframe_interim(filas, modelo).astype(str)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="no hay data/ en este entorno")
def test_regresion_periodos_igual_al_csv_actual():
    extractor = ExtractorPeriodosLegislativos(base_dir="no-se-usa")
    raiz = ET.parse(DATA_DIR / "raw" / "periodos_legislativos.xml").getroot()

    df_nuevo = _df_desde_filas(extractor._parsear(raiz), extractor.MODELO)
    df_nuevo = df_nuevo.sort_values("fecha_inicio").reset_index(drop=True)
    df_viejo = _leer_csv(DATA_DIR / "interim" / "periodos.csv")

    assert df_nuevo.equals(df_viejo.reset_index(drop=True))


@pytest.mark.skipif(not DATA_DIR.exists(), reason="no hay data/ en este entorno")
def test_regresion_votaciones_proyecto_igual_al_csv_actual():
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")
    raiz = ET.parse(DATA_DIR / "raw" / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml").getroot()

    df_nuevo = _df_desde_filas(extractor._parsear(raiz), extractor.MODELO)
    df_viejo = _leer_csv(DATA_DIR / "interim" / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv")

    assert df_nuevo.reset_index(drop=True).equals(df_viejo.reset_index(drop=True))


@pytest.mark.skipif(not DATA_DIR.exists(), reason="no hay data/ en este entorno")
def test_regresion_detalle_votaciones_igual_al_csv_actual():
    df_viejo = _leer_csv(DATA_DIR / "interim" / "votaciones" / "detalle_votaciones.csv")
    orden_ids = df_viejo["votacion_id"].drop_duplicates().tolist()

    extractor = ExtractorDetalleVotaciones(base_dir="no-se-usa", ids_votaciones=orden_ids)
    filas = []
    for votacion_id in orden_ids:
        raiz = ET.parse(DATA_DIR / "raw" / "votaciones" / f"votacion_{votacion_id}.xml").getroot()
        filas.extend(extractor._parsear(raiz))

    df_nuevo = _df_desde_filas(filas, extractor.MODELO)
    assert df_nuevo.reset_index(drop=True).equals(df_viejo.reset_index(drop=True))


@pytest.mark.skipif(not DATA_DIR.exists(), reason="no hay data/ en este entorno")
def test_regresion_diputados_y_militancias_igual_al_csv_actual():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)
    raiz = ET.parse(DATA_DIR / "raw" / "diputados" / "diputados_periodo_10.xml").getroot()

    df_diputados_nuevo = _df_desde_filas(extractor._parsear(raiz), extractor.MODELO)
    df_diputados_viejo = _leer_csv(DATA_DIR / "interim" / "diputados.csv")
    assert df_diputados_nuevo.reset_index(drop=True).equals(
        df_diputados_viejo.reset_index(drop=True)
    )

    df_militancias_nuevo = _df_desde_filas(
        extractor._parsear_militancias(raiz), extractor.MODELO_MILITANCIA
    )
    df_militancias_viejo = _leer_csv(DATA_DIR / "interim" / "militancias.csv")
    assert df_militancias_nuevo.reset_index(drop=True).equals(
        df_militancias_viejo.reset_index(drop=True)
    )
