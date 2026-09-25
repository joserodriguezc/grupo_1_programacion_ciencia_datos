# F3/test/unit/test_extractores.py

import pandas as pd
import pytest

from F3.src.nucleo.extractores import (
    ExtractorDetalleVotaciones,
    ExtractorDiputados,
    ExtractorPeriodosLegislativos,
    ExtractorVotacionesProyecto,
    RespuestaCruda,
)

# ---------------------------------------------------------------------------
# 1. _parsear() aislado: no toca red, no toca disco
# ---------------------------------------------------------------------------

def test_parsear_construye_filas_esperadas():
    extractor = ExtractorPeriodosLegislativos(base_dir="no-se-usa")

    # Zeep devuelve objetos accesibles como diccionario (p["Id"]);
    # un dict normal cumple esa misma interfaz para efectos del test.
    datos_falsos = [
        {"Id": 55, "Nombre": "Período 2018-2022",
         "FechaInicio": "2018-03-11", "FechaTermino": "2022-03-11"},
        {"Id": 56, "Nombre": "Período 2022-2026",
         "FechaInicio": "2022-03-11", "FechaTermino": "2026-03-11"},
    ]
    respuesta = RespuestaCruda(datos=datos_falsos, xml_crudo="<xml/>")

    filas = extractor._parsear(respuesta)

    assert filas == [
        {"periodo_id": 55, "nombre": "Período 2018-2022",
         "fecha_inicio": "2018-03-11", "fecha_termino": "2022-03-11"},
        {"periodo_id": 56, "nombre": "Período 2022-2026",
         "fecha_inicio": "2022-03-11", "fecha_termino": "2026-03-11"},
    ]


# ---------------------------------------------------------------------------
# 2. extraer() completo, con _consultar() reemplazado (sin red)
# ---------------------------------------------------------------------------

def test_extraer_guarda_crudo_y_csv_ordenado(tmp_path, monkeypatch):
    extractor = ExtractorPeriodosLegislativos(base_dir=tmp_path)

    respuesta_falsa = RespuestaCruda(
        datos=[
            {"Id": 56, "Nombre": "Período 2022-2026",
             "FechaInicio": "2022-03-11", "FechaTermino": "2026-03-11"},
            {"Id": 55, "Nombre": "Período 2018-2022",
             "FechaInicio": "2018-03-11", "FechaTermino": "2022-03-11"},
        ],
        xml_crudo="<envelope>periodos-fake</envelope>",
    )
    monkeypatch.setattr(extractor, "_consultar", lambda: respuesta_falsa)

    df = extractor.extraer()

    # -- XML crudo se guardó tal cual --
    ruta_xml = tmp_path / "data" / "raw" / "periodos_legislativos.xml"
    assert ruta_xml.read_text(encoding="utf-8") == "<envelope>periodos-fake</envelope>"

    # -- CSV interim existe y tiene las columnas del contrato de José --
    ruta_csv = tmp_path / "data" / "interim" / "periodos.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorPeriodosLegislativos.MODELO.COLUMNAS)

    # -- quedó ordenado por fecha_inicio, aunque llegó desordenado --
    assert df["periodo_id"].tolist() == ["55", "56"]


# ---------------------------------------------------------------------------
# 3. Fila inválida: debe fallar con el número de fila, no descartarla
# ---------------------------------------------------------------------------

def test_extraer_falla_con_fila_invalida(tmp_path, monkeypatch):
    extractor = ExtractorPeriodosLegislativos(base_dir=tmp_path)

    respuesta_falsa = RespuestaCruda(
        datos=[
            {"Id": 55, "Nombre": "Período válido",
             "FechaInicio": "2018-03-11", "FechaTermino": "2022-03-11"},
            {"Id": "no-es-numero", "Nombre": "Período roto",
             "FechaInicio": "2022-03-11", "FechaTermino": "2026-03-11"},
        ],
        xml_crudo="<envelope/>",
    )
    monkeypatch.setattr(extractor, "_consultar", lambda: respuesta_falsa)

    with pytest.raises(ValueError, match="fila 2"):
        extractor.extraer()


# ---------------------------------------------------------------------------
# 4. ExtractorVotacionesProyecto: _parsear() contra la forma real de zeep
#    (enums como {'_value_1': texto, 'Valor': código}, ver boletín 11092-07)
# ---------------------------------------------------------------------------

def _votacion_proyecto_falsa(**overrides) -> dict:
    base = {
        "Id": 42724,
        "Descripcion": "Boletín N°11092-07",
        "Fecha": "2024-08-26T19:03:25",
        "TotalSi": 65,
        "TotalNo": 22,
        "TotalAbstencion": 36,
        "TotalDispensado": 0,
        "Quorum": {"_value_1": "Quórum Calificado", "Valor": 2},
        "Resultado": {"_value_1": "Aprobado", "Valor": 1},
        "Tipo": {"_value_1": "Proyecto de Ley", "Valor": 1},
        "TipoVotacionProyectoLey": {"_value_1": "Única", "Valor": 6},
        "Articulo": "Texto del artículo.",
        "TramiteConstitucional": {"_value_1": "Comisión Mixta", "Id": 4},
        "TramiteReglamentario": {"_value_1": "Sin Informe", "Id": 7},
    }
    base.update(overrides)
    return base


def test_votaciones_proyecto_parsear_extrae_texto_legible_de_los_enums():
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")

    proyecto_falso = {
        "Votaciones": {
            "VotacionProyectoLey": [
                _votacion_proyecto_falsa(),
                _votacion_proyecto_falsa(Id=21219, Articulo=None),
            ]
        }
    }
    respuesta = RespuestaCruda(datos=proyecto_falso, xml_crudo="<xml/>")

    filas = extractor._parsear(respuesta)

    assert len(filas) == 2
    assert filas[0]["numero_boletin"] == "11092-07"
    assert filas[0]["Quorum"] == "Quórum Calificado"
    assert filas[0]["Resultado"] == "Aprobado"
    assert filas[0]["TramiteConstitucional"] == "Comisión Mixta"
    assert filas[1]["Id"] == 21219
    assert filas[1]["Articulo"] is None


def test_votaciones_proyecto_parsear_acepta_una_sola_votacion_sin_lista():
    # A veces zeep no envuelve en lista cuando hay un único elemento.
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")

    proyecto_falso = {"Votaciones": {"VotacionProyectoLey": _votacion_proyecto_falsa()}}
    respuesta = RespuestaCruda(datos=proyecto_falso, xml_crudo="<xml/>")

    filas = extractor._parsear(respuesta)

    assert len(filas) == 1
    assert filas[0]["Id"] == 42724


def test_votaciones_proyecto_parsear_sin_votaciones_devuelve_lista_vacia():
    extractor = ExtractorVotacionesProyecto(base_dir="no-se-usa", numero_boletin="11092-07")

    respuesta = RespuestaCruda(datos={"Votaciones": None}, xml_crudo="<xml/>")

    assert extractor._parsear(respuesta) == []


def test_votaciones_proyecto_extraer_guarda_crudo_y_csv(tmp_path, monkeypatch):
    extractor = ExtractorVotacionesProyecto(base_dir=tmp_path, numero_boletin="11092-07")

    respuesta_falsa = RespuestaCruda(
        datos={"Votaciones": {"VotacionProyectoLey": [_votacion_proyecto_falsa()]}},
        xml_crudo="<envelope>votaciones-fake</envelope>",
    )
    monkeypatch.setattr(extractor, "_consultar", lambda: respuesta_falsa)

    df = extractor.extraer()

    ruta_xml = tmp_path / "data" / "raw" / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"
    assert ruta_xml.read_text(encoding="utf-8") == "<envelope>votaciones-fake</envelope>"

    ruta_csv = tmp_path / "data" / "interim" / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorVotacionesProyecto.MODELO.COLUMNAS)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# 5. ExtractorDetalleVotaciones: N consultas -> N XML -> 1 CSV consolidado
# ---------------------------------------------------------------------------

def _votacion_detalle_falsa(votos: list[dict], **overrides) -> dict:
    base = {
        "Id": 42724,
        "Descripcion": "Boletín N° 11092-07",
        "Fecha": "2024-08-26T19:03:25",
        "TotalSi": 65,
        "TotalNo": 22,
        "TotalAbstencion": 36,
        "TotalDispensado": 0,
        "Quorum": {"_value_1": "Quórum Calificado", "Valor": 2},
        "Resultado": {"_value_1": "Aprobado", "Valor": 1},
        "Tipo": {"_value_1": "Proyecto de Ley", "Valor": 1},
        "Votos": {"Voto": votos},
    }
    base.update(overrides)
    return base


def _voto_falso(**overrides) -> dict:
    base = {
        "Diputado": {
            "Id": 803,
            "Nombre": "René",
            "Nombre2": None,
            "ApellidoPaterno": "Alinco",
            "ApellidoMaterno": "Bustos",
        },
        "OpcionVoto": {"_value_1": "En Contra", "Valor": 0},
    }
    base.update(overrides)
    return base


def test_detalle_votaciones_parsear_extrae_un_voto_por_diputado():
    extractor = ExtractorDetalleVotaciones(base_dir="no-se-usa", ids_votaciones=[42724])

    votacion_falsa = _votacion_detalle_falsa(
        votos=[
            _voto_falso(),
            _voto_falso(
                Diputado={
                    "Id": 872,
                    "Nombre": "Jaime",
                    "Nombre2": None,
                    "ApellidoPaterno": "Mulet",
                    "ApellidoMaterno": "Martínez",
                },
                OpcionVoto={"_value_1": "Afirmativo", "Valor": 1},
            ),
        ]
    )
    respuesta = RespuestaCruda(datos=votacion_falsa, xml_crudo="<xml/>")

    filas = extractor._parsear(respuesta)

    assert len(filas) == 2
    assert filas[0]["diputado_id"] == 803
    assert filas[0]["opcion_voto"] == "En Contra"
    assert filas[0]["opcion_codigo"] == 0
    assert filas[0]["quorum"] == "Quórum Calificado"
    assert filas[1]["diputado_id"] == 872
    assert filas[1]["opcion_voto"] == "Afirmativo"


def test_detalle_votaciones_parsear_sin_votos_devuelve_lista_vacia():
    extractor = ExtractorDetalleVotaciones(base_dir="no-se-usa", ids_votaciones=[42724])

    respuesta = RespuestaCruda(
        datos=_votacion_detalle_falsa(votos=None), xml_crudo="<xml/>"
    )

    assert extractor._parsear(respuesta) == []


def test_detalle_votaciones_extraer_guarda_un_xml_por_id_y_un_solo_csv(tmp_path, monkeypatch):
    extractor = ExtractorDetalleVotaciones(base_dir=tmp_path, ids_votaciones=[42724, 21219])

    respuestas_por_id = {
        42724: RespuestaCruda(
            datos=_votacion_detalle_falsa(votos=[_voto_falso()], Id=42724),
            xml_crudo="<envelope>42724</envelope>",
        ),
        21219: RespuestaCruda(
            datos=_votacion_detalle_falsa(votos=[_voto_falso()], Id=21219),
            xml_crudo="<envelope>21219</envelope>",
        ),
    }
    monkeypatch.setattr(
        extractor, "_consultar_una", lambda votacion_id: respuestas_por_id[votacion_id]
    )

    df = extractor.extraer()

    ruta_xml_1 = tmp_path / "data" / "raw" / "votaciones" / "votacion_42724.xml"
    ruta_xml_2 = tmp_path / "data" / "raw" / "votaciones" / "votacion_21219.xml"
    assert ruta_xml_1.read_text(encoding="utf-8") == "<envelope>42724</envelope>"
    assert ruta_xml_2.read_text(encoding="utf-8") == "<envelope>21219</envelope>"

    ruta_csv = tmp_path / "data" / "interim" / "votaciones" / "detalle_votaciones.csv"
    assert ruta_csv.exists()
    df_leido = pd.read_csv(ruta_csv, dtype=str)
    assert list(df_leido.columns) == list(ExtractorDetalleVotaciones.MODELO.COLUMNAS)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 6. ExtractorDiputados: una consulta -> dos salidas (diputados + militancias)
# ---------------------------------------------------------------------------

def _diputado_periodo_falso(militancias, **overrides) -> dict:
    diputado = {
        "Id": 1096,
        "Nombre": "María Candelaria",
        "Nombre2": None,
        "ApellidoPaterno": "Acevedo",
        "ApellidoMaterno": "Sáez",
        "FechaNacimiento": "1958-09-12 00:00:00",
        "RUT": None,
        "RUTDV": None,
        "Sexo": {"_value_1": "Femenino", "Valor": 0},
        "Militancias": {"Militancia": militancias},
    }
    base = {
        "FechaInicio": "2022-03-11 00:00:00",
        "FechaTermino": "2026-03-10 23:59:59",
        "Diputado": diputado,
    }
    base.update(overrides)
    return base


def _militancia_falsa(partido_id="PC", alias="PC") -> dict:
    return {
        "FechaInicio": "2022-03-11 00:00:00",
        "FechaTermino": "2026-03-10 23:59:59",
        "Partido": {"Id": partido_id, "Nombre": f"Partido {alias}", "Alias": alias},
    }


def test_diputados_parsear_mapea_columnas_y_decodifica_sexo():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    respuesta = RespuestaCruda(
        datos=[_diputado_periodo_falso(militancias=None)], xml_crudo="<xml/>"
    )

    filas = extractor._parsear(respuesta)

    assert len(filas) == 1
    assert filas[0]["diputado_id"] == 1096
    assert filas[0]["periodo_id"] == "10"
    assert filas[0]["sexo_valor"] == 0
    assert filas[0]["sexo_desc"] == "Femenino"


def test_diputados_parsear_acepta_un_solo_diputado_sin_lista():
    # Igual que con las votaciones: con un único DiputadoPeriodo, zeep
    # puede no envolverlo en lista.
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    respuesta = RespuestaCruda(
        datos=_diputado_periodo_falso(militancias=None), xml_crudo="<xml/>"
    )

    filas = extractor._parsear(respuesta)

    assert len(filas) == 1
    assert filas[0]["diputado_id"] == 1096


def test_diputados_militancias_admite_coleccion_vacia():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    respuesta = RespuestaCruda(
        datos=[_diputado_periodo_falso(militancias=None)], xml_crudo="<xml/>"
    )

    assert extractor._parsear_militancias(respuesta) == []


def test_diputados_militancias_admite_una_sola_sin_lista():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    respuesta = RespuestaCruda(
        datos=[_diputado_periodo_falso(militancias=_militancia_falsa())],
        xml_crudo="<xml/>",
    )

    filas = extractor._parsear_militancias(respuesta)

    assert len(filas) == 1
    assert filas[0]["partido_alias"] == "PC"
    assert filas[0]["diputado_id"] == 1096


def test_diputados_militancias_admite_varios_partidos():
    extractor = ExtractorDiputados(base_dir="no-se-usa", periodo_id=10)

    respuesta = RespuestaCruda(
        datos=[
            _diputado_periodo_falso(
                militancias=[_militancia_falsa("PC", "PC"), _militancia_falsa("DC", "DC")]
            )
        ],
        xml_crudo="<xml/>",
    )

    filas = extractor._parsear_militancias(respuesta)

    assert [fila["partido_alias"] for fila in filas] == ["PC", "DC"]


def test_diputados_extraer_guarda_un_xml_y_dos_csv(tmp_path, monkeypatch):
    extractor = ExtractorDiputados(base_dir=tmp_path, periodo_id=10)

    respuesta_falsa = RespuestaCruda(
        datos=[
            _diputado_periodo_falso(militancias=_militancia_falsa()),
            _diputado_periodo_falso(
                militancias=None,
                Diputado={
                    "Id": 1097,
                    "Nombre": "Eric",
                    "Nombre2": None,
                    "ApellidoPaterno": "Aedo",
                    "ApellidoMaterno": "Jeldres",
                    "FechaNacimiento": "1968-07-13 00:00:00",
                    "RUT": None,
                    "RUTDV": None,
                    "Sexo": {"_value_1": "Masculino", "Valor": 1},
                    "Militancias": {"Militancia": None},
                },
            ),
        ],
        xml_crudo="<envelope>diputados-fake</envelope>",
    )
    monkeypatch.setattr(extractor, "_consultar", lambda: respuesta_falsa)

    df_diputados, df_militancias = extractor.extraer()

    ruta_xml = tmp_path / "data" / "raw" / "diputados" / "diputados_periodo_10.xml"
    assert ruta_xml.read_text(encoding="utf-8") == "<envelope>diputados-fake</envelope>"

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