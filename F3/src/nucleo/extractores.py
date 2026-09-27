# F3/src/nucleo/extractores.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree
from zeep import Client
from zeep.plugins import HistoryPlugin

from .modelos import (
    Diputado,
    Militancia,
    ModeloInterim,
    PeriodoLegislativo,
    VotacionProyecto,
    VotoNominal,
)

NS_SOAP = "http://schemas.xmlsoap.org/soap/envelope/"
NS_V1 = "http://opendata.camara.cl/camaradiputados/v1"


def _resultado(raiz: ET.Element, metodo: str) -> ET.Element:
    """Navega el sobre SOAP hasta el elemento *Result con los datos reales."""
    cuerpo = raiz.find(f"{{{NS_SOAP}}}Body")
    respuesta = cuerpo.find(f"{{{NS_V1}}}{metodo}Response")
    return respuesta.find(f"{{{NS_V1}}}{metodo}Result")


def _hijo(elemento: ET.Element | None, nombre: str) -> ET.Element | None:
    if elemento is None:
        return None
    return elemento.find(f"{{{NS_V1}}}{nombre}")


def _hijos(elemento: ET.Element | None, nombre: str) -> list[ET.Element]:
    if elemento is None:
        return []
    return elemento.findall(f"{{{NS_V1}}}{nombre}")


def _texto(elemento: ET.Element | None, nombre: str) -> str | None:
    return _valor_texto(_hijo(elemento, nombre))


def _valor_texto(elemento: ET.Element | None) -> str | None:
    """Texto de un elemento (enums como Quorum, Resultado, Tipo, OpcionVoto...)."""
    if elemento is None:
        return None
    return elemento.text


def _codigo_atributo(elemento: ET.Element | None) -> str | None:
    """Atributo Valor de un enum (<Quorum Valor="1">Quórum Simple</Quorum>)."""
    if elemento is None:
        return None
    return elemento.get("Valor")


@dataclass
class RespuestaCruda:
    datos: Any
    xml_crudo: str


def construir_dataframe_interim(filas: list[dict], modelo: type[ModeloInterim]) -> pd.DataFrame:
    """Aplica el modelo fila por fila y arma el DataFrame final. Si una fila
    falla, detiene todo con su número de fila — no descarta silenciosamente."""
    filas_validadas = []
    for numero_fila, fila in enumerate(filas, start=1):
        try:
            filas_validadas.append(modelo.from_dict(fila).to_interim_dict())
        except (TypeError, ValueError) as error:
            raise ValueError(f"{modelo.__name__}, fila {numero_fila}: {error}") from error
    return pd.DataFrame(filas_validadas, columns=modelo.COLUMNAS)


class ExtractorBase(ABC):
    """Template Method: mismo flujo para las 4 fuentes.
    descargar() -> única etapa con red, guarda el XML en data/raw/.
    procesar()  -> lee el XML ya guardado, parsea, valida, guarda el CSV.
    extraer()   -> descargar() + procesar(); se puede reprocesar sin red
    llamando procesar() de nuevo (por ejemplo, si cambia una validación)."""

    MODELO: type[ModeloInterim]  # cada subclase de salida única lo fija

    def __init__(self, base_dir: Path | str):
        self._base_dir = Path(base_dir)
        self._raw_dir = self._base_dir / "data" / "raw"
        self._interim_dir = self._base_dir / "data" / "interim"

    def extraer(self) -> pd.DataFrame:
        self.descargar()
        return self.procesar()

    def descargar(self) -> None:
        respuesta = self._consultar()
        self._guardar_crudo(respuesta)

    def procesar(self) -> pd.DataFrame:
        raiz = ET.parse(self._raw_path()).getroot()
        filas = self._parsear(raiz)
        df = construir_dataframe_interim(filas, self.MODELO)
        df = self._post_procesar(df)
        self._guardar_interim(df)
        return df

    @abstractmethod
    def _consultar(self) -> RespuestaCruda: ...

    @abstractmethod
    def _parsear(self, raiz: ET.Element) -> list[dict]:
        """XML crudo (ya en disco) -> lista de dicts, esquema de self.MODELO.COLUMNAS."""
        ...

    @abstractmethod
    def _raw_path(self) -> Path: ...

    @abstractmethod
    def _interim_path(self) -> Path: ...

    def _post_procesar(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _guardar_crudo(self, respuesta: RespuestaCruda) -> None:
        ruta = self._raw_path()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(respuesta.xml_crudo, encoding="utf-8")
        self._log(f"XML crudo guardado en: {ruta}")

    def _guardar_interim(self, df: pd.DataFrame) -> None:
        ruta = self._interim_path()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False, encoding="utf-8")
        self._log(f"Guardado en: {ruta} ({len(df)} filas)")

    def _log(self, mensaje: str) -> None:
        print(f"[{self.__class__.__name__}] {mensaje}")

    @staticmethod
    def _envelope_a_texto(envelope) -> str:
        return etree.tostring(
            envelope, pretty_print=True, xml_declaration=True, encoding="UTF-8"
        ).decode("utf-8")


class ExtractorPeriodosLegislativos(ExtractorBase):
    """Extractor: retornarPeriodosLegislativos vía SOAP."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"
    MODELO = PeriodoLegislativo

    def _consultar(self) -> RespuestaCruda:
        history = HistoryPlugin()
        cliente = Client(self.WSDL, plugins=[history])
        datos = cliente.service.retornarPeriodosLegislativos()
        xml_crudo = self._envelope_a_texto(history.last_received["envelope"])
        return RespuestaCruda(datos=datos, xml_crudo=xml_crudo)

    def _parsear(self, raiz: ET.Element) -> list[dict]:
        resultado = _resultado(raiz, "retornarPeriodosLegislativos")
        return [
            {
                "periodo_id": _texto(p, "Id"),
                "nombre": _texto(p, "Nombre"),
                "fecha_inicio": _texto(p, "FechaInicio"),
                "fecha_termino": _texto(p, "FechaTermino"),
            }
            for p in _hijos(resultado, "PeriodoLegislativo")
        ]

    def _post_procesar(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("fecha_inicio").reset_index(drop=True)

    def _raw_path(self) -> Path:
        return self._raw_dir / "periodos_legislativos.xml"

    def _interim_path(self) -> Path:
        return self._interim_dir / "periodos.csv"


class ExtractorVotacionesProyecto(ExtractorBase):
    """Extractor: retornarVotacionesXProyectoLey vía SOAP.

    Recibe un número de boletín y devuelve una fila por cada votación
    del proyecto de ley asociado (VotacionProyectoLey)."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"
    MODELO = VotacionProyecto

    def __init__(self, base_dir: Path | str, numero_boletin: str):
        super().__init__(base_dir)
        self._numero_boletin = str(numero_boletin)

    def _consultar(self) -> RespuestaCruda:
        history = HistoryPlugin()
        cliente = Client(self.WSDL, plugins=[history])
        datos = cliente.service.retornarVotacionesXProyectoLey(
            prmNumeroBoletin=self._numero_boletin
        )
        xml_crudo = self._envelope_a_texto(history.last_received["envelope"])
        return RespuestaCruda(datos=datos, xml_crudo=xml_crudo)

    def _parsear(self, raiz: ET.Element) -> list[dict]:
        resultado = _resultado(raiz, "retornarVotacionesXProyectoLey")
        votaciones = _hijos(_hijo(resultado, "Votaciones"), "VotacionProyectoLey")

        return [
            {
                "numero_boletin": self._numero_boletin,
                "Id": _texto(v, "Id"),
                "Descripcion": _texto(v, "Descripcion"),
                "Fecha": _texto(v, "Fecha"),
                "TotalSi": _texto(v, "TotalSi"),
                "TotalNo": _texto(v, "TotalNo"),
                "TotalAbstencion": _texto(v, "TotalAbstencion"),
                "TotalDispensado": _texto(v, "TotalDispensado"),
                "Quorum": _valor_texto(_hijo(v, "Quorum")),
                "Resultado": _valor_texto(_hijo(v, "Resultado")),
                "Tipo": _valor_texto(_hijo(v, "Tipo")),
                "TipoVotacionProyectoLey": _valor_texto(_hijo(v, "TipoVotacionProyectoLey")),
                "Articulo": _texto(v, "Articulo"),
                "TramiteConstitucional": _valor_texto(_hijo(v, "TramiteConstitucional")),
                "TramiteReglamentario": _valor_texto(_hijo(v, "TramiteReglamentario")),
            }
            for v in votaciones
        ]

    def _raw_path(self) -> Path:
        return self._raw_dir / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"

    def _interim_path(self) -> Path:
        return self._interim_dir / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"


class ExtractorDetalleVotaciones(ExtractorBase):
    """Extractor: retornarVotacionDetalle vía SOAP.

    A diferencia de los otros extractores, no es "1 consulta -> 1 XML ->
    1 CSV": recibe una lista de ids de votación (los `Id` que entrega
    ExtractorVotacionesProyecto) y hace una consulta por cada uno, cada
    una con su propio XML crudo. Por eso sobrescribe descargar()/procesar()
    en vez de solo _consultar()/_parsear()."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"
    MODELO = VotoNominal

    def __init__(self, base_dir: Path | str, ids_votaciones: list[int]):
        super().__init__(base_dir)
        self._ids_votaciones = list(ids_votaciones)

    def descargar(self) -> None:
        for votacion_id in self._ids_votaciones:
            respuesta = self._consultar_una(votacion_id)
            self._guardar_crudo_de(votacion_id, respuesta)

    def procesar(self) -> pd.DataFrame:
        filas: list[dict] = []
        for votacion_id in self._ids_votaciones:
            raiz = ET.parse(self._raw_path_de(votacion_id)).getroot()
            filas.extend(self._parsear(raiz))

        df = construir_dataframe_interim(filas, self.MODELO)
        df = self._post_procesar(df)
        self._guardar_interim(df)
        return df

    def _consultar_una(self, votacion_id: int) -> RespuestaCruda:
        history = HistoryPlugin()
        cliente = Client(self.WSDL, plugins=[history])
        datos = cliente.service.retornarVotacionDetalle(prmVotacionId=votacion_id)
        xml_crudo = self._envelope_a_texto(history.last_received["envelope"])
        return RespuestaCruda(datos=datos, xml_crudo=xml_crudo)

    def _parsear(self, raiz: ET.Element) -> list[dict]:
        votacion = _resultado(raiz, "retornarVotacionDetalle")
        votos = _hijos(_hijo(votacion, "Votos"), "Voto")

        filas = []
        for voto in votos:
            diputado = _hijo(voto, "Diputado")
            opcion = _hijo(voto, "OpcionVoto")
            filas.append(
                {
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
            )
        return filas

    def _raw_path_de(self, votacion_id: int) -> Path:
        return self._raw_dir / "votaciones" / f"votacion_{votacion_id}.xml"

    def _guardar_crudo_de(self, votacion_id: int, respuesta: RespuestaCruda) -> None:
        ruta = self._raw_path_de(votacion_id)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(respuesta.xml_crudo, encoding="utf-8")
        self._log(f"XML crudo guardado en: {ruta}")

    def _consultar(self) -> RespuestaCruda:
        raise NotImplementedError(
            "ExtractorDetalleVotaciones consulta una votación a la vez; "
            "usar _consultar_una(votacion_id)."
        )

    def _raw_path(self) -> Path:
        raise NotImplementedError(
            "Hay un XML por votación, no uno solo; ver _raw_path_de()."
        )

    def _interim_path(self) -> Path:
        return self._interim_dir / "votaciones" / "detalle_votaciones.csv"


class ExtractorDiputados(ExtractorBase):
    """Extractor: retornarDiputadosXPeriodo vía SOAP.

    Como en ExtractorDetalleVotaciones, no encaja en "1 consulta -> 1
    CSV": una sola consulta produce DOS salidas (diputados.csv y
    militancias.csv, ver esquemas.py), así que sobrescribe procesar()
    y define _parsear_militancias() además de _parsear()."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSDiputado.asmx?WSDL"
    MODELO = Diputado
    MODELO_MILITANCIA = Militancia

    def __init__(self, base_dir: Path | str, periodo_id: str):
        super().__init__(base_dir)
        self._periodo_id = str(periodo_id)

    def procesar(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raiz = ET.parse(self._raw_path()).getroot()

        df_diputados = construir_dataframe_interim(self._parsear(raiz), self.MODELO)
        df_diputados = self._post_procesar(df_diputados)
        self._guardar_interim(df_diputados)

        df_militancias = construir_dataframe_interim(
            self._parsear_militancias(raiz), self.MODELO_MILITANCIA
        )
        self._guardar_interim_militancias(df_militancias)

        return df_diputados, df_militancias

    def _consultar(self) -> RespuestaCruda:
        history = HistoryPlugin()
        cliente = Client(self.WSDL, plugins=[history])
        datos = cliente.service.retornarDiputadosXPeriodo(prmPeriodoID=self._periodo_id)
        xml_crudo = self._envelope_a_texto(history.last_received["envelope"])
        return RespuestaCruda(datos=datos, xml_crudo=xml_crudo)

    def _parsear(self, raiz: ET.Element) -> list[dict]:
        resultado = _resultado(raiz, "retornarDiputadosXPeriodo")
        filas = []
        for dp in _hijos(resultado, "DiputadoPeriodo"):
            d = _hijo(dp, "Diputado")
            sexo = _hijo(d, "Sexo")
            filas.append(
                {
                    "diputado_id": _texto(d, "Id"),
                    "nombre": _texto(d, "Nombre"),
                    "nombre2": _texto(d, "Nombre2"),
                    "apellido_paterno": _texto(d, "ApellidoPaterno"),
                    "apellido_materno": _texto(d, "ApellidoMaterno"),
                    "fecha_nacimiento": _texto(d, "FechaNacimiento"),
                    "rut": _texto(d, "RUT"),
                    "rut_dv": _texto(d, "RUTDV"),
                    "sexo_valor": _codigo_atributo(sexo),
                    "sexo_desc": _valor_texto(sexo),
                    "periodo_id": self._periodo_id,
                    "fecha_inicio_periodo": _texto(dp, "FechaInicio"),
                    "fecha_termino_periodo": _texto(dp, "FechaTermino"),
                }
            )
        return filas

    def _parsear_militancias(self, raiz: ET.Element) -> list[dict]:
        resultado = _resultado(raiz, "retornarDiputadosXPeriodo")
        filas = []
        for dp in _hijos(resultado, "DiputadoPeriodo"):
            d = _hijo(dp, "Diputado")
            for m in _hijos(_hijo(d, "Militancias"), "Militancia"):
                partido = _hijo(m, "Partido")
                filas.append(
                    {
                        "diputado_id": _texto(d, "Id"),
                        "partido_id": _texto(partido, "Id"),
                        "partido_nombre": _texto(partido, "Nombre"),
                        "partido_alias": _texto(partido, "Alias"),
                        "fecha_inicio": _texto(m, "FechaInicio"),
                        "fecha_termino": _texto(m, "FechaTermino"),
                    }
                )
        return filas

    def _guardar_interim_militancias(self, df: pd.DataFrame) -> None:
        ruta = self._interim_path_militancias()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False, encoding="utf-8")
        self._log(f"Guardado en: {ruta} ({len(df)} filas)")

    def _raw_path(self) -> Path:
        return self._raw_dir / "diputados" / f"diputados_periodo_{self._periodo_id}.xml"

    def _interim_path(self) -> Path:
        return self._interim_dir / "diputados.csv"

    def _interim_path_militancias(self) -> Path:
        return self._interim_dir / "militancias.csv"
