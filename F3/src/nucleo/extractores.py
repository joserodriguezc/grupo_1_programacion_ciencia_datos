# F3/src/nucleo/extractores.py
from __future__ import annotations

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


def _texto_valor(campo: Any) -> Any:
    """Los enums de la API (Quorum, Resultado, Tipo, OpcionVoto, ...)
    llegan como {'_value_1': 'Quórum Calificado', 'Valor': 2}; esto
    devuelve el texto legible. None se propaga tal cual."""
    if campo is None:
        return None
    return campo["_value_1"]


def _codigo_valor(campo: Any) -> Any:
    """Mismo enum, pero el código numérico/entero asociado."""
    if campo is None:
        return None
    return campo["Valor"]


def _como_lista(contenedor: Any, clave: str) -> list:
    """Los listados de la API vienen como {'Voto': [...]} pero si sólo
    hay un elemento, zeep a veces lo entrega sin envolver en lista."""
    if contenedor is None:
        return []
    elementos = contenedor[clave]
    if elementos is None:
        return []
    if not isinstance(elementos, list):
        return [elementos]
    return elementos


def _normalizar_lista(valor: Any) -> list:
    """Igual que _como_lista, pero para un resultado que ya es la lista
    (o el único elemento) en sí, sin contenedor con clave alrededor —
    el caso de retornarDiputadosXPeriodo."""
    if valor is None:
        return []
    if not isinstance(valor, list):
        return [valor]
    return valor


@dataclass
class RespuestaCruda:
    datos: Any
    xml_crudo: str


def construir_dataframe_interim(filas: list[dict], modelo: type[ModeloInterim]) -> pd.DataFrame:
    """Aplica el modelo fila por fila (XML->dict ya hecho antes) y arma
    el DataFrame final. Si una fila falla, detiene todo con su número
    de fila — no descarta silenciosamente."""
    filas_validadas = []
    for numero_fila, fila in enumerate(filas, start=1):
        try:
            filas_validadas.append(modelo.from_dict(fila).to_interim_dict())
        except (TypeError, ValueError) as error:
            raise ValueError(f"{modelo.__name__}, fila {numero_fila}: {error}") from error
    return pd.DataFrame(filas_validadas, columns=modelo.COLUMNAS)


class ExtractorBase(ABC):
    """Template Method: mismo flujo para las 4 fuentes.
    consultar -> guardar crudo -> parsear a filas (dict) -> validar con
    el dataclass del modelo -> DataFrame -> guardar en data/interim/.
    """

    MODELO: type[ModeloInterim]  # cada subclase de salida única lo fija

    def __init__(self, base_dir: Path | str):
        self._base_dir = Path(base_dir)
        self._raw_dir = self._base_dir / "data" / "raw"
        self._interim_dir = self._base_dir / "data" / "interim"

    def extraer(self) -> pd.DataFrame:
        respuesta = self._consultar()
        self._guardar_crudo(respuesta)
        filas = self._parsear(respuesta)
        df = construir_dataframe_interim(filas, self.MODELO)
        df = self._post_procesar(df)
        self._guardar_interim(df)
        return df

    @abstractmethod
    def _consultar(self) -> RespuestaCruda: ...

    @abstractmethod
    def _parsear(self, respuesta: RespuestaCruda) -> list[dict]:
        """XML/respuesta SOAP -> lista de dicts. Mismo esquema que espera
        self.MODELO.COLUMNAS. Válido tanto para la variante ElementTree
        como para la recursiva (etapa 3): ambas deben entregar dicts
        con las mismas claves."""
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

    def _parsear(self, respuesta: RespuestaCruda) -> list[dict]:
        return [
            {
                "periodo_id": p["Id"],
                "nombre": p["Nombre"],
                "fecha_inicio": p["FechaInicio"],
                "fecha_termino": p["FechaTermino"],
            }
            for p in respuesta.datos
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

    def _parsear(self, respuesta: RespuestaCruda) -> list[dict]:
        proyecto = respuesta.datos
        votaciones = _como_lista(proyecto["Votaciones"], "VotacionProyectoLey")

        return [
            {
                "numero_boletin": self._numero_boletin,
                "Id": v["Id"],
                "Descripcion": v["Descripcion"],
                "Fecha": v["Fecha"],
                "TotalSi": v["TotalSi"],
                "TotalNo": v["TotalNo"],
                "TotalAbstencion": v["TotalAbstencion"],
                "TotalDispensado": v["TotalDispensado"],
                "Quorum": _texto_valor(v["Quorum"]),
                "Resultado": _texto_valor(v["Resultado"]),
                "Tipo": _texto_valor(v["Tipo"]),
                "TipoVotacionProyectoLey": _texto_valor(v["TipoVotacionProyectoLey"]),
                "Articulo": v["Articulo"],
                "TramiteConstitucional": _texto_valor(v["TramiteConstitucional"]),
                "TramiteReglamentario": _texto_valor(v["TramiteReglamentario"]),
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
    una con su propio XML crudo. Por eso sobrescribe extraer() completo
    en vez de sólo _consultar()/_parsear()."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"
    MODELO = VotoNominal

    def __init__(self, base_dir: Path | str, ids_votaciones: list[int]):
        super().__init__(base_dir)
        self._ids_votaciones = list(ids_votaciones)

    def extraer(self) -> pd.DataFrame:
        filas: list[dict] = []
        for votacion_id in self._ids_votaciones:
            respuesta = self._consultar_una(votacion_id)
            self._guardar_crudo_de(votacion_id, respuesta)
            filas.extend(self._parsear(respuesta))

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

    def _parsear(self, respuesta: RespuestaCruda) -> list[dict]:
        votacion = respuesta.datos
        votos = _como_lista(votacion["Votos"], "Voto")

        filas = []
        for voto in votos:
            diputado = voto["Diputado"]
            opcion = voto["OpcionVoto"]
            filas.append(
                {
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
            )
        return filas

    def _guardar_crudo_de(self, votacion_id: int, respuesta: RespuestaCruda) -> None:
        ruta = self._raw_dir / "votaciones" / f"votacion_{votacion_id}.xml"
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
            "Hay un XML por votación, no uno solo; ver _guardar_crudo_de()."
        )

    def _interim_path(self) -> Path:
        return self._interim_dir / "votaciones" / "detalle_votaciones.csv"


class ExtractorDiputados(ExtractorBase):
    """Extractor: retornarDiputadosXPeriodo vía SOAP.

    Como en ExtractorDetalleVotaciones, no encaja en "1 consulta -> 1
    CSV": una sola consulta produce DOS salidas (diputados.csv y
    militancias.csv, ver esquemas.py), así que sobrescribe extraer()
    y define _parsear_militancias() además de _parsear()."""

    WSDL = "https://opendata.camara.cl/camaradiputados/WServices/WSDiputado.asmx?WSDL"
    MODELO = Diputado
    MODELO_MILITANCIA = Militancia

    def __init__(self, base_dir: Path | str, periodo_id: str):
        super().__init__(base_dir)
        self._periodo_id = str(periodo_id)

    def extraer(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        respuesta = self._consultar()
        self._guardar_crudo(respuesta)

        df_diputados = construir_dataframe_interim(self._parsear(respuesta), self.MODELO)
        df_diputados = self._post_procesar(df_diputados)
        self._guardar_interim(df_diputados)

        df_militancias = construir_dataframe_interim(
            self._parsear_militancias(respuesta), self.MODELO_MILITANCIA
        )
        self._guardar_interim_militancias(df_militancias)

        return df_diputados, df_militancias

    def _consultar(self) -> RespuestaCruda:
        history = HistoryPlugin()
        cliente = Client(self.WSDL, plugins=[history])
        datos = cliente.service.retornarDiputadosXPeriodo(prmPeriodoID=self._periodo_id)
        xml_crudo = self._envelope_a_texto(history.last_received["envelope"])
        return RespuestaCruda(datos=datos, xml_crudo=xml_crudo)

    def _parsear(self, respuesta: RespuestaCruda) -> list[dict]:
        filas = []
        for dp in _normalizar_lista(respuesta.datos):
            d = dp["Diputado"]
            sexo = d["Sexo"]
            filas.append(
                {
                    "diputado_id": d["Id"],
                    "nombre": d["Nombre"],
                    "nombre2": d["Nombre2"],
                    "apellido_paterno": d["ApellidoPaterno"],
                    "apellido_materno": d["ApellidoMaterno"],
                    "fecha_nacimiento": d["FechaNacimiento"],
                    "rut": d["RUT"],
                    "rut_dv": d["RUTDV"],
                    "sexo_valor": _codigo_valor(sexo),
                    "sexo_desc": _texto_valor(sexo),
                    "periodo_id": self._periodo_id,
                    "fecha_inicio_periodo": dp["FechaInicio"],
                    "fecha_termino_periodo": dp["FechaTermino"],
                }
            )
        return filas

    def _parsear_militancias(self, respuesta: RespuestaCruda) -> list[dict]:
        filas = []
        for dp in _normalizar_lista(respuesta.datos):
            d = dp["Diputado"]
            for m in _como_lista(d["Militancias"], "Militancia"):
                partido = m["Partido"]
                filas.append(
                    {
                        "diputado_id": d["Id"],
                        "partido_id": partido["Id"],
                        "partido_nombre": partido["Nombre"],
                        "partido_alias": partido["Alias"],
                        "fecha_inicio": m["FechaInicio"],
                        "fecha_termino": m["FechaTermino"],
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