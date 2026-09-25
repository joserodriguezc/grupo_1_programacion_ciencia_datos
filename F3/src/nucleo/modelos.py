"""Modelos de las filas que se escribirán en los CSV interim."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Self

from .esquemas import (
    COLUMNAS_DIPUTADO,
    COLUMNAS_MILITANCIA,
    COLUMNAS_PERIODO_LEGISLATIVO,
    COLUMNAS_VOTACION_PROYECTO,
    COLUMNAS_VOTO_NOMINAL,
)

from ._campos import (
    entero_no_negativo,
    fecha_iso,
    texto_obligatorio,
    texto_opcional,
)
class ModeloInterim:
    """Contrato común para construir y serializar una fila."""

    COLUMNAS: ClassVar[tuple[str, ...]]

    @classmethod
    def from_dict(cls, fila: Mapping[str, object]) -> Self:
        if not isinstance(fila, Mapping):
            raise TypeError(f"{cls.__name__}: se esperaba un diccionario de fila")

        faltantes = set(cls.COLUMNAS) - fila.keys()
        adicionales = fila.keys() - set(cls.COLUMNAS)

        if faltantes or adicionales:
            raise ValueError(
                f"{cls.__name__}: columnas faltantes={sorted(faltantes)}, "
                f"columnas adicionales={sorted(adicionales)}"
            )

        # La dataclass recibe los valores; __post_init__ valida la fila.
        return cls(**{columna: fila[columna] for columna in cls.COLUMNAS})

    def to_interim_dict(self) -> dict[str, object]:
        """Devuelve las columnas con los nombres y el orden exigidos por F2."""
        return {columna: getattr(self, columna) for columna in self.COLUMNAS}


@dataclass(frozen=True)
class PeriodoLegislativo(ModeloInterim):
    periodo_id: str
    nombre: str
    fecha_inicio: str
    fecha_termino: str

    COLUMNAS: ClassVar[tuple[str, ...]] = COLUMNAS_PERIODO_LEGISLATIVO
    
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "periodo_id",
            entero_no_negativo(
                self.periodo_id,
                "PeriodoLegislativo.periodo_id",
            ),
        )

        texto_obligatorio(self.nombre, "PeriodoLegislativo.nombre")

        for campo in ("fecha_inicio", "fecha_termino"):
            object.__setattr__(
                self,
                campo,
                fecha_iso(
                    getattr(self, campo),
                    f"PeriodoLegislativo.{campo}",
                ),
            )
            
@dataclass(frozen=True)
class Diputado(ModeloInterim):
    diputado_id: str
    nombre: str
    nombre2: str
    apellido_paterno: str
    apellido_materno: str
    fecha_nacimiento: str
    rut: str
    rut_dv: str
    sexo_valor: str
    sexo_desc: str
    periodo_id: str
    fecha_inicio_periodo: str
    fecha_termino_periodo: str

    COLUMNAS: ClassVar[tuple[str, ...]] = COLUMNAS_DIPUTADO

    def __post_init__(self) -> None:
        for campo in ("diputado_id", "periodo_id"):
            object.__setattr__(
                self,
                campo,
                entero_no_negativo(
                    getattr(self, campo),
                    f"Diputado.{campo}",
                ),
            )

        for campo in (
            "nombre",
            "apellido_paterno",
            "apellido_materno",
            "sexo_desc",
        ):
            texto_obligatorio(getattr(self, campo), f"Diputado.{campo}")

        for campo in ("nombre2", "rut", "rut_dv"):
            object.__setattr__(
                self,
                campo,
                texto_opcional(getattr(self, campo), f"Diputado.{campo}"),
            )

        sexo = entero_no_negativo(
            self.sexo_valor,
            "Diputado.sexo_valor",
        )
        if sexo not in {"0", "1"}:
            raise ValueError(
                f"Diputado.sexo_valor: código desconocido {sexo!r}"
            )
        object.__setattr__(self, "sexo_valor", sexo)

        for campo in (
            "fecha_nacimiento",
            "fecha_inicio_periodo",
            "fecha_termino_periodo",
        ):
            object.__setattr__(
                self,
                campo,
                fecha_iso(
                    getattr(self, campo),
                    f"Diputado.{campo}",
                    permitir_vacia=(campo == "fecha_termino_periodo"),
                ),
            )

@dataclass(frozen=True)
class Militancia(ModeloInterim):
    diputado_id: str
    partido_id: str
    partido_nombre: str
    partido_alias: str
    fecha_inicio: str
    fecha_termino: str

    COLUMNAS: ClassVar[tuple[str, ...]] = COLUMNAS_MILITANCIA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diputado_id",
            entero_no_negativo(
                self.diputado_id,
                "Militancia.diputado_id",
            ),
        )

        # partido_id sigue siendo texto: "PC", "IND", etc.
        for campo in ("partido_id", "partido_nombre", "partido_alias"):
            texto_obligatorio(getattr(self, campo), f"Militancia.{campo}")

        for campo in ("fecha_inicio", "fecha_termino"):
            object.__setattr__(
                self,
                campo,
                fecha_iso(
                    getattr(self, campo),
                    f"Militancia.{campo}",
                    permitir_vacia=(campo == "fecha_termino"),
                ),
            )

@dataclass(frozen=True)
class VotacionProyecto(ModeloInterim):
    numero_boletin: str
    Id: str
    Descripcion: str
    Fecha: str
    TotalSi: str
    TotalNo: str
    TotalAbstencion: str
    TotalDispensado: str
    Quorum: str
    Resultado: str
    Tipo: str
    TipoVotacionProyectoLey: str
    Articulo: str
    TramiteConstitucional: str
    TramiteReglamentario: str

    COLUMNAS: ClassVar[tuple[str, ...]] = COLUMNAS_VOTACION_PROYECTO

    def __post_init__(self) -> None:
        for campo in (
            "Id",
            "TotalSi",
            "TotalNo",
            "TotalAbstencion",
            "TotalDispensado",
        ):
            object.__setattr__(
                self,
                campo,
                entero_no_negativo(
                    getattr(self, campo),
                    f"VotacionProyecto.{campo}",
                ),
            )

        for campo in (
            "numero_boletin",
            "Descripcion",
            "Quorum",
            "Resultado",
            "Tipo",
            "TipoVotacionProyectoLey",
            "TramiteConstitucional",
            "TramiteReglamentario",
        ):
            texto_obligatorio(
                getattr(self, campo),
                f"VotacionProyecto.{campo}",
            )

        object.__setattr__(
            self,
            "Articulo",
            texto_opcional(self.Articulo, "VotacionProyecto.Articulo"),
        )
        object.__setattr__(
            self,
            "Fecha",
            fecha_iso(
                self.Fecha,
                "VotacionProyecto.Fecha",
                separador_datetime="T",
            ),
        )
@dataclass(frozen=True)
class VotoNominal(ModeloInterim):
    diputado_id: str
    nombre: str
    nombre2: str
    apellido_paterno: str
    apellido_materno: str
    opcion_codigo: str
    opcion_voto: str
    votacion_id: str
    descripcion: str
    fecha: str
    total_si: str
    total_no: str
    total_abstencion: str
    total_dispensado: str
    quorum_codigo: str
    quorum: str
    resultado_codigo: str
    resultado: str
    tipo_codigo: str
    tipo: str

    COLUMNAS: ClassVar[tuple[str, ...]] = COLUMNAS_VOTO_NOMINAL

    def __post_init__(self) -> None:
        for campo in (
            "diputado_id",
            "opcion_codigo",
            "votacion_id",
            "total_si",
            "total_no",
            "total_abstencion",
            "total_dispensado",
            "quorum_codigo",
            "resultado_codigo",
            "tipo_codigo",
        ):
            object.__setattr__(
                self,
                campo,
                entero_no_negativo(
                    getattr(self, campo),
                    f"VotoNominal.{campo}",
                ),
            )

        for campo in (
            "nombre",
            "apellido_paterno",
            "apellido_materno",
            "opcion_voto",
            "descripcion",
            "quorum",
            "resultado",
            "tipo",
        ):
            texto_obligatorio(getattr(self, campo), f"VotoNominal.{campo}")

        object.__setattr__(
            self,
            "nombre2",
            texto_opcional(self.nombre2, "VotoNominal.nombre2"),
        )
        object.__setattr__(
            self,
            "fecha",
            fecha_iso(
                self.fecha,
                "VotoNominal.fecha",
                separador_datetime="T",
            ),
        )