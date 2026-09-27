"""Resultados estructurados de las reglas de calidad de datos de F3."""

from dataclasses import dataclass
from typing import Literal

EstadoValidacion = Literal["ok", "advertencia", "error"]


@dataclass(frozen=True, slots=True)
class RegistroValidacion:
    """Resultado de aplicar una regla a una tabla."""

    tabla: str
    regla: str
    estado: EstadoValidacion
    detalle: str
    filas: tuple[object, ...] = ()
    columnas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.estado not in ("ok", "advertencia", "error"):
            raise ValueError(
                f"Estado de validación desconocido: {self.estado!r}"
            )

    @property
    def bloquea(self) -> bool:
        return self.estado == "error"


@dataclass(frozen=True, slots=True)
class ReporteValidacion:
    """Agrupa los resultados de las reglas aplicadas a una tabla."""

    tabla: str
    registros: tuple[RegistroValidacion, ...]

    @property
    def aprobado(self) -> bool:
        return not any(registro.bloquea for registro in self.registros)

    @property
    def errores(self) -> tuple[RegistroValidacion, ...]:
        return tuple(
            registro for registro in self.registros
            if registro.bloquea
        )

    @property
    def advertencias(self) -> tuple[RegistroValidacion, ...]:
        return tuple(
            registro for registro in self.registros
            if registro.estado == "advertencia"
        )