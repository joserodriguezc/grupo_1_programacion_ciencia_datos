"""Validación de campos individuales de las filas interim."""

from datetime import date, datetime


def entero_no_negativo(valor: object, campo: str) -> str:
    if isinstance(valor, bool) or not isinstance(valor, (int, str)):
        raise ValueError(f"{campo}: entero inválido {valor!r}")

    texto = str(valor)
    if not texto.isdecimal():
        raise ValueError(f"{campo}: se esperaba un entero no negativo; recibido {valor!r}")

    return texto


def texto_obligatorio(valor: object, campo: str) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{campo}: texto obligatorio inválido {valor!r}")

    return valor


def texto_opcional(valor: object, campo: str) -> str:
    if valor is None:
        return ""

    if not isinstance(valor, str):
        raise ValueError(f"{campo}: se esperaba texto o None; recibido {valor!r}")

    return valor


def fecha_iso(
    valor: object,
    campo: str,
    *,
    permitir_vacia: bool = False,
    separador_datetime: str = " ",
) -> str:
    if permitir_vacia and (valor is None or valor == ""):
        return ""

    if isinstance(valor, datetime):
        valor = valor.isoformat(sep=separador_datetime)
    elif isinstance(valor, date):
        valor = valor.isoformat()

    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{campo}: fecha obligatoria inválida {valor!r}")

    try:
        datetime.fromisoformat(valor)
    except ValueError as error:
        raise ValueError(f"{campo}: fecha ISO inválida {valor!r}") from error

    return valor
