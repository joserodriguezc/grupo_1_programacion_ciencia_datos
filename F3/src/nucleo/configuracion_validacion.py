"""Contratos explícitos para los cinco CSV de entrada interim."""

from .esquemas import (
    COLUMNAS_DIPUTADO,
    COLUMNAS_MILITANCIA,
    COLUMNAS_PERIODO_LEGISLATIVO,
    COLUMNAS_VOTACION_PROYECTO,
    COLUMNAS_VOTO_NOMINAL,
)
from .registro_validacion import RegistroValidacion
from .validacion_dataset import (
    ReglaDataset,
    ValidadorDataset,
    regla_clave_unica,
    regla_columnas_exactas,
    regla_completitud,
    regla_correspondencia_codigos,
    regla_enteros_no_negativos,
    regla_fechas_validas,
    regla_intervalos_validos,
    regla_solapamientos_vigencia,
)


def _regla_sexo() -> ReglaDataset:
    def evaluar(df, tabla):
        if "sexo_valor" not in df.columns or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla,
                "sexo_valor",
                "error",
                "Falta sexo_valor o hay columnas duplicadas",
                columnas=("sexo_valor",),
            )

        invalidos = ~df["sexo_valor"].astype("string").isin(["0", "1"])
        return RegistroValidacion(
            tabla,
            "sexo_valor",
            "error" if invalidos.any() else "ok",
            "Código de sexo desconocido"
            if invalidos.any()
            else "Código de sexo válido",
            filas=tuple(df.index[invalidos].tolist()),
            columnas=("sexo_valor",) if invalidos.any() else (),
        )

    return evaluar


def _reglas_comunes(columnas, clave, opcionales=()):
    obligatorias = tuple(c for c in columnas if c not in opcionales)
    return (
        regla_columnas_exactas(columnas),
        regla_completitud(obligatorias),
        regla_clave_unica(clave),
    )


def construir_validador_interim() -> ValidadorDataset:
    """Crea reglas para entradas interim; no lee archivos ni modifica tablas."""
    return ValidadorDataset(
        {
            "periodos": _reglas_comunes(
                COLUMNAS_PERIODO_LEGISLATIVO,
                ("periodo_id",),
            ) + (
                regla_enteros_no_negativos(("periodo_id",)),
                regla_intervalos_validos(permitir_termino_abierto=False),
            ),
            "diputados": _reglas_comunes(
                COLUMNAS_DIPUTADO,
                ("diputado_id",),
                ("nombre2", "rut", "rut_dv", "fecha_termino_periodo"),
            ) + (
                regla_enteros_no_negativos(("diputado_id", "periodo_id")),
                regla_fechas_validas(
                    ("fecha_nacimiento", "fecha_inicio_periodo"),
                    ("fecha_termino_periodo",),
                ),
                _regla_sexo(),
            ),
            "militancias": _reglas_comunes(
                COLUMNAS_MILITANCIA,
                ("diputado_id", "partido_id", "fecha_inicio"),
                ("fecha_termino",),
            ) + (
                regla_enteros_no_negativos(("diputado_id",)),
                regla_intervalos_validos(estado_invertido="advertencia"),
                regla_solapamientos_vigencia(
                    estado_solapamiento="advertencia"
                ),
                regla_correspondencia_codigos("partido_id", "partido_nombre"),
            ),
            "proyecto_ley": _reglas_comunes(
                COLUMNAS_VOTACION_PROYECTO,
                ("Id",),
                ("Articulo",),
            ) + (
                regla_enteros_no_negativos(
                    (
                        "Id",
                        "TotalSi",
                        "TotalNo",
                        "TotalAbstencion",
                        "TotalDispensado",
                    )
                ),
                regla_fechas_validas(("Fecha",)),
            ),
            "detalle_votaciones": _reglas_comunes(
                COLUMNAS_VOTO_NOMINAL,
                ("votacion_id", "diputado_id"),
                ("nombre2",),
            ) + (
                regla_enteros_no_negativos(
                    (
                        "votacion_id",
                        "diputado_id",
                        "opcion_codigo",
                        "quorum_codigo",
                        "resultado_codigo",
                        "tipo_codigo",
                        "total_si",
                        "total_no",
                        "total_abstencion",
                        "total_dispensado",
                    )
                ),
                regla_fechas_validas(("fecha",)),
                regla_correspondencia_codigos("opcion_codigo", "opcion_voto"),
                regla_correspondencia_codigos("quorum_codigo", "quorum"),
                regla_correspondencia_codigos("resultado_codigo", "resultado"),
                regla_correspondencia_codigos("tipo_codigo", "tipo"),
            ),
        }
    )