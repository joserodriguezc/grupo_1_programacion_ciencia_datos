"""Ejecuta reglas configurables sobre una tabla completa."""

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import pandas as pd

from F2.src.validaciones import (
    buscar_duplicados_clave,
    detectar_solapamientos_vigencia,
    resumir_completitud,
    validar_columnas_exactas,
)

from .registro_validacion import RegistroValidacion, ReporteValidacion

ReglaDataset = Callable[[pd.DataFrame, str], RegistroValidacion]


class ValidadorDataset:
    """Aplica todas las reglas de una tabla y reúne sus diagnósticos."""

    def __init__(
        self,
        reglas_por_tabla: Mapping[str, Sequence[ReglaDataset]],
    ) -> None:
        self._reglas_por_tabla = {
            tabla: tuple(reglas)
            for tabla, reglas in reglas_por_tabla.items()
        }

    def validar(self, tabla: str, df: pd.DataFrame) -> ReporteValidacion:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un DataFrame")

        if tabla not in self._reglas_por_tabla or not self._reglas_por_tabla[tabla]:
            raise ValueError(f"No hay reglas configuradas para {tabla!r}")

        registros: list[RegistroValidacion] = []

        for regla in self._reglas_por_tabla[tabla]:
            registro = regla(df, tabla)

            if not isinstance(registro, RegistroValidacion):
                raise TypeError(
                    "Cada regla debe devolver un RegistroValidacion"
                )

            if registro.tabla != tabla:
                raise ValueError(
                    f"La regla {registro.regla!r} devolvió la tabla "
                    f"{registro.tabla!r} en vez de {tabla!r}"
                )

            registros.append(registro)

        return ReporteValidacion(
            tabla=tabla,
            registros=tuple(registros),
        )


def regla_columnas_exactas(
    columnas_esperadas: Sequence[str],
) -> ReglaDataset:
    """Adapta el control de esquema de F2 al registro uniforme de F3."""
    esperadas = tuple(columnas_esperadas)

    def evaluar(
        df: pd.DataFrame,
        tabla: str,
    ) -> RegistroValidacion:
        resultado = validar_columnas_exactas(df, esperadas)
        valido = bool(resultado["valido"])

        detalle = (
            "Columnas y orden correctos"
            if valido
            else (
                f"Faltantes: {resultado['faltantes']}; "
                f"extras: {resultado['extras']}; "
                f"orden correcto: {resultado['orden_correcto']}"
            )
        )

        return RegistroValidacion(
            tabla=tabla,
            regla="columnas_exactas",
            estado="ok" if valido else "error",
            detalle=detalle,
            columnas=(
                tuple(cast(Sequence[str], resultado["observadas"]))
                if not valido
                else ()
            ),
        )

    return evaluar

def regla_clave_unica(
    columnas_clave: Sequence[str],
) -> ReglaDataset:
    """Registra las filas que repiten una clave simple o compuesta."""
    clave = tuple(columnas_clave)

    if not clave or len(set(clave)) != len(clave):
        raise ValueError(
            "La clave debe contener columnas distintas y no estar vacía"
        )

    def evaluar(
        df: pd.DataFrame,
        tabla: str,
    ) -> RegistroValidacion:
        faltantes = [
            columna for columna in clave
            if columna not in df.columns
        ]

        if faltantes:
            return RegistroValidacion(
                tabla=tabla,
                regla="clave_unica",
                estado="error",
                detalle=(
                    f"No se puede comprobar la clave {clave}: "
                    f"faltan {faltantes}"
                ),
                columnas=clave,
            )

        duplicados = buscar_duplicados_clave(df, clave)
        cantidad = len(duplicados)

        return RegistroValidacion(
            tabla=tabla,
            regla="clave_unica",
            estado="error" if cantidad else "ok",
            detalle=(
                f"{cantidad} filas repiten la clave {clave}"
                if cantidad
                else f"Clave {clave} sin duplicados"
            ),
            filas=tuple(duplicados.index.tolist()),
            columnas=clave if cantidad else (),
        )

    return evaluar

def regla_completitud(
    columnas_obligatorias: Sequence[str],
) -> ReglaDataset:
    """Detecta nulos y cadenas en blanco en los campos obligatorios."""
    obligatorias = tuple(columnas_obligatorias)

    if not obligatorias or len(set(obligatorias)) != len(obligatorias):
        raise ValueError(
            "Indica columnas obligatorias distintas y no vacías"
        )

    def evaluar(
        df: pd.DataFrame,
        tabla: str,
    ) -> RegistroValidacion:
        faltantes = [
            col for col in obligatorias
            if col not in df.columns
        ]

        if faltantes or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla=tabla,
                regla="completitud",
                estado="error",
                detalle=(
                    f"No se puede comprobar completitud: "
                    f"faltan {faltantes}; "
                    f"columnas duplicadas: {df.columns.has_duplicates}"
                ),
                columnas=obligatorias,
            )

        resumen = resumir_completitud(
            df.loc[:, list(obligatorias)]
        )
        nulos = dict(zip(resumen["columna"], resumen["nulos"]))

        filas_incompletas = pd.Series(False, index=df.index)
        columnas_afectadas: list[str] = []
        detalles: list[str] = []

        for columna in obligatorias:
            serie = df[columna]
            vacios = (
                serie.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )
            incompletos = serie.isna() | vacios
            filas_incompletas |= incompletos

            cantidad = int(incompletos.sum())

            if cantidad:
                columnas_afectadas.append(columna)
                detalles.append(
                    f"{columna}: {cantidad} vacíos "
                    f"({nulos[columna]} nulos originales)"
                )

        return RegistroValidacion(
            tabla=tabla,
            regla="completitud",
            estado="error" if columnas_afectadas else "ok",
            detalle=(
                "; ".join(detalles)
                if detalles
                else "Campos obligatorios completos"
            ),
            filas=tuple(df.index[filas_incompletas].tolist()),
            columnas=tuple(columnas_afectadas),
        )

    return evaluar

def regla_intervalos_validos(
    *,
    inicio_col: str = "fecha_inicio",
    termino_col: str = "fecha_termino",
    estado_invertido: str = "error",
    permitir_termino_abierto: bool = True,
) -> ReglaDataset:
    """Separa fechas inválidas (error) de intervalos invertidos (configurable)."""
    if estado_invertido not in ("advertencia", "error"):
        raise ValueError("estado_invertido debe ser 'advertencia' o 'error'")

    def evaluar(df: pd.DataFrame, tabla: str) -> RegistroValidacion:
        faltantes = [c for c in (inicio_col, termino_col) if c not in df.columns]
        if faltantes or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla=tabla,
                regla="intervalos_validos",
                estado="error",
                detalle=(
                    f"Faltan columnas {faltantes}; "
                    f"columnas duplicadas: {df.columns.has_duplicates}"
                ),
                columnas=(inicio_col, termino_col),
            )

        inicio = pd.to_datetime(df[inicio_col], errors="coerce", format="mixed")
        termino = pd.to_datetime(df[termino_col], errors="coerce", format="mixed")

        termino_abierto = df[termino_col].isna() | (
            df[termino_col].astype("string").str.strip().eq("").fillna(False)
        )
        termino_invalido = termino.isna() & ~termino_abierto
        if not permitir_termino_abierto:
            termino_invalido = termino.isna()

        invalidos = inicio.isna() | termino_invalido
        if invalidos.any():
            return RegistroValidacion(
                tabla=tabla,
                regla="intervalos_validos",
                estado="error",
                detalle=f"{int(invalidos.sum())} fila(s) con fecha ausente o inválida",
                filas=tuple(df.index[invalidos].tolist()),
                columnas=(inicio_col, termino_col),
            )

        invertidos = termino.notna() & (inicio > termino)
        return RegistroValidacion(
            tabla=tabla,
            regla="intervalos_validos",
            estado=estado_invertido if invertidos.any() else "ok",
            detalle=f"{int(invertidos.sum())} intervalo(s) invertido(s)",
            filas=tuple(df.index[invertidos].tolist()),
            columnas=(inicio_col, termino_col) if invertidos.any() else (),
        )

    return evaluar


def regla_solapamientos_vigencia(
    *,
    id_col: str = "diputado_id",
    inicio_col: str = "fecha_inicio",
    termino_col: str = "fecha_termino",
    estado_solapamiento: str = "error",
) -> ReglaDataset:
    """Detecta solapamientos entre intervalos válidos, con extremos incluidos."""
    if estado_solapamiento not in ("advertencia", "error"):
        raise ValueError("estado_solapamiento debe ser 'advertencia' o 'error'")

    columnas = (id_col, inicio_col, termino_col)

    def evaluar(df: pd.DataFrame, tabla: str) -> RegistroValidacion:
        faltantes = [c for c in columnas if c not in df.columns]
        if faltantes or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla=tabla,
                regla="solapamientos_vigencia",
                estado="error",
                detalle=(
                    f"Faltan columnas {faltantes}; "
                    f"columnas duplicadas: {df.columns.has_duplicates}"
                ),
                columnas=columnas,
            )

        inicio = pd.to_datetime(df[inicio_col], errors="coerce", format="mixed")
        termino = pd.to_datetime(df[termino_col], errors="coerce", format="mixed")
        invertidos = termino.notna() & (inicio > termino)

        try:
            conflictos = detectar_solapamientos_vigencia(
                df.loc[~invertidos],
                id_col=id_col,
                inicio_col=inicio_col,
                termino_col=termino_col,
            )
        except ValueError as exc:
            return RegistroValidacion(
                tabla=tabla,
                regla="solapamientos_vigencia",
                estado="error",
                detalle=str(exc),
                columnas=columnas,
            )

        if conflictos.empty:
            return RegistroValidacion(
                tabla=tabla,
                regla="solapamientos_vigencia",
                estado="ok",
                detalle="No hay vigencias superpuestas",
            )

        indices = (
            indice
            for par in conflictos[["indice_a", "indice_b"]].itertuples(
                index=False, name=None
            )
            for indice in par
        )
        filas = tuple(dict.fromkeys(indices))

        return RegistroValidacion(
            tabla=tabla,
            regla="solapamientos_vigencia",
            estado=estado_solapamiento,
            detalle=f"{len(conflictos)} solapamiento(s) de vigencias",
            filas=filas,
            columnas=columnas,
        )

    return evaluar

def regla_fechas_validas(
    obligatorias: Sequence[str], opcionales: Sequence[str] = ()
) -> ReglaDataset:
    """Comprueba fechas sin convertir ni sobrescribir los datos de origen."""
    requeridas = tuple(obligatorias) + tuple(opcionales)
    if not requeridas or len(requeridas) != len(set(requeridas)):
        raise ValueError("Indica columnas de fecha distintas y no vacías")

    def evaluar(df: pd.DataFrame, tabla: str) -> RegistroValidacion:
        faltantes = [c for c in requeridas if c not in df.columns]
        if faltantes or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla=tabla,
                regla="fechas_validas",
                estado="error",
                detalle=f"Faltan {faltantes}; columnas duplicadas: {df.columns.has_duplicates}",
                columnas=requeridas,
            )

        filas: list[object] = []
        columnas: list[str] = []

        for columna in requeridas:
            serie = df[columna]
            vacios = serie.isna() | (
                serie.astype("string").str.strip().eq("").fillna(False)
            )
            parsed = pd.to_datetime(serie, errors="coerce", format="mixed")
            invalidos = parsed.isna() & (
                ~vacios
                if columna in opcionales
                else pd.Series(True, index=df.index)
            )

            if invalidos.any():
                columnas.append(columna)
                filas.extend(df.index[invalidos].tolist())

        return RegistroValidacion(
            tabla=tabla,
            regla="fechas_validas",
            estado="error" if columnas else "ok",
            detalle=f"Fechas inválidas en {columnas}" if columnas else "Fechas válidas",
            filas=tuple(dict.fromkeys(filas)),
            columnas=tuple(columnas),
        )

    return evaluar


def regla_enteros_no_negativos(columnas_numericas: Sequence[str]) -> ReglaDataset:
    """Comprueba conteos e identificadores decimales sin hacer casting."""
    columnas_esperadas = tuple(columnas_numericas)
    if not columnas_esperadas or len(set(columnas_esperadas)) != len(columnas_esperadas):
        raise ValueError("Indica columnas numéricas distintas y no vacías")

    def evaluar(df: pd.DataFrame, tabla: str) -> RegistroValidacion:
        faltantes = [c for c in columnas_esperadas if c not in df.columns]
        if faltantes or df.columns.has_duplicates:
            return RegistroValidacion(
                tabla=tabla,
                regla="enteros_no_negativos",
                estado="error",
                detalle=f"Faltan {faltantes}; columnas duplicadas: {df.columns.has_duplicates}",
                columnas=columnas_esperadas,
            )

        filas: list[object] = []
        columnas: list[str] = []

        for columna in columnas_esperadas:
            valido = (
                df[columna]
                .astype("string")
                .str.fullmatch(r"[0-9]+")
                .fillna(False)
            )
            if (~valido).any():
                columnas.append(columna)
                filas.extend(df.index[~valido].tolist())

        return RegistroValidacion(
            tabla=tabla,
            regla="enteros_no_negativos",
            estado="error" if columnas else "ok",
            detalle=f"Enteros inválidos en {columnas}" if columnas else "Enteros válidos",
            filas=tuple(dict.fromkeys(filas)),
            columnas=tuple(columnas),
        )

    return evaluar

def regla_correspondencia_codigos(
    codigo: str,
    descripcion: str,
    *,
    estado_inconsistencia: str = "advertencia",
) -> ReglaDataset:
    """Informa códigos con varias descripciones y descripciones con varios códigos."""
    if codigo == descripcion:
        raise ValueError("Las columnas de código y descripción deben ser distintas")
    if estado_inconsistencia not in ("advertencia", "error"):
        raise ValueError("estado_inconsistencia debe ser 'advertencia' o 'error'")

    def evaluar(df: pd.DataFrame, tabla: str) -> RegistroValidacion:
        if (
            codigo not in df.columns
            or descripcion not in df.columns
            or df.columns.has_duplicates
        ):
            return RegistroValidacion(
                tabla=tabla,
                regla=f"correspondencia.{codigo}.{descripcion}",
                estado="error",
                detalle=(
                    "Faltan columnas para comprobar la correspondencia "
                    "o hay columnas duplicadas"
                ),
                columnas=(codigo, descripcion),
            )

        codigos = df.groupby(codigo, dropna=False)[descripcion].nunique(
            dropna=False
        )
        descripciones = df.groupby(descripcion, dropna=False)[codigo].nunique(
            dropna=False
        )

        codigos_ambiguos = codigos.index[codigos > 1]
        descripciones_ambiguas = descripciones.index[descripciones > 1]

        afectadas = df[codigo].isin(codigos_ambiguos) | df[descripcion].isin(
            descripciones_ambiguas
        )
        filas = tuple(df.index[afectadas].tolist())

        return RegistroValidacion(
            tabla=tabla,
            regla=f"correspondencia.{codigo}.{descripcion}",
            estado=estado_inconsistencia if filas else "ok",
            detalle=(
                f"{len(codigos_ambiguos)} código(s) con varias descripciones; "
                f"{len(descripciones_ambiguas)} descripción(es) con varios códigos"
            ),
            filas=filas,
            columnas=(codigo, descripcion) if filas else (),
        )

    return evaluar