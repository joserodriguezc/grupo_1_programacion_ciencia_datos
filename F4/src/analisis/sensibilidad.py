from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from F4.src.analisis.bcall import ModeloBCall, ResultadoBCall
from F4.src.analisis.cohesion import (
    IndicesCohesion,
    ParametrosCohesion,
    ResumenCohesion,
)
from F4.src.analisis.cohesion import desde_repositorio as cohesion_desde_repositorio
from F4.src.analisis.posicion_partidos import (
    PosicionPartido,
    ResultadoPosicionPartidos,
)

RUTA_MATRIZ_POR_DEFECTO = Path("F4/data/processed/matriz_ternaria.csv")
RUTA_AFILIACION_POR_DEFECTO = Path("F4/data/processed/afiliacion_por_votacion.csv")
RUTA_CONFIG_POR_DEFECTO = Path("F4/config/analisis.toml")
RUTA_SALIDA_POR_DEFECTO = Path("F4/data/reports/sensibilidad.csv")


class ErrorSensibilidad(ValueError):
    """La sensibilidad no puede estimarse respetando el contrato del proyecto."""


@dataclass(frozen=True, slots=True)
class ResultadoEscenario:
    """Resultados recalculados para un escenario completo."""

    bcall: ResultadoBCall
    partidos: ResultadoPosicionPartidos
    n_columnas_entrada: int
    n_votaciones_utilizables: int


class AnalisisRobustez:
    """Recalcula resultados bajo retiro de votaciones y umbrales alternativos."""

    def __init__(
        self,
        *,
        pivot: Hashable,
        metodo_distancia: int,
        threshold_bcall: float,
        min_decisiones_partido_votacion: int,
        min_votaciones_partido: int,
        umbrales_cobertura: tuple[float, ...] = (0.40, 0.60, 0.80),
    ) -> None:
        if not umbrales_cobertura:
            raise ValueError("Debe existir al menos un umbral de cobertura.")

        for umbral in umbrales_cobertura:
            if not 0 <= float(umbral) <= 1:
                raise ValueError("Los umbrales de cobertura deben estar entre 0 y 1.")

        self.pivot = pivot
        self.metodo_distancia = metodo_distancia
        self.threshold_bcall = float(threshold_bcall)
        self.min_decisiones_partido_votacion = min_decisiones_partido_votacion
        self.min_votaciones_partido = min_votaciones_partido
        self.umbrales_cobertura = tuple(float(x) for x in umbrales_cobertura)

    def calcular(
        self,
        matriz_ternaria: pd.DataFrame,
        afiliacion: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calcula sensibilidad leave-one-vote-out de d1, d2 y P_p."""
        matriz = self._validar_matriz(matriz_ternaria)
        self._validar_afiliacion(afiliacion)

        base = self._ejecutar_escenario(matriz, afiliacion)

        filas: list[dict[str, Any]] = []

        for votacion_retirada in matriz.columns:
            matriz_alt = matriz.drop(columns=[votacion_retirada])
            afiliacion_alt = afiliacion[
                afiliacion["votacion_id"].map(_clave_id)
                != _clave_id(votacion_retirada)
            ].copy()

            try:
                alternativo = self._ejecutar_escenario(
                    matriz_alt,
                    afiliacion_alt,
                )
            except (ValueError, ErrorSensibilidad) as exc:
                filas.extend(
                    self._filas_escenario_fallido(
                        base=base,
                        votacion_retirada=votacion_retirada,
                        error=str(exc),
                    )
                )
                continue

            filas.extend(
                self._comparar_individual(
                    base=base,
                    alternativo=alternativo,
                    votacion_retirada=votacion_retirada,
                    metrica="d1",
                )
            )
            filas.extend(
                self._comparar_individual(
                    base=base,
                    alternativo=alternativo,
                    votacion_retirada=votacion_retirada,
                    metrica="d2",
                )
            )
            filas.extend(
                self._comparar_partidos(
                    base=base,
                    alternativo=alternativo,
                    votacion_retirada=votacion_retirada,
                )
            )

        resultado = pd.DataFrame(filas)
        if resultado.empty:
            raise ErrorSensibilidad("No se generaron escenarios de sensibilidad.")

        columnas_orden = [
            "familia",
            "metodo",
            "unidad",
            "id",
            "escenario",
            "votacion_retirada",
            "umbral_cobertura",
            "valor_base",
            "valor_alternativo",
            "diferencia",
            "diferencia_abs",
            "cambio_signo",
            "cambio_orden",
            "desplazamiento_orden",
            "n_efectivo_base",
            "n_efectivo_alternativo",
            "cobertura_observada_base",
            "cobertura_observada_alternativa",
            "publicable_base",
            "publicable_alternativo",
            "cambio_publicabilidad",
            "razon_NA_base",
            "razon_NA_alternativa",
            "orientacion_escenario",
            "orientacion_alternativa_evaluada",
            "estado",
        ]
        return resultado.reindex(columns=columnas_orden)

    def _ejecutar_escenario(
        self,
        matriz: pd.DataFrame,
        afiliacion: pd.DataFrame,
    ) -> ResultadoEscenario:
        bcall = ModeloBCall().calcular_auto(
            matriz,
            pivot=self.pivot,
            distance_method=self.metodo_distancia,
            threshold=self.threshold_bcall,
        )

        partidos = PosicionPartido(
            min_decisiones_partido_votacion=self.min_decisiones_partido_votacion,
            min_votaciones_partido=self.min_votaciones_partido,
        ).calcular(
            bcall.votos_orientados,
            afiliacion,
            bcall.diputados,
        )

        return ResultadoEscenario(
            bcall=bcall,
            partidos=partidos,
            n_columnas_entrada=int(matriz.shape[1]),
            n_votaciones_utilizables=int(bcall.votos_orientados.shape[1]),
        )

    def _comparar_individual(
        self,
        *,
        base: ResultadoEscenario,
        alternativo: ResultadoEscenario,
        votacion_retirada: Hashable,
        metrica: str,
    ) -> list[dict[str, Any]]:
        if metrica not in {"d1", "d2"}:
            raise ValueError("La métrica individual debe ser d1 o d2.")

        base_df = base.bcall.diputados
        alt_df = alternativo.bcall.diputados
        ids = base_df.index.union(alt_df.index)

        valores_base = base_df[metrica].reindex(ids)
        valores_alt = alt_df[metrica].reindex(ids)
        orden_base = _orden_relativo(valores_base)
        orden_alt = _orden_relativo(valores_alt)

        filas: list[dict[str, Any]] = []

        for diputado_id in ids:
            valor_base = valores_base.get(diputado_id, np.nan)
            valor_alt = valores_alt.get(diputado_id, np.nan)

            n_base = _obtener_n_efectivo_individual(base_df, diputado_id)
            n_alt = _obtener_n_efectivo_individual(alt_df, diputado_id)

            cobertura_base = _proporcion(
                n_base,
                base.n_votaciones_utilizables,
            )
            cobertura_alt = _proporcion(
                n_alt,
                alternativo.n_votaciones_utilizables,
            )

            razon_base = _razon_individual(base_df, diputado_id, metrica)
            razon_alt = _razon_individual(alt_df, diputado_id, metrica)

            for umbral in self.umbrales_cobertura:
                publicable_base = _publicable(
                    valor_base,
                    cobertura_base,
                    umbral,
                )
                publicable_alt = _publicable(
                    valor_alt,
                    cobertura_alt,
                    umbral,
                )

                filas.append(
                    self._fila_comparacion(
                        familia="posicion_individual",
                        metodo=f"bcall_{metrica}",
                        unidad="diputado",
                        identificador=diputado_id,
                        votacion_retirada=votacion_retirada,
                        umbral=umbral,
                        valor_base=valor_base,
                        valor_alt=valor_alt,
                        orden_base=orden_base.get(diputado_id, np.nan),
                        orden_alt=orden_alt.get(diputado_id, np.nan),
                        n_base=n_base,
                        n_alt=n_alt,
                        cobertura_base=cobertura_base,
                        cobertura_alt=cobertura_alt,
                        publicable_base=publicable_base,
                        publicable_alt=publicable_alt,
                        razon_base=razon_base,
                        razon_alt=razon_alt,
                    )
                )

        return filas

    def _comparar_partidos(
        self,
        *,
        base: ResultadoEscenario,
        alternativo: ResultadoEscenario,
        votacion_retirada: Hashable,
    ) -> list[dict[str, Any]]:
        base_df = base.partidos.resumen.set_index("partido_id")
        alt_df = alternativo.partidos.resumen.set_index("partido_id")
        ids = base_df.index.union(alt_df.index)

        valores_base = base_df["P_p"].reindex(ids)
        valores_alt = alt_df["P_p"].reindex(ids)
        orden_base = _orden_relativo(valores_base)
        orden_alt = _orden_relativo(valores_alt)

        filas: list[dict[str, Any]] = []

        for partido_id in ids:
            valor_base = valores_base.get(partido_id, np.nan)
            valor_alt = valores_alt.get(partido_id, np.nan)

            n_base = _obtener_entero(base_df, partido_id, "n_votaciones")
            n_alt = _obtener_entero(alt_df, partido_id, "n_votaciones")

            cobertura_base = _proporcion(
                n_base,
                base.n_votaciones_utilizables,
            )
            cobertura_alt = _proporcion(
                n_alt,
                alternativo.n_votaciones_utilizables,
            )

            razon_base = _obtener_texto(base_df, partido_id, "razon_NA")
            razon_alt = _obtener_texto(alt_df, partido_id, "razon_NA")

            for umbral in self.umbrales_cobertura:
                publicable_base = _publicable(
                    valor_base,
                    cobertura_base,
                    umbral,
                )
                publicable_alt = _publicable(
                    valor_alt,
                    cobertura_alt,
                    umbral,
                )

                filas.append(
                    self._fila_comparacion(
                        familia="posicion_partidaria",
                        metodo="perfil_partidario_P_p",
                        unidad="partido",
                        identificador=partido_id,
                        votacion_retirada=votacion_retirada,
                        umbral=umbral,
                        valor_base=valor_base,
                        valor_alt=valor_alt,
                        orden_base=orden_base.get(partido_id, np.nan),
                        orden_alt=orden_alt.get(partido_id, np.nan),
                        n_base=n_base,
                        n_alt=n_alt,
                        cobertura_base=cobertura_base,
                        cobertura_alt=cobertura_alt,
                        publicable_base=publicable_base,
                        publicable_alt=publicable_alt,
                        razon_base=razon_base,
                        razon_alt=razon_alt,
                    )
                )

        return filas

    @staticmethod
    def _fila_comparacion(
        *,
        familia: str,
        metodo: str,
        unidad: str,
        identificador: Any,
        votacion_retirada: Hashable,
        umbral: float,
        valor_base: Any,
        valor_alt: Any,
        orden_base: Any,
        orden_alt: Any,
        n_base: int,
        n_alt: int,
        cobertura_base: float,
        cobertura_alt: float,
        publicable_base: bool,
        publicable_alt: bool,
        razon_base: str | None,
        razon_alt: str | None,
    ) -> dict[str, Any]:
        diferencia = (
            float(valor_alt) - float(valor_base)
            if pd.notna(valor_base) and pd.notna(valor_alt)
            else np.nan
        )

        cambio_signo = (
            bool(np.sign(float(valor_base)) != np.sign(float(valor_alt)))
            if pd.notna(valor_base) and pd.notna(valor_alt)
            else pd.NA
        )

        cambio_orden = (
            bool(float(orden_base) != float(orden_alt))
            if pd.notna(orden_base) and pd.notna(orden_alt)
            else pd.NA
        )
        desplazamiento = (
            abs(float(orden_alt) - float(orden_base))
            if pd.notna(orden_base) and pd.notna(orden_alt)
            else np.nan
        )

        return {
            "familia": familia,
            "metodo": metodo,
            "unidad": unidad,
            "id": identificador,
            "escenario": "retiro_una_votacion",
            "votacion_retirada": votacion_retirada,
            "umbral_cobertura": umbral,
            "valor_base": valor_base,
            "valor_alternativo": valor_alt,
            "diferencia": diferencia,
            "diferencia_abs": abs(diferencia) if pd.notna(diferencia) else np.nan,
            "cambio_signo": cambio_signo,
            "cambio_orden": cambio_orden,
            "desplazamiento_orden": desplazamiento,
            "n_efectivo_base": n_base,
            "n_efectivo_alternativo": n_alt,
            "cobertura_observada_base": cobertura_base,
            "cobertura_observada_alternativa": cobertura_alt,
            "publicable_base": publicable_base,
            "publicable_alternativo": publicable_alt,
            "cambio_publicabilidad": bool(publicable_base != publicable_alt),
            "razon_NA_base": razon_base,
            "razon_NA_alternativa": razon_alt,
            "orientacion_escenario": "recalculada_con_pivote_configurado",
            "orientacion_alternativa_evaluada": False,
            "estado": "DESCRIPTIVO_SIN_PADRON",
        }

    def _filas_escenario_fallido(
        self,
        *,
        base: ResultadoEscenario,
        votacion_retirada: Hashable,
        error: str,
    ) -> list[dict[str, Any]]:
        filas: list[dict[str, Any]] = []

        unidades = [
            (
                "posicion_individual",
                "bcall_d1",
                "diputado",
                base.bcall.diputados.index,
                base.bcall.diputados["d1"],
                base.bcall.diputados["m_i"],
            ),
            (
                "posicion_individual",
                "bcall_d2",
                "diputado",
                base.bcall.diputados.index,
                base.bcall.diputados["d2"],
                base.bcall.diputados["m_i"],
            ),
        ]

        partidos = base.partidos.resumen.set_index("partido_id")
        unidades.append(
            (
                "posicion_partidaria",
                "perfil_partidario_P_p",
                "partido",
                partidos.index,
                partidos["P_p"],
                partidos["n_votaciones"],
            )
        )

        for familia, metodo, unidad, ids, valores, n_base in unidades:
            for identificador in ids:
                for umbral in self.umbrales_cobertura:
                    filas.append(
                        {
                            "familia": familia,
                            "metodo": metodo,
                            "unidad": unidad,
                            "id": identificador,
                            "escenario": "retiro_una_votacion",
                            "votacion_retirada": votacion_retirada,
                            "umbral_cobertura": umbral,
                            "valor_base": valores.get(identificador, np.nan),
                            "valor_alternativo": np.nan,
                            "diferencia": np.nan,
                            "diferencia_abs": np.nan,
                            "cambio_signo": pd.NA,
                            "cambio_orden": pd.NA,
                            "desplazamiento_orden": np.nan,
                            "n_efectivo_base": int(n_base.get(identificador, 0)),
                            "n_efectivo_alternativo": 0,
                            "cobertura_observada_base": np.nan,
                            "cobertura_observada_alternativa": np.nan,
                            "publicable_base": False,
                            "publicable_alternativo": False,
                            "cambio_publicabilidad": False,
                            "razon_NA_base": None,
                            "razon_NA_alternativa": f"ESCENARIO_NO_ESTIMABLE: {error}",
                            "orientacion_escenario": "no_estimable",
                            "orientacion_alternativa_evaluada": False,
                            "estado": "ESCENARIO_NO_ESTIMABLE",
                        }
                    )
        return filas

    @staticmethod
    def _validar_matriz(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ErrorSensibilidad("matriz_ternaria debe ser un DataFrame no vacío.")
        if df.index.has_duplicates or df.index.hasnans:
            raise ErrorSensibilidad("Los diputado_id deben ser únicos y no nulos.")
        if df.columns.has_duplicates or df.columns.hasnans:
            raise ErrorSensibilidad("Los votacion_id deben ser únicos y no nulos.")
        if df.shape[1] < 2:
            raise ErrorSensibilidad(
                "Se requieren al menos dos votaciones para leave-one-vote-out."
            )
        numerico = df.apply(pd.to_numeric, errors="coerce")
        invalidos = df.notna() & numerico.isna()
        if invalidos.any(axis=None):
            raise ErrorSensibilidad("La matriz ternaria contiene valores no numéricos.")
        if not (numerico.isna() | numerico.isin([-1, 0, 1])).all().all():
            raise ErrorSensibilidad("La matriz solo admite -1, 0, 1 y NA.")
        return numerico.astype(float).copy()

    @staticmethod
    def _validar_afiliacion(df: pd.DataFrame) -> None:
        requeridas = {
            "diputado_id",
            "votacion_id",
            "fecha",
            "partido_id",
            "partido_nombre",
            "partido_alias",
        }
        if not isinstance(df, pd.DataFrame):
            raise TypeError("afiliacion debe ser un pandas.DataFrame.")
        faltantes = sorted(requeridas - set(df.columns))
        if faltantes:
            raise ErrorSensibilidad(
                f"Faltan columnas requeridas de afiliación: {faltantes}."
            )
        if df.duplicated(["diputado_id", "votacion_id"]).any():
            raise ErrorSensibilidad(
                "La afiliación debe ser única por diputado × votación."
            )


class SensibilidadCohesion:
    """Sensibilidad de la cohesión partidaria resumida (mediana por partido).

    Escenarios:
    - retiro_una_votacion: se retira una votación por vez.
    - sin_votaciones_unanimes: se retiran juntas las votaciones unánimes en la sala.
    - minimo_decisiones_<m>: mínimo alternativo de decisiones por partido × votación.
    - indice_alternativo: Rice y entropía frente al Agreement Index, con los mismos datos.

    Los índices por partido × votación no dependen de otras votaciones, por lo que
    retirar una votación equivale a recalcular el resumen sin sus filas. Las columnas
    son las mismas de la sensibilidad de posición, con familia "cohesion_partidaria".
    """

    INDICES = {
        "mediana_agreement_index": ("mediana_ai", "n_validas_ai"),
        "mediana_rice": ("mediana_rice", "n_validas_rice"),
        "mediana_cohesion_entropica": ("mediana_entropia", "n_validas_entropia"),
    }
    CONTRASTES = {
        "contraste_rice_vs_ai": "mediana_rice",
        "contraste_entropia_vs_ai": "mediana_cohesion_entropica",
    }

    def __init__(
        self,
        parametros: ParametrosCohesion | None = None,
        *,
        umbrales_cobertura: tuple[float, ...] = (0.40, 0.60, 0.80),
        minimos_alternativos: tuple[int, ...] = (3,),
    ) -> None:
        self.p = parametros or ParametrosCohesion()
        self.umbrales_cobertura = tuple(float(x) for x in umbrales_cobertura)
        self.minimos_alternativos = tuple(int(m) for m in minimos_alternativos)

    def calcular(self, por_votacion: pd.DataFrame) -> pd.DataFrame:
        """Compara el resumen base con cada escenario alternativo."""
        votacion = por_votacion.assign(votacion_id=por_votacion["votacion_id"].map(_clave_id))
        base = self._resumen(votacion)
        filas: list[dict[str, Any]] = []

        orden = (votacion[["votacion_id", "fecha"]].drop_duplicates()
                 .sort_values(["fecha", "votacion_id"])["votacion_id"])
        for retirada in orden:
            alt = self._resumen(votacion[votacion["votacion_id"] != retirada])
            filas += self._comparar(base, alt, "retiro_una_votacion", retirada)

        unanimes = sorted(votacion.loc[votacion["votacion_unanime_sala"].astype(bool),
                                       "votacion_id"].unique())
        if unanimes:
            alt = self._resumen(votacion[~votacion["votacion_id"].isin(unanimes)])
            filas += self._comparar(base, alt, "sin_votaciones_unanimes", "|".join(unanimes))

        for minimo in self.minimos_alternativos:
            parametros = ParametrosCohesion(
                min_decisiones_publicacion=minimo,
                min_binarios_publicacion_rice=max(minimo, self.p.min_binarios_publicacion_rice),
                categorias_entropia=self.p.categorias_entropia,
                min_votaciones_resumen=self.p.min_votaciones_resumen,
            )
            columnas = ["partido_id", "partido_alias", "votacion_id", "fecha",
                        "Y", "N", "A", "T"]
            recalculada = IndicesCohesion(parametros).calcular(votacion[columnas])
            alt = self._resumen(recalculada, parametros)
            filas += self._comparar(base, alt, f"minimo_decisiones_{minimo}", pd.NA)

        filas += self._contrastar_indices(base)
        return pd.DataFrame(filas)

    def _resumen(
        self, por_votacion: pd.DataFrame, parametros: ParametrosCohesion | None = None
    ) -> pd.DataFrame:
        resumen = ResumenCohesion(parametros or self.p).calcular(por_votacion)
        resumen["n_votaciones_escenario"] = por_votacion["votacion_id"].nunique()
        return resumen.set_index("partido_id")

    def _comparar(
        self, base: pd.DataFrame, alt: pd.DataFrame, escenario: str, retirada: Any
    ) -> list[dict[str, Any]]:
        filas = []
        for metodo, (columna, validas) in self.INDICES.items():
            filas += self._filas(base, alt, metodo, columna, columna, validas, validas,
                                 escenario, retirada)
        return filas

    def _contrastar_indices(self, base: pd.DataFrame) -> list[dict[str, Any]]:
        filas = []
        ai, validas_ai = self.INDICES["mediana_agreement_index"]
        for metodo, alternativo in self.CONTRASTES.items():
            columna, validas = self.INDICES[alternativo]
            filas += self._filas(base, base, metodo, ai, columna, validas_ai, validas,
                                 "indice_alternativo", pd.NA)
        return filas

    def _filas(
        self, base: pd.DataFrame, alt: pd.DataFrame, metodo: str, col_base: str,
        col_alt: str, validas_base: str, validas_alt: str, escenario: str, retirada: Any,
    ) -> list[dict[str, Any]]:
        ids = base.index.union(alt.index)
        valores_base = base[col_base].reindex(ids)
        valores_alt = alt[col_alt].reindex(ids)
        orden_base, orden_alt = _orden_relativo(valores_base), _orden_relativo(valores_alt)
        filas = []
        for partido_id in ids:
            n_base = _obtener_entero(base, partido_id, validas_base)
            n_alt = _obtener_entero(alt, partido_id, validas_alt)
            cobertura_base = _proporcion(n_base, int(base["n_votaciones_escenario"].iloc[0]))
            cobertura_alt = _proporcion(n_alt, int(alt["n_votaciones_escenario"].iloc[0]))
            publicable_resumen_base = _publicable_resumen(base, partido_id)
            publicable_resumen_alt = _publicable_resumen(alt, partido_id)
            for umbral in self.umbrales_cobertura:
                fila = AnalisisRobustez._fila_comparacion(
                    familia="cohesion_partidaria",
                    metodo=metodo,
                    unidad="partido",
                    identificador=partido_id,
                    votacion_retirada=retirada,
                    umbral=umbral,
                    valor_base=valores_base.get(partido_id, np.nan),
                    valor_alt=valores_alt.get(partido_id, np.nan),
                    orden_base=orden_base.get(partido_id, np.nan),
                    orden_alt=orden_alt.get(partido_id, np.nan),
                    n_base=n_base,
                    n_alt=n_alt,
                    cobertura_base=cobertura_base,
                    cobertura_alt=cobertura_alt,
                    publicable_base=publicable_resumen_base and _publicable(
                        valores_base.get(partido_id, np.nan), cobertura_base, umbral),
                    publicable_alt=publicable_resumen_alt and _publicable(
                        valores_alt.get(partido_id, np.nan), cobertura_alt, umbral),
                    razon_base=_obtener_texto(base, partido_id, "razon_no_publicable_resumen"),
                    razon_alt=_obtener_texto(alt, partido_id, "razon_no_publicable_resumen"),
                )
                fila.update(escenario=escenario, orientacion_escenario="no_aplica",
                            orientacion_alternativa_evaluada=pd.NA)
                filas.append(fila)
        return filas


def _publicable_resumen(df: pd.DataFrame, identificador: Any) -> bool:
    if identificador not in df.index:
        return False
    return bool(df.at[identificador, "publicable_resumen"])


DECIMALES_ORDEN = 12


def _orden_relativo(valores: pd.Series) -> pd.Series:
    """Orden interno usado solo para medir si cambia el orden entre escenarios.

    Se redondea antes de ordenar: con muchos empates exactos, diferencias de punto
    flotante (~1e-16) entre escenarios rompían empates y marcaban cambios de orden
    inexistentes.
    """
    return valores.round(DECIMALES_ORDEN).rank(
        method="average", ascending=True, na_option="keep"
    )


def _clave_id(valor: Any) -> str:
    if pd.isna(valor):
        return "<NA>"
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor)


def _obtener_n_efectivo_individual(
    df: pd.DataFrame,
    identificador: Any,
) -> int:
    if identificador not in df.index or "m_i" not in df.columns:
        return 0
    valor = df.at[identificador, "m_i"]
    return 0 if pd.isna(valor) else int(valor)


def _obtener_entero(
    df: pd.DataFrame,
    identificador: Any,
    columna: str,
) -> int:
    if identificador not in df.index or columna not in df.columns:
        return 0
    valor = df.at[identificador, columna]
    return 0 if pd.isna(valor) else int(float(valor))


def _obtener_texto(
    df: pd.DataFrame,
    identificador: Any,
    columna: str,
) -> str | None:
    if identificador not in df.index or columna not in df.columns:
        return "UNIDAD_NO_PRESENTE_EN_ESCENARIO"
    valor = df.at[identificador, columna]
    return None if pd.isna(valor) else str(valor)


def _razon_individual(
    df: pd.DataFrame,
    identificador: Any,
    metrica: str,
) -> str | None:
    columna = "razon_NA_d1" if metrica == "d1" else "razon_NA_d2"
    return _obtener_texto(df, identificador, columna)


def _proporcion(numerador: int, denominador: int) -> float:
    if denominador <= 0:
        return np.nan
    return float(numerador) / float(denominador)


def _publicable(
    valor: Any,
    cobertura: float,
    umbral: float,
) -> bool:
    return bool(
        pd.notna(valor)
        and pd.notna(cobertura)
        and float(cobertura) >= float(umbral)
    )


def _cargar_matriz(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la matriz ternaria: {ruta}")
    tabla = pd.read_csv(ruta)
    if "diputado_id" not in tabla.columns:
        raise ErrorSensibilidad("matriz_ternaria.csv no contiene diputado_id.")
    tabla = tabla.set_index("diputado_id")
    tabla.columns = [
        int(c) if str(c).isdigit() else c
        for c in tabla.columns
    ]
    return tabla


def _cargar_config(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la configuración: {ruta}")
    with ruta.open("rb") as archivo:
        return tomllib.load(archivo)


def ejecutar(
    *,
    ruta_matriz: str | Path = RUTA_MATRIZ_POR_DEFECTO,
    ruta_afiliacion: str | Path = RUTA_AFILIACION_POR_DEFECTO,
    ruta_config: str | Path = RUTA_CONFIG_POR_DEFECTO,
    ruta_salida: str | Path = RUTA_SALIDA_POR_DEFECTO,
    raiz: Path | None = None,
) -> pd.DataFrame:
    """Ejecuta la sensibilidad individual y partidaria y exporta el CSV."""
    raiz = (raiz or Path(__file__).resolve().parents[3]).resolve()

    def resolver(ruta: str | Path) -> Path:
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    matriz = _cargar_matriz(resolver(ruta_matriz))

    ruta_afi = resolver(ruta_afiliacion)
    if not ruta_afi.exists():
        raise FileNotFoundError(f"No existe la afiliación histórica: {ruta_afi}")
    afiliacion = pd.read_csv(ruta_afi)

    config = _cargar_config(resolver(ruta_config))
    cfg_bcall = config["bcall"]
    cfg_orientacion = config["orientacion"]
    cfg_pos = config["posicion_partidos"]
    cfg_cobertura = config["cobertura"]

    umbrales = tuple(
        float(x)
        for x in cfg_cobertura.get(
            "umbrales_sensibilidad",
            [0.40, 0.60, 0.80],
        )
    )

    analisis = AnalisisRobustez(
        pivot=cfg_orientacion["pivot"],
        metodo_distancia=cfg_bcall["metodo_distancia"],
        threshold_bcall=cfg_bcall["threshold"],
        min_decisiones_partido_votacion=cfg_pos[
            "min_decisiones_por_partido_votacion"
        ],
        min_votaciones_partido=cfg_pos["min_votaciones_publicacion"],
        umbrales_cobertura=umbrales,
    )

    resultado = analisis.calcular(matriz, afiliacion)

    cohesion = SensibilidadCohesion(
        ParametrosCohesion.desde_toml(resolver(ruta_config)),
        umbrales_cobertura=umbrales,
    ).calcular(cohesion_desde_repositorio(raiz))
    resultado = pd.concat(
        [resultado, cohesion.reindex(columns=resultado.columns)], ignore_index=True
    )

    salida = resolver(ruta_salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False, lineterminator="\n")

    return resultado

@dataclass(frozen=True, slots=True)
class ResultadoSensibilidadClustering:
    resumen: pd.DataFrame
    detalle: pd.DataFrame


def _comparar_clusters(base: pd.Series, alt: pd.Series):
    """ARI y alineación por máximo solapamiento sobre diputados comunes."""
    from scipy.optimize import linear_sum_assignment

    if not base.index.equals(alt.index) or len(base) < 2:
        raise ValueError(
            "La comparación requiere al menos dos IDs comunes alineados."
        )

    tabla = pd.crosstab(base, alt)
    c = tabla.to_numpy(dtype=float)

    def pares(x):
        return float(np.sum(x * (x - 1) / 2))

    total = len(base) * (len(base) - 1) / 2
    esperados = (
        pares(c.sum(axis=1)) * pares(c.sum(axis=0)) / total
    )
    maximo = (
        pares(c.sum(axis=1)) + pares(c.sum(axis=0))
    ) / 2

    ari = (
        1.0
        if np.isclose(maximo, esperados)
        else (pares(c) - esperados) / (maximo - esperados)
    )

    # Las etiquetas numéricas pueden intercambiarse entre ejecuciones.
    filas, columnas = linear_sum_assignment(-c)
    mapa: dict[Any, Any] = {
        tabla.columns[j]: tabla.index[i]
        for i, j in zip(filas, columnas)
    }

    # Puede haber grupos sin correspondencia en el universo común.
    for etiqueta in tabla.columns:
        if etiqueta not in mapa:
            mapa[etiqueta] = f"SIN_CORRESPONDENCIA:{etiqueta}"

    # Un empate entre alineaciones afecta la interpretación individual,
    # pero no afecta el ARI, que es independiente de las etiquetas.
    optimo = c[filas, columnas].sum()
    ambiguo = False

    for i, j in zip(filas, columnas):
        costo = -c.copy()
        costo[i, j] = c.sum() + 1
        f, co = linear_sum_assignment(costo)

        if costo[f, co].sum() == -optimo:
            ambiguo = True
            break

    return ari, mapa, ambiguo


class SensibilidadClustering:
    """Retiro de votos, bloques y filtros del clustering nominal.

    Cada escenario recalcula selección de columnas, cobertura individual,
    cobertura por pares y agrupamiento.

    El ARI se calcula exclusivamente sobre diputados comunes.
    Entradas y salidas del universo se registran por separado.
    """

    def __init__(
        self,
        *,
        cobertura_base=0.80,
        umbrales_cobertura=(0.40, 0.60, 0.80, 1.00),
        n_clusters=2,
    ):
        from F4.src.analisis.agrupamiento import AgrupamientoDiputados

        AgrupamientoDiputados(
            cobertura_minima=cobertura_base,
            n_clusters=n_clusters,
        )

        if not umbrales_cobertura:
            raise ValueError(
                "Debe existir al menos un umbral alternativo."
            )

        for umbral in umbrales_cobertura:
            AgrupamientoDiputados(
                cobertura_minima=umbral,
                n_clusters=n_clusters,
            )

        self.cobertura_base = float(cobertura_base)
        self.umbrales = tuple(
            dict.fromkeys(float(u) for u in umbrales_cobertura)
        )
        self.n_clusters = n_clusters

    def calcular(self, matriz_nominal, *, bloques=None):
        from F4.src.analisis.agrupamiento import (
            AgrupamientoDiputados,
            ErrorAgrupamiento,
        )

        def ejecutar(matriz, umbral):
            return AgrupamientoDiputados(
                cobertura_minima=umbral,
                n_clusters=self.n_clusters,
            ).calcular(matriz)

        base = ejecutar(matriz_nominal, self.cobertura_base)
        b = base.diputados.set_index("diputado_id")["cluster"]

        escenarios = [("base", (), self.cobertura_base)]

        escenarios += [
            (
                f"retiro_votacion:{v}",
                (v,),
                self.cobertura_base,
            )
            for v in matriz_nominal.columns
        ]

        # Normalizar IDs permite recibir bloques con enteros o strings.
        columnas = {
            _clave_id(v): v for v in matriz_nominal.columns
        }

        if len(columnas) != len(matriz_nominal.columns):
            raise ErrorSensibilidad(
                "Hay identificadores de votación equivalentes."
            )

        for nombre, ids in (bloques or {}).items():
            ids = tuple(
                dict.fromkeys(_clave_id(v) for v in ids)
            )

            if (
                not nombre
                or not ids
                or any(v not in columnas for v in ids)
            ):
                raise ErrorSensibilidad(
                    f"Bloque inválido: {nombre!r}."
                )

            escenarios.append(
                (
                    f"retiro_bloque:{nombre}",
                    tuple(columnas[v] for v in ids),
                    self.cobertura_base,
                )
            )

        escenarios += [
            (f"cobertura:{u:g}", (), u)
            for u in self.umbrales
            if u != self.cobertura_base
        ]

        resumen, detalle = [], []

        for escenario, retiradas, umbral in escenarios:
            fila = {
                "escenario": escenario,
                "votaciones_retiradas": "|".join(
                    map(str, retiradas)
                ),
                "cobertura_base": self.cobertura_base,
                "cobertura_alternativa": umbral,
                "n_clusters": self.n_clusters,
                "n_diputados_base": len(b),
                "n_votaciones_base": len(
                    base.votaciones_incluidas
                ),
                "min_covotos_base": base.min_covotos,
                "n_diputados_alternativo": pd.NA,
                "n_excluidos_alternativo": pd.NA,
                "n_votaciones_alternativo": pd.NA,
                "min_covotos_alternativo": pd.NA,
                "n_comunes": pd.NA,
                "n_entran": pd.NA,
                "n_salen": pd.NA,
                "ari": np.nan,
                "n_cambios_grupo": pd.NA,
                "alineacion_ambigua": pd.NA,
                "estado": "ESCENARIO_NO_ESTIMABLE",
                "razon_NA": None,
            }

            try:
                alt = (
                    base
                    if escenario == "base"
                    else ejecutar(
                        matriz_nominal.drop(
                            columns=list(retiradas)
                        ),
                        umbral,
                    )
                )
            except ErrorAgrupamiento as exc:
                # Un fallo de estimación no representa un ARI de cero,
                # ni significa que todos los diputados fueron excluidos.
                fila["razon_NA"] = str(exc)
                resumen.append(fila)
                continue

            a = alt.diputados.set_index("diputado_id")["cluster"]
            comunes = b.index.intersection(a.index, sort=False)

            ari, mapa, ambiguo = np.nan, {}, pd.NA

            if len(comunes) >= 2:
                ari, mapa, ambiguo = _comparar_clusters(
                    b.loc[comunes],
                    a.loc[comunes],
                )

            alineadas = a.map(mapa)
            comparables = comunes[
                alineadas.reindex(comunes).notna()
            ]
            cambios = b.loc[comparables].ne(
                alineadas.loc[comparables]
            )

            fila.update(
                n_diputados_alternativo=len(a),
                n_excluidos_alternativo=len(alt.exclusiones),
                n_votaciones_alternativo=len(
                    alt.votaciones_incluidas
                ),
                min_covotos_alternativo=alt.min_covotos,
                n_comunes=len(comunes),
                n_entran=len(a.index.difference(b.index)),
                n_salen=len(b.index.difference(a.index)),
                ari=ari,
                n_cambios_grupo=(
                    int(cambios.sum())
                    if len(comparables)
                    else pd.NA
                ),
                alineacion_ambigua=ambiguo,
                estado="DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS",
                razon_NA=(
                    "MENOS_DE_DOS_DIPUTADOS_COMUNES"
                    if len(comunes) < 2
                    else None
                ),
            )
            resumen.append(fila)

            for diputado in b.index.union(a.index, sort=False):
                en_base = diputado in b.index
                en_alt = diputado in a.index

                detalle.append(
                    {
                        "escenario": escenario,
                        "diputado_id": diputado,
                        "incluido_base": en_base,
                        "incluido_alternativo": en_alt,
                        "cluster_base": b.get(
                            diputado, pd.NA
                        ),
                        "cluster_alternativo": a.get(
                            diputado, pd.NA
                        ),
                        "cluster_alternativo_alineado": (
                            alineadas.get(diputado, pd.NA)
                        ),
                        "cambio_grupo": cambios.get(
                            diputado, pd.NA
                        ),
                        "alineacion_ambigua": ambiguo,
                        "estado_universo": (
                            "COMUN"
                            if en_base and en_alt
                            else ("SALE" if en_base else "ENTRA")
                        ),
                    }
                )

        return ResultadoSensibilidadClustering(
            resumen=pd.DataFrame(resumen),
            detalle=pd.DataFrame(detalle),
        )


def ejecutar_clustering(
    *,
    ruta_matriz="F4/data/processed/matriz_nominal.csv",
    directorio_salida="F4/data/reports",
    cobertura_base=0.80,
    umbrales_cobertura=(0.40, 0.60, 0.80, 1.00),
    n_clusters=2,
    bloques=None,
    raiz=None,
):
    from F4.src.analisis.afinidad import cargar_matriz_nominal

    raiz = (
        Path(raiz)
        if raiz is not None
        else Path(__file__).resolve().parents[3]
    ).resolve()

    def resolver(ruta):
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    resultado = SensibilidadClustering(
        cobertura_base=cobertura_base,
        umbrales_cobertura=umbrales_cobertura,
        n_clusters=n_clusters,
    ).calcular(
        cargar_matriz_nominal(resolver(ruta_matriz)),
        bloques=bloques,
    )

    salida = resolver(directorio_salida)
    salida.mkdir(parents=True, exist_ok=True)

    resultado.resumen.to_csv(
        salida / "sensibilidad_clustering.csv",
        index=False,
        lineterminator="\n",
    )
    resultado.detalle.to_csv(
        salida / "sensibilidad_clustering_diputados.csv",
        index=False,
        lineterminator="\n",
    )

    return resultado


def _main_clustering(args):
    try:
        bloques = {}

        for especificacion in args.bloque_clustering:
            nombre, separador, ids = especificacion.partition("=")

            if (
                not separador
                or not nombre.strip()
                or nombre.strip() in bloques
            ):
                raise ErrorSensibilidad(
                    "Use bloques únicos con formato nombre=id1,id2."
                )

            bloques[nombre.strip()] = [
                v.strip() for v in ids.split(",")
            ]

        resultado = ejecutar_clustering(
            ruta_matriz=args.matriz_clustering,
            directorio_salida=args.salida_clustering,
            cobertura_base=args.cobertura_clustering,
            n_clusters=args.clusters,
            bloques=bloques,
        )

    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    columnas = [
        "escenario",
        "n_diputados_alternativo",
        "n_comunes",
        "ari",
        "n_cambios_grupo",
    ]
    print(resultado.resumen[columnas].to_string(index=False))

    fallidos = resultado.resumen["estado"].eq(
        "ESCENARIO_NO_ESTIMABLE"
    ).sum()

    print(
        f"Escenarios no estimables: {fallidos}; "
        "revisar razon_NA en el CSV."
    )

    return 0


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalcula posición individual y partidaria retirando "
            "una votación por vez."
        )
    )
    parser.add_argument("--matriz", default=str(RUTA_MATRIZ_POR_DEFECTO))
    parser.add_argument("--afiliacion", default=str(RUTA_AFILIACION_POR_DEFECTO))
    parser.add_argument("--config", default=str(RUTA_CONFIG_POR_DEFECTO))
    parser.add_argument("--salida", default=str(RUTA_SALIDA_POR_DEFECTO))
    
    parser.add_argument(
        "--solo-clustering",
        action="store_true",
        help="Ejecutar la sensibilidad del clustering nominal.",
    )
    parser.add_argument(
        "--matriz-clustering",
        default="F4/data/processed/matriz_nominal.csv",
    )
    parser.add_argument(
        "--salida-clustering",
        default="F4/data/reports",
    )
    parser.add_argument(
        "--cobertura-clustering",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--bloque-clustering",
        action="append",
        default=[],
        help="Bloque a retirar: nombre=id1,id2. Puede repetirse.",
    )
    
    return parser


def main() -> int:
    args = _crear_parser().parse_args()

    if args.solo_clustering:
        return _main_clustering(args)

    try:
        resultado = ejecutar(
            ruta_matriz=args.matriz,
            ruta_afiliacion=args.afiliacion,
            ruta_config=args.config,
            ruta_salida=args.salida,
        )
    except (ErrorSensibilidad, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    posicion = resultado[resultado["familia"] != "cohesion_partidaria"]
    escenarios = posicion["votacion_retirada"].nunique()
    print(
        "Sensibilidad calculada: "
        f"{escenarios} escenarios leave-one-vote-out, "
        f"{len(posicion)} comparaciones."
    )
    cohesion = resultado[resultado["familia"] == "cohesion_partidaria"]
    print(
        "Cohesión: "
        f"{cohesion['escenario'].nunique()} tipos de escenario, "
        f"{len(cohesion)} comparaciones."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
