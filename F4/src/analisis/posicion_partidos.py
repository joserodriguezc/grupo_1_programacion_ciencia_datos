from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from F4.src.analisis.bcall import ModeloBCall


RUTA_MATRIZ_POR_DEFECTO = Path("F4/data/processed/matriz_ternaria.csv")
RUTA_AFILIACION_POR_DEFECTO = Path("F4/data/processed/afiliacion_por_votacion.csv")
RUTA_CONFIG_POR_DEFECTO = Path("F4/config/analisis.toml")
RUTA_SALIDA_POR_DEFECTO = Path(
    "F4/data/results/partidos/posicion_partidaria.csv"
)

COLUMNAS_AFILIACION = (
    "diputado_id",
    "votacion_id",
    "fecha",
    "partido_id",
    "partido_nombre",
    "partido_alias",
)


class ErrorPosicionPartido(ValueError):
    """La entrada no permite estimar posiciones partidarias de forma válida."""


@dataclass(frozen=True, slots=True)
class ResultadoPosicionPartidos:
    """Resultados por partido × votación y resumen por partido."""

    por_votacion: pd.DataFrame
    resumen: pd.DataFrame

    def combinado(self) -> pd.DataFrame:
        """Devuelve una tabla única lista para exportación."""
        pv = self.por_votacion.copy()
        pv.insert(0, "nivel", "partido_votacion")

        rs = self.resumen.copy()
        rs.insert(0, "nivel", "partido")

        columnas = list(dict.fromkeys([*pv.columns, *rs.columns]))
        return pd.concat(
            [pv.reindex(columns=columnas), rs.reindex(columns=columnas)],
            ignore_index=True,
        )


class PosicionPartido:
    """Agrega votos orientados por partido respetando afiliación histórica."""

    def __init__(
        self,
        *,
        min_decisiones_partido_votacion: int = 2,
        min_votaciones_partido: int = 2,
    ) -> None:
        if (
            isinstance(min_decisiones_partido_votacion, bool)
            or not isinstance(min_decisiones_partido_votacion, int)
            or min_decisiones_partido_votacion < 1
        ):
            raise ValueError(
                "min_decisiones_partido_votacion debe ser un entero positivo."
            )

        if (
            isinstance(min_votaciones_partido, bool)
            or not isinstance(min_votaciones_partido, int)
            or min_votaciones_partido < 1
        ):
            raise ValueError(
                "min_votaciones_partido debe ser un entero positivo."
            )

        self.min_decisiones_partido_votacion = min_decisiones_partido_votacion
        self.min_votaciones_partido = min_votaciones_partido

    def calcular(
        self,
        votos_orientados: pd.DataFrame,
        afiliacion: pd.DataFrame,
        diputados_bcall: pd.DataFrame,
    ) -> ResultadoPosicionPartidos:
        """Calcula posición por votación y perfil global de cada partido.

        ``votos_orientados`` debe contener la salida ``u_ij`` de B-Call:
        filas = diputados, columnas = votaciones y valores reales o NA.

        ``diputados_bcall`` debe contener al menos ``d1`` por diputado.
        """
        u = self._validar_votos_orientados(votos_orientados)
        afiliacion = self._validar_afiliacion(afiliacion)
        diputados = self._validar_diputados_bcall(diputados_bcall)

        largo = (
            u.rename_axis(index="diputado_id", columns="votacion_id")
            .stack(future_stack=True)
            .rename("u_ij")
            .reset_index()
            .dropna(subset=["u_ij"])
            .reset_index(drop=True)
        )

        # Normalizar solo para el cruce; los valores exportados se conservan.
        largo["_dip_key"] = largo["diputado_id"].map(_clave_id)
        largo["_vot_key"] = largo["votacion_id"].map(_clave_id)

        afi = afiliacion.copy()
        afi["_dip_key"] = afi["diputado_id"].map(_clave_id)
        afi["_vot_key"] = afi["votacion_id"].map(_clave_id)

        if afi.duplicated(["_dip_key", "_vot_key"]).any():
            raise ErrorPosicionPartido(
                "afiliacion contiene claves diputado_id × votacion_id duplicadas."
            )

        base = largo.merge(
            afi,
            on=["_dip_key", "_vot_key"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_afiliacion"),
        )

        # Cada u_ij proviene de una decisión observada, por lo que debería existir
        # afiliación histórica correspondiente. La militancia puede ser nula.
        sin_fila_afiliacion = base["votacion_id_afiliacion"].isna()
        if sin_fila_afiliacion.any():
            ejemplos = base.loc[
                sin_fila_afiliacion, ["diputado_id", "votacion_id"]
            ].head(5)
            raise ErrorPosicionPartido(
                "Faltan filas de afiliación para votos orientados. "
                f"Ejemplos: {ejemplos.to_dict('records')}."
            )

        # Usar identificadores originales de la tabla de afiliación.
        base["diputado_id"] = base["diputado_id_afiliacion"]
        base["votacion_id"] = base["votacion_id_afiliacion"]

        sin_partido = base["partido_id"].isna() | base["partido_nombre"].isna()
        base_valida = base.loc[~sin_partido].copy()

        por_votacion = self._calcular_por_votacion(base_valida)
        resumen = self._calcular_resumen(
            por_votacion=por_votacion,
            base=base_valida,
            diputados_bcall=diputados,
        )

        return ResultadoPosicionPartidos(
            por_votacion=por_votacion,
            resumen=resumen,
        )

    def _calcular_por_votacion(self, base: pd.DataFrame) -> pd.DataFrame:
        columnas = [
            "partido_id",
            "partido_nombre",
            "partido_alias",
            "votacion_id",
            "fecha",
            "b_pj",
            "n_decisiones",
            "n_diputados",
            "incluido",
            "razon_exclusion",
        ]

        if base.empty:
            return pd.DataFrame(columns=columnas)

        agrupado = (
            base.groupby(
                [
                    "partido_id",
                    "partido_nombre",
                    "partido_alias",
                    "votacion_id",
                    "fecha",
                ],
                dropna=False,
                sort=False,
            )
            .agg(
                b_pj=("u_ij", "mean"),
                n_decisiones=("u_ij", "count"),
                n_diputados=("diputado_id", "nunique"),
            )
            .reset_index()
        )

        agrupado["incluido"] = (
            agrupado["n_decisiones"] >= self.min_decisiones_partido_votacion
        )
        agrupado["razon_exclusion"] = np.where(
            agrupado["incluido"],
            None,
            "DECISIONES_INSUFICIENTES_PARTIDO_VOTACION",
        )

        # El valor puede calcularse descriptivamente aun cuando no alcance el
        # mínimo; la bandera incluido controla si entra al perfil P_p.
        return agrupado.loc[:, columnas]

    def _calcular_resumen(
        self,
        *,
        por_votacion: pd.DataFrame,
        base: pd.DataFrame,
        diputados_bcall: pd.DataFrame,
    ) -> pd.DataFrame:
        columnas = [
            "partido_id",
            "partido_nombre",
            "partido_alias",
            "P_p",
            "n_votaciones",
            "n_decisiones",
            "n_diputados",
            "mediana_d1",
            "iqr_d1",
            "incluido",
            "razon_NA",
            "estado",
        ]

        partidos = (
            base[["partido_id", "partido_nombre", "partido_alias"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        filas: list[dict[str, Any]] = []

        for partido in partidos.itertuples(index=False):
            detalle = por_votacion[
                por_votacion["partido_id"].eq(partido.partido_id)
            ]
            validas = detalle[detalle["incluido"]]

            p_p = validas["b_pj"].mean() if not validas.empty else np.nan
            n_votaciones = int(validas["votacion_id"].nunique())
            n_decisiones = int(validas["n_decisiones"].sum())

            miembros = (
                base.loc[
                    base["partido_id"].eq(partido.partido_id),
                    "diputado_id",
                ]
                .drop_duplicates()
            )

            d1 = (
                diputados_bcall.reindex(miembros.map(_clave_id))
                .loc[:, "d1"]
                .dropna()
            )

            mediana = float(d1.median()) if not d1.empty else np.nan
            iqr = (
                float(d1.quantile(0.75) - d1.quantile(0.25))
                if not d1.empty
                else np.nan
            )

            incluido = n_votaciones >= self.min_votaciones_partido
            razon = (
                None
                if incluido
                else "VOTACIONES_INSUFICIENTES_PARA_PERFIL_PARTIDARIO"
            )

            filas.append(
                {
                    "partido_id": partido.partido_id,
                    "partido_nombre": partido.partido_nombre,
                    "partido_alias": partido.partido_alias,
                    "P_p": p_p if incluido else np.nan,
                    "n_votaciones": n_votaciones,
                    "n_decisiones": n_decisiones,
                    "n_diputados": int(len(d1.index.unique())),
                    "mediana_d1": mediana,
                    "iqr_d1": iqr,
                    "incluido": incluido,
                    "razon_NA": razon,
                    # Sin padrón verificable, el perfil se conserva como
                    # descriptivo y no se presenta como ranking concluyente.
                    "estado": "DESCRIPTIVO_SIN_PADRON",
                }
            )

        return pd.DataFrame(filas, columns=columnas)

    @staticmethod
    def _validar_votos_orientados(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("votos_orientados debe ser un pandas.DataFrame.")
        if df.empty:
            raise ErrorPosicionPartido("votos_orientados está vacío.")
        if df.index.has_duplicates or df.index.hasnans:
            raise ErrorPosicionPartido(
                "Los diputado_id de votos_orientados deben ser únicos y no nulos."
            )
        if df.columns.has_duplicates or df.columns.hasnans:
            raise ErrorPosicionPartido(
                "Los votacion_id de votos_orientados deben ser únicos y no nulos."
            )

        numerico = df.apply(pd.to_numeric, errors="coerce")
        invalidos = df.notna() & numerico.isna()
        if invalidos.any(axis=None):
            raise ErrorPosicionPartido(
                "votos_orientados solo puede contener valores numéricos o NA."
            )
        return numerico.astype(float).copy()

    @staticmethod
    def _validar_afiliacion(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("afiliacion debe ser un pandas.DataFrame.")

        faltantes = [c for c in COLUMNAS_AFILIACION if c not in df.columns]
        if faltantes:
            raise ErrorPosicionPartido(
                f"Faltan columnas de afiliación: {faltantes}."
            )

        if df[["diputado_id", "votacion_id"]].isna().any(axis=None):
            raise ErrorPosicionPartido(
                "diputado_id y votacion_id no pueden ser nulos en afiliación."
            )

        if df.duplicated(["diputado_id", "votacion_id"]).any():
            raise ErrorPosicionPartido(
                "La afiliación debe tener una fila única por diputado × votación."
            )

        return df.loc[:, COLUMNAS_AFILIACION].copy()

    @staticmethod
    def _validar_diputados_bcall(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("diputados_bcall debe ser un pandas.DataFrame.")
        if "d1" not in df.columns:
            raise ErrorPosicionPartido("diputados_bcall debe contener d1.")
        if df.index.has_duplicates or df.index.hasnans:
            raise ErrorPosicionPartido(
                "diputados_bcall requiere diputado_id únicos y no nulos."
            )

        salida = df.copy()
        salida.index = salida.index.map(_clave_id)
        return salida


def _clave_id(valor: Any) -> str:
    """Clave estable para cruces entre CSV y DataFrames."""
    if pd.isna(valor):
        return "<NA>"
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor)


def _cargar_matriz(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la matriz ternaria: {ruta}")

    tabla = pd.read_csv(ruta)
    if "diputado_id" not in tabla.columns:
        raise ErrorPosicionPartido(
            "La matriz ternaria debe incluir la columna diputado_id."
        )

    tabla = tabla.set_index("diputado_id")
    tabla.columns = [_convertir_id_columna(c) for c in tabla.columns]
    return tabla


def _convertir_id_columna(valor: Any) -> Any:
    texto = str(valor)
    try:
        return int(texto)
    except ValueError:
        return texto


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
) -> ResultadoPosicionPartidos:
    """Ejecuta B-Call y la agregación partidaria desde los productos procesados."""
    raiz = (raiz or Path(__file__).resolve().parents[3]).resolve()

    def resolver(ruta: str | Path) -> Path:
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    matriz = _cargar_matriz(resolver(ruta_matriz))

    ruta_afi = resolver(ruta_afiliacion)
    if not ruta_afi.exists() or ruta_afi.stat().st_size == 0:
        raise FileNotFoundError(
            "No existe una tabla de afiliación utilizable. "
            "Genere primero afiliacion_por_votacion.csv."
        )
    afiliacion = pd.read_csv(ruta_afi)

    config = _cargar_config(resolver(ruta_config))
    cfg_bcall = config["bcall"]
    cfg_orientacion = config["orientacion"]
    cfg_posicion = config["posicion_partidos"]

    resultado_bcall = ModeloBCall().calcular_auto(
        matriz,
        pivot=cfg_orientacion["pivot"],
        distance_method=cfg_bcall["metodo_distancia"],
        threshold=cfg_bcall["threshold"],
    )

    modelo = PosicionPartido(
        min_decisiones_partido_votacion=cfg_posicion[
            "min_decisiones_por_partido_votacion"
        ],
        min_votaciones_partido=cfg_posicion["min_votaciones_publicacion"],
    )
    resultado = modelo.calcular(
        resultado_bcall.votos_orientados,
        afiliacion,
        resultado_bcall.diputados,
    )

    salida = resolver(ruta_salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.combinado().to_csv(salida, index=False)

    return resultado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula la posición relativa de los partidos."
    )
    parser.add_argument("--matriz", default=str(RUTA_MATRIZ_POR_DEFECTO))
    parser.add_argument("--afiliacion", default=str(RUTA_AFILIACION_POR_DEFECTO))
    parser.add_argument("--config", default=str(RUTA_CONFIG_POR_DEFECTO))
    parser.add_argument("--salida", default=str(RUTA_SALIDA_POR_DEFECTO))
    return parser


def main() -> int:
    args = _crear_parser().parse_args()

    try:
        resultado = ejecutar(
            ruta_matriz=args.matriz,
            ruta_afiliacion=args.afiliacion,
            ruta_config=args.config,
            ruta_salida=args.salida,
        )
    except (ErrorPosicionPartido, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Posición partidaria calculada: "
        f"{len(resultado.resumen)} partidos, "
        f"{len(resultado.por_votacion)} combinaciones partido × votación."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
