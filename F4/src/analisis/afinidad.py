from __future__ import annotations

import argparse
import itertools
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUTA_MATRIZ_POR_DEFECTO = Path("F4/data/processed/matriz_nominal.csv")
RUTA_CONFIG_POR_DEFECTO = Path("F4/config/analisis.toml")
RUTA_SALIDA_POR_DEFECTO = Path("F4/data/results/pares/afinidad_diputados.csv")
RUTA_MATRIZ_HAMMING_POR_DEFECTO = Path(
    "F4/data/results/pares/matriz_hamming.csv"
)
RUTA_AFILIACION_POR_DEFECTO = Path("F4/data/processed/afiliacion_por_votacion.csv")
RUTA_HAMMING_PARTIDOS_POR_DEFECTO = Path("F4/data/results/pares/hamming_partidos.csv")

CATEGORIAS_VALIDAS = frozenset({"Sí", "No", "Abstención"})


class ErrorAfinidad(ValueError):
    """La entrada no permite calcular afinidad de forma válida."""


@dataclass(frozen=True, slots=True)
class ResultadoAfinidad:
    """Tabla de pares y matriz simétrica de distancia de Hamming."""

    pares: pd.DataFrame
    matriz_hamming: pd.DataFrame

    @property
    def n_pares(self) -> int:
        return int(len(self.pares))

    @property
    def n_pares_incluidos(self) -> int:
        if self.pares.empty:
            return 0
        return int(self.pares["incluido"].sum())


class AfinidadPares:
    """Calcula afinidad nominal para todos los pares de diputados."""

    def __init__(self, *, min_covotos: int = 2) -> None:
        if isinstance(min_covotos, bool) or not isinstance(min_covotos, int):
            raise TypeError("min_covotos debe ser un entero.")
        if min_covotos < 1:
            raise ValueError("min_covotos debe ser al menos 1.")
        self.min_covotos = min_covotos

    def calcular(self, matriz_nominal: pd.DataFrame) -> ResultadoAfinidad:
        matriz = self._validar_matriz(matriz_nominal)
        diputados = list(matriz.index)
        votaciones = list(matriz.columns)
        n_votaciones = len(votaciones)

        unanimidad = self._detectar_votaciones_unanimes(matriz)
        filas: list[dict[str, Any]] = []

        for diputado_i, diputado_j in itertools.combinations(diputados, 2):
            vi = matriz.loc[diputado_i]
            vj = matriz.loc[diputado_j]

            mascara_covoto = vi.notna() & vj.notna()
            n_covotos = int(mascara_covoto.sum())

            if n_covotos == 0:
                coincidencias = 0
                acuerdo = np.nan
                hamming = np.nan
                prevalencia_si = np.nan
            else:
                vi_cov = vi[mascara_covoto]
                vj_cov = vj[mascara_covoto]
                coincidencias = int((vi_cov == vj_cov).sum())
                acuerdo = coincidencias / n_covotos
                hamming = 1.0 - acuerdo
                prevalencia_si = float(
                    (
                        vi_cov.eq("Sí").sum()
                        + vj_cov.eq("Sí").sum()
                    )
                    / (2 * n_covotos)
                )

            mascara_sin_unanimes = mascara_covoto & ~unanimidad
            n_covotos_sin_unanimes = int(mascara_sin_unanimes.sum())

            if n_covotos_sin_unanimes == 0:
                acuerdo_sin_unanimes = np.nan
                hamming_sin_unanimes = np.nan
            else:
                iguales_sin = int(
                    (
                        vi[mascara_sin_unanimes]
                        == vj[mascara_sin_unanimes]
                    ).sum()
                )
                acuerdo_sin_unanimes = iguales_sin / n_covotos_sin_unanimes
                hamming_sin_unanimes = 1.0 - acuerdo_sin_unanimes

            incluido = n_covotos >= self.min_covotos
            if n_covotos == 0:
                razon = "SIN_COVOTOS"
            elif not incluido:
                razon = "COVOTOS_INSUFICIENTES"
            else:
                razon = None

            filas.append(
                {
                    "diputado_i": diputado_i,
                    "diputado_j": diputado_j,
                    "coincidencias": coincidencias,
                    "n_covotos": n_covotos,
                    # Sin padrón verificable esto es cobertura descriptiva sobre
                    # las columnas del corpus, no tasa de asistencia/elegibilidad.
                    "cobertura_corpus": (
                        n_covotos / n_votaciones if n_votaciones else np.nan
                    ),
                    "acuerdo": acuerdo,
                    "hamming": hamming,
                    "prevalencia_si_covotos": prevalencia_si,
                    "n_covotos_sin_unanimes": n_covotos_sin_unanimes,
                    "acuerdo_sin_unanimes": acuerdo_sin_unanimes,
                    "hamming_sin_unanimes": hamming_sin_unanimes,
                    "incluido": incluido,
                    "razon_exclusion": razon,
                    "estado": "DESCRIPTIVO_SIN_PADRON",
                }
            )

        pares = pd.DataFrame(filas)
        matriz_hamming = self._construir_matriz_hamming(pares, diputados)
        self._validar_resultados(pares, matriz_hamming, diputados)

        return ResultadoAfinidad(
            pares=pares,
            matriz_hamming=matriz_hamming,
        )

    @staticmethod
    def _validar_matriz(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("matriz_nominal debe ser un pandas.DataFrame.")
        if df.empty:
            raise ErrorAfinidad("La matriz nominal está vacía.")
        if df.index.has_duplicates or df.index.hasnans:
            raise ErrorAfinidad(
                "Los diputado_id de la matriz deben ser únicos y no nulos."
            )
        if df.columns.has_duplicates or df.columns.hasnans:
            raise ErrorAfinidad(
                "Los votacion_id de la matriz deben ser únicos y no nulos."
            )

        # Validar únicamente celdas realmente observadas. Los NA son parte
        # válida del contrato y representan ausencia de decisión registrada.
        valores = df.to_numpy(dtype=object).ravel()
        observados = {str(valor) for valor in valores if pd.notna(valor)}
        invalidos = sorted(observados - CATEGORIAS_VALIDAS)
        if invalidos:
            raise ErrorAfinidad(
                "La matriz nominal contiene categorías no reconocidas: "
                f"{invalidos}."
            )

        return df.copy()

    @staticmethod
    def _detectar_votaciones_unanimes(matriz: pd.DataFrame) -> pd.Series:
        """True si todos los votos observados de la votación son iguales."""
        return matriz.apply(
            lambda columna: columna.dropna().nunique() == 1
            and columna.dropna().size > 0,
            axis=0,
        )

    @staticmethod
    def _construir_matriz_hamming(
        pares: pd.DataFrame,
        diputados: list[Any],
    ) -> pd.DataFrame:
        # Construir primero el ndarray evita modificar una vista read-only de
        # pandas/NumPy cuando Copy-on-Write está activo.
        valores = np.full(
            (len(diputados), len(diputados)),
            np.nan,
            dtype=float,
        )
        np.fill_diagonal(valores, 0.0)

        matriz = pd.DataFrame(
            valores,
            index=pd.Index(diputados, name="diputado_id"),
            columns=diputados,
        )

        for fila in pares.itertuples(index=False):
            # Solo pares que cumplen el mínimo de co-votos reciben distancia
            # utilizable en la matriz. Los demás permanecen como NA.
            if fila.incluido and pd.notna(fila.hamming):
                matriz.loc[fila.diputado_i, fila.diputado_j] = fila.hamming
                matriz.loc[fila.diputado_j, fila.diputado_i] = fila.hamming

        return matriz

    @staticmethod
    def _validar_resultados(
        pares: pd.DataFrame,
        matriz_hamming: pd.DataFrame,
        diputados: list[Any],
    ) -> None:
        esperado = len(diputados) * (len(diputados) - 1) // 2
        if len(pares) != esperado:
            raise ErrorAfinidad(
                f"Se esperaban {esperado} pares y se obtuvieron {len(pares)}."
            )

        con_covotos = pares["n_covotos"].gt(0)
        diferencia = (
            pares.loc[con_covotos, "hamming"]
            - (1.0 - pares.loc[con_covotos, "acuerdo"])
        ).abs()
        if (diferencia > 1e-12).any():
            raise ErrorAfinidad("No se cumple hamming = 1 - acuerdo.")

        if not np.allclose(
            np.diag(matriz_hamming.to_numpy(dtype=float)),
            0.0,
            equal_nan=False,
        ):
            raise ErrorAfinidad("La diagonal de Hamming debe ser cero.")

        valores = matriz_hamming.to_numpy(dtype=float)
        if not np.allclose(valores, valores.T, equal_nan=True):
            raise ErrorAfinidad("La matriz de Hamming debe ser simétrica.")


def _clave(valor: Any) -> str:
    """Normaliza IDs leídos como número o texto (20629, '20629', 20629.0)."""
    texto = str(valor)
    return texto[:-2] if texto.endswith(".0") else texto


def comparaciones_entre_partidos(
    matriz_nominal: pd.DataFrame,
    afiliacion: pd.DataFrame,
) -> pd.DataFrame:
    """Comparaciones de co-votos entre partidos, por votación.

    Cada par de diputados con decisión registrada en una votación es una comparación; cada
    diputado cuenta para el partido vigente en esa votación (afiliación histórica). El par
    de partidos se ordena alfabéticamente y un mismo partido se compara consigo mismo.
    Devuelve partido_a, partido_b, votacion_id, n_comparaciones y n_distintos.
    """
    matriz = AfinidadPares._validar_matriz(matriz_nominal)
    partido = (
        afiliacion.assign(
            diputado_id=afiliacion["diputado_id"].map(_clave),
            votacion_id=afiliacion["votacion_id"].map(_clave),
        )
        .dropna(subset=["partido_alias"])
        .set_index(["diputado_id", "votacion_id"])["partido_alias"]
        .astype(str)
    )
    columnas = ["partido_a", "partido_b", "votacion_id", "n_comparaciones", "n_distintos"]
    filas = []
    for votacion_id, columna in matriz.items():
        clave = _clave(votacion_id)
        observados = columna.dropna().rename("voto").to_frame()
        observados["partido"] = [
            partido.get((_clave(d), clave)) for d in observados.index
        ]
        observados = observados.dropna(subset=["partido"])
        if observados.empty:
            continue
        # Partido × categoría: n_p,c diputados del partido p que votaron c.
        tabla = pd.crosstab(observados["partido"], observados["voto"])
        tabla = tabla.sort_index()
        conteo = tabla.to_numpy(dtype=np.int64)
        total = conteo.sum(axis=1)
        nombres = list(tabla.index)
        for i, j in itertools.combinations_with_replacement(range(len(nombres)), 2):
            if i == j:
                n = int(total[i] * (total[i] - 1) // 2)
                iguales = int((conteo[i] * (conteo[i] - 1) // 2).sum())
            else:
                n = int(total[i] * total[j])
                iguales = int((conteo[i] * conteo[j]).sum())
            if n:
                filas.append((nombres[i], nombres[j], votacion_id, n, n - iguales))
    return pd.DataFrame(filas, columns=columnas)


def hamming_entre_partidos(
    matriz_nominal: pd.DataFrame | None = None,
    afiliacion: pd.DataFrame | None = None,
    *,
    comparaciones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Distancia de Hamming media entre partidos sobre todas sus comparaciones de co-votos.

    hamming = votos distintos / comparaciones, con el partido vigente en cada votación.
    Los pares sin comparaciones no aparecen (no se imputan). Acepta las comparaciones ya
    calculadas para evitar recalcularlas.
    """
    if comparaciones is None:
        comparaciones = comparaciones_entre_partidos(matriz_nominal, afiliacion)
    resumen = comparaciones.groupby(["partido_a", "partido_b"], as_index=False).agg(
        n_comparaciones=("n_comparaciones", "sum"),
        n_distintos=("n_distintos", "sum"),
        n_votaciones=("votacion_id", "nunique"),
    )
    resumen.insert(2, "hamming", resumen["n_distintos"] / resumen["n_comparaciones"])
    resumen["estado"] = "DESCRIPTIVO_SIN_PADRON"
    return resumen


def cargar_matriz_nominal(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la matriz nominal: {ruta}")

    tabla = pd.read_csv(ruta)
    if "diputado_id" not in tabla.columns:
        raise ErrorAfinidad(
            "matriz_nominal.csv debe contener la columna diputado_id."
        )

    return tabla.set_index("diputado_id")


def cargar_min_covotos(ruta_config: str | Path) -> int:
    ruta = Path(ruta_config)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la configuración: {ruta}")
    with ruta.open("rb") as archivo:
        config = tomllib.load(archivo)

    try:
        return int(config["pares"]["min_covotos"])
    except KeyError as exc:
        raise ErrorAfinidad(
            "analisis.toml no contiene pares.min_covotos."
        ) from exc


def guardar_resultados(
    resultado: ResultadoAfinidad,
    ruta_pares: str | Path,
    ruta_matriz: str | Path,
) -> None:
    ruta_pares = Path(ruta_pares)
    ruta_matriz = Path(ruta_matriz)
    ruta_pares.parent.mkdir(parents=True, exist_ok=True)
    ruta_matriz.parent.mkdir(parents=True, exist_ok=True)

    resultado.pares.to_csv(ruta_pares, index=False, lineterminator="\n")
    resultado.matriz_hamming.to_csv(ruta_matriz, index=True, lineterminator="\n")


def ejecutar(
    *,
    ruta_matriz: str | Path = RUTA_MATRIZ_POR_DEFECTO,
    ruta_config: str | Path = RUTA_CONFIG_POR_DEFECTO,
    ruta_salida: str | Path = RUTA_SALIDA_POR_DEFECTO,
    ruta_matriz_hamming: str | Path = RUTA_MATRIZ_HAMMING_POR_DEFECTO,
    ruta_afiliacion: str | Path = RUTA_AFILIACION_POR_DEFECTO,
    ruta_hamming_partidos: str | Path = RUTA_HAMMING_PARTIDOS_POR_DEFECTO,
    raiz: Path | None = None,
) -> ResultadoAfinidad:
    raiz = (raiz or Path(__file__).resolve().parents[3]).resolve()

    def resolver(ruta: str | Path) -> Path:
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    matriz = cargar_matriz_nominal(resolver(ruta_matriz))
    min_covotos = cargar_min_covotos(resolver(ruta_config))

    resultado = AfinidadPares(min_covotos=min_covotos).calcular(matriz)
    guardar_resultados(
        resultado,
        resolver(ruta_salida),
        resolver(ruta_matriz_hamming),
    )

    partidos = hamming_entre_partidos(matriz, pd.read_csv(resolver(ruta_afiliacion)))
    destino = resolver(ruta_hamming_partidos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    partidos.to_csv(destino, index=False, lineterminator="\n")
    return resultado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula afinidad nominal entre pares de diputados."
    )
    parser.add_argument("--matriz", default=str(RUTA_MATRIZ_POR_DEFECTO))
    parser.add_argument("--config", default=str(RUTA_CONFIG_POR_DEFECTO))
    parser.add_argument("--salida", default=str(RUTA_SALIDA_POR_DEFECTO))
    parser.add_argument(
        "--matriz-hamming",
        default=str(RUTA_MATRIZ_HAMMING_POR_DEFECTO),
    )
    return parser


def main() -> int:
    args = _crear_parser().parse_args()

    try:
        resultado = ejecutar(
            ruta_matriz=args.matriz,
            ruta_config=args.config,
            ruta_salida=args.salida,
            ruta_matriz_hamming=args.matriz_hamming,
        )
    except (ErrorAfinidad, FileNotFoundError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Afinidad calculada: "
        f"{resultado.n_pares} pares, "
        f"{resultado.n_pares_incluidos} con el mínimo de co-votos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
