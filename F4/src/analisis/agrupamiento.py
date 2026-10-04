"""Clustering descriptivo mediante Hamming y enlace promedio."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from .afinidad import (
    CATEGORIAS_VALIDAS,
    AfinidadPares,
    cargar_matriz_nominal,
)


class ErrorAgrupamiento(ValueError):
    """El universo no permite construir una partición comparable."""


@dataclass(frozen=True, slots=True)
class ResultadoAgrupamiento:
    diputados: pd.DataFrame
    exclusiones: pd.DataFrame
    matriz_hamming: pd.DataFrame
    enlace: np.ndarray
    votaciones_incluidas: tuple
    votaciones_excluidas: tuple
    min_covotos: int


class AgrupamientoDiputados:
    """Agrupa votos registrados; no estima asistencia ni ideología general."""

    def __init__(
        self,
        *,
        cobertura_minima: float = 0.80,
        n_clusters: int = 2,
    ):
        if (
            isinstance(cobertura_minima, bool)
            or not isinstance(cobertura_minima, Real)
            or not np.isfinite(cobertura_minima)
            or not 0 < cobertura_minima <= 1
        ):
            raise ValueError("cobertura_minima debe estar en (0, 1].")

        if (
            isinstance(n_clusters, bool)
            or not isinstance(n_clusters, Integral)
            or n_clusters < 2
        ):
            raise ValueError(
                "n_clusters debe ser un entero mayor o igual a 2."
            )

        self.cobertura_minima = float(cobertura_minima)
        self.n_clusters = int(n_clusters)

    def calcular(
        self,
        matriz_nominal: pd.DataFrame,
    ) -> ResultadoAgrupamiento:
        self._validar_entrada(matriz_nominal)
        matriz = matriz_nominal.copy()

        # Seleccionar columnas antes de filtrar personas.
        # Las votaciones unánimes o sin observaciones se excluyen.
        informativas = matriz.nunique(dropna=True).gt(1)
        incluidas = tuple(matriz.columns[informativas])
        excluidas = tuple(matriz.columns[~informativas])

        if len(incluidas) < 2:
            raise ErrorAgrupamiento(
                "Se requieren al menos dos votaciones informativas."
            )

        matriz = matriz.loc[:, informativas]
        n_votaciones = matriz.shape[1]

        # La tolerancia evita errores de techo por coma flotante.
        minimo = max(
            1,
            ceil(self.cobertura_minima * n_votaciones - 1e-12),
        )

        conteos = matriz.notna().sum(axis=1)
        conservar = conteos.ge(minimo)

        seleccion = pd.DataFrame(
            {
                "n_votos_observados": conteos,
                "n_votaciones_informativas": n_votaciones,
                "cobertura_corpus": conteos / n_votaciones,
            }
        )
        seleccion.index.name = "diputado_id"

        exclusiones = seleccion.loc[~conservar].copy()
        exclusiones["razon_exclusion"] = "COBERTURA_INSUFICIENTE"

        universo = matriz.loc[conservar]

        if len(universo) < self.n_clusters:
            raise ErrorAgrupamiento(
                f"Quedan {len(universo)} diputados para "
                f"{self.n_clusters} clusters."
            )

        # Hamming nominal = desacuerdos / decisiones compartidas.
        # AfinidadPares deja NA si un par no alcanza el mínimo.
        afinidad = AfinidadPares(min_covotos=minimo).calcular(universo)
        distancias = afinidad.matriz_hamming
        valores = distancias.to_numpy(dtype=float)

        insuficientes = afinidad.pares.loc[
            ~afinidad.pares["incluido"]
        ]

        if not insuficientes.empty:
            ejemplos = insuficientes[
                ["diputado_i", "diputado_j", "n_covotos"]
            ].head(5).to_dict("records")

            raise ErrorAgrupamiento(
                f"{len(insuficientes)} pares no alcanzan "
                f"{minimo} co-votos. Ejemplos: {ejemplos}. "
                "No se imputan distancias; revise un universo "
                "alternativo explícito."
            )

        if not np.isfinite(valores).all():
            raise ErrorAgrupamiento(
                "La matriz de distancias contiene NA o infinitos."
            )

        if not np.allclose(
            valores, valores.T, atol=1e-12, rtol=0
        ):
            raise ErrorAgrupamiento(
                "La matriz de distancias no es simétrica."
            )

        if not np.allclose(
            np.diag(valores), 0, atol=1e-12, rtol=0
        ):
            raise ErrorAgrupamiento(
                "La diagonal de distancias debe ser cero."
            )

        if (valores < 0).any() or (valores > 1).any():
            raise ErrorAgrupamiento(
                "Las distancias deben estar entre cero y uno."
            )

        if not (valores > 0).any():
            raise ErrorAgrupamiento(
                "No hay separación entre los perfiles retenidos."
            )

        # linkage recibe distancias condensadas.
        # Pasar directamente la matriz cuadrada cambiaría el análisis.
        enlace = linkage(
            squareform(valores, checks=False),
            method="average",
        )

        # Etiquetas neutrales: 1, 2, ..., n_clusters.
        etiquetas = (
            cut_tree(enlace, n_clusters=self.n_clusters).ravel() + 1
        )

        diputados = seleccion.loc[conservar].copy()
        diputados["cluster"] = etiquetas
        diputados["n_clusters"] = self.n_clusters
        diputados["cobertura_minima"] = self.cobertura_minima
        diputados["min_covotos"] = minimo
        diputados["estado"] = (
            "DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS"
        )

        return ResultadoAgrupamiento(
            diputados=diputados.reset_index(),
            exclusiones=exclusiones.reset_index(),
            matriz_hamming=distancias,
            enlace=enlace,
            votaciones_incluidas=incluidas,
            votaciones_excluidas=excluidas,
            min_covotos=minimo,
        )

    @staticmethod
    def _validar_entrada(matriz: pd.DataFrame) -> None:
        if not isinstance(matriz, pd.DataFrame):
            raise TypeError(
                "matriz_nominal debe ser un DataFrame."
            )

        if matriz.empty:
            raise ErrorAgrupamiento(
                "La matriz nominal está vacía."
            )

        for eje in (matriz.index, matriz.columns):
            if (
                isinstance(eje, pd.MultiIndex)
                or eje.has_duplicates
                or eje.hasnans
            ):
                raise ErrorAgrupamiento(
                    "Los identificadores deben ser únicos y no nulos."
                )

        observados = set(matriz.stack().dropna().tolist())
        invalidos = observados - CATEGORIAS_VALIDAS

        if invalidos:
            raise ErrorAgrupamiento(
                f"Categorías inválidas: {invalidos}."
            )


def guardar_resultados(
    resultado: ResultadoAgrupamiento,
    directorio: str | Path,
) -> None:
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    resultado.diputados.to_csv(
        directorio / "clusters_diputados.csv",
        index=False,
        lineterminator="\n",
    )

    resultado.exclusiones.to_csv(
        directorio / "exclusiones_clustering.csv",
        index=False,
        lineterminator="\n",
    )

    resultado.matriz_hamming.to_csv(
        directorio / "matriz_hamming_clustering.csv",
        lineterminator="\n",
    )

    pd.DataFrame(
        resultado.enlace,
        columns=[
            "nodo_izquierdo",
            "nodo_derecho",
            "distancia",
            "n_diputados",
        ],
    ).to_csv(
        directorio / "enlace_clustering.csv",
        index=False,
        lineterminator="\n",
    )

    votaciones = pd.DataFrame(
        [
            (v, True, "INFORMATIVA")
            for v in resultado.votaciones_incluidas
        ]
        + [
            (v, False, "SIN_VARIACION_OBSERVADA")
            for v in resultado.votaciones_excluidas
        ],
        columns=["votacion_id", "incluida", "razon"],
    )

    votaciones.to_csv(
        directorio / "seleccion_votaciones_clustering.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--matriz",
        default="F4/data/processed/matriz_nominal.csv",
    )
    parser.add_argument(
        "--salida",
        default="F4/data/results/grupos",
    )
    parser.add_argument(
        "--cobertura-minima",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
    )

    args = parser.parse_args()
    raiz = Path(__file__).resolve().parents[3]

    def resolver(ruta: str) -> Path:
        path = Path(ruta)
        return path if path.is_absolute() else raiz / path

    modelo = AgrupamientoDiputados(
        cobertura_minima=args.cobertura_minima,
        n_clusters=args.n_clusters,
    )

    resultado = modelo.calcular(
        cargar_matriz_nominal(resolver(args.matriz))
    )

    guardar_resultados(resultado, resolver(args.salida))

    print(f"Diputados incluidos: {len(resultado.diputados)}")
    print(f"Diputados excluidos: {len(resultado.exclusiones)}")
    print(
        "Votaciones informativas:",
        len(resultado.votaciones_incluidas),
    )
    print(
        "Mínimo de co-votos exigido:",
        resultado.min_covotos,
    )
    print(
        "Tamaños de clusters:",
        resultado.diputados["cluster"].value_counts().to_dict(),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())