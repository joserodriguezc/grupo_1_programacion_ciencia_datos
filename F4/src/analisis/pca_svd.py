"""PCA descriptivo sobre votos ternarios completos mediante SVD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)


class ErrorPCA(ValueError):
    """Los datos no permiten estimar el PCA solicitado."""


@dataclass(frozen=True, slots=True)
class ResultadoPCA:
    coordenadas: pd.DataFrame
    cargas: pd.DataFrame
    varianza: pd.DataFrame
    exclusiones: pd.DataFrame
    seleccion_votaciones: pd.DataFrame
    medias: pd.Series
    rango: int


class PCASVD:
    """PCA centrado, sin escalamiento y sobre casos completos.

    Las cargas son los coeficientes de los ejes, de modo que:
        coordenadas = (X - medias) @ cargas

    No se asigna significado ideológico a los componentes.
    """

    def __init__(self, *, n_componentes: int = 2):
        if (
            isinstance(n_componentes, bool)
            or not isinstance(n_componentes, Integral)
            or n_componentes < 1
        ):
            raise ValueError(
                "n_componentes debe ser un entero positivo."
            )

        self.n_componentes = int(n_componentes)

    def calcular(
        self,
        matriz_ternaria: pd.DataFrame,
    ) -> ResultadoPCA:
        self._validar(matriz_ternaria)
        matriz = matriz_ternaria.astype(float).copy()

        # Seleccionar columnas antes de seleccionar personas.
        informativas = matriz.nunique(dropna=True).gt(1)

        seleccion = pd.DataFrame(
            {
                "votacion_id": matriz.columns,
                "incluida": informativas.to_numpy(),
                "razon": np.where(
                    informativas,
                    "INFORMATIVA",
                    "SIN_VARIACION_OBSERVADA",
                ),
            }
        )

        matriz = matriz.loc[:, informativas]

        if matriz.shape[1] == 0:
            raise ErrorPCA(
                "No hay votaciones con variación observada."
            )

        # Los faltantes no se reemplazan por abstención ni por medias.
        completos = matriz.notna().all(axis=1)
        faltantes = matriz.isna().sum(axis=1)

        exclusiones = pd.DataFrame(
            {
                "diputado_id": matriz.index[~completos],
                "n_faltantes": faltantes.loc[
                    ~completos
                ].to_numpy(),
                "n_votaciones_informativas": matriz.shape[1],
                "razon_exclusion": "CASO_INCOMPLETO",
            }
        )

        x = matriz.loc[completos]

        if len(x) < 2:
            raise ErrorPCA(
                "Se requieren al menos dos diputados completos."
            )

        medias = x.mean(axis=0)
        centrada = x.sub(
            medias, axis=1
        ).to_numpy(dtype=float)

        # X centrada = U @ diag(s) @ V.T
        _, s, vt = np.linalg.svd(
            centrada,
            full_matrices=False,
        )

        tolerancia = (
            max(centrada.shape)
            * np.finfo(float).eps
            * s[0]
        )
        rango = int(np.sum(s > tolerancia))

        if self.n_componentes > rango:
            raise ErrorPCA(
                f"El rango efectivo es {rango}; "
                f"se solicitaron {self.n_componentes} componentes."
            )

        ejes = vt[:self.n_componentes].T.copy()

        # Convención de signo reproducible:
        # coeficiente de mayor magnitud positivo.
        # No representa una orientación izquierda/derecha.
        for k in range(self.n_componentes):
            j = int(np.argmax(np.abs(ejes[:, k])))

            if ejes[j, k] < 0:
                ejes[:, k] *= -1

        nombres = [
            f"PC{k + 1}"
            for k in range(self.n_componentes)
        ]

        coordenadas = pd.DataFrame(
            centrada @ ejes,
            index=x.index,
            columns=nombres,
        ).rename_axis("diputado_id").reset_index()

        coordenadas["n_votaciones"] = x.shape[1]
        coordenadas["estado"] = "DESCRIPTIVO_CASOS_COMPLETOS"

        cargas = pd.DataFrame(
            ejes,
            index=x.columns,
            columns=nombres,
        ).rename_axis("votacion_id").reset_index()

        # Reportar todos los componentes del espectro,
        # incluso aquellos sin coordenadas exportadas.
        autovalores = s ** 2 / (len(x) - 1)
        proporciones = autovalores / autovalores.sum()

        varianza = pd.DataFrame(
            {
                "componente": [
                    f"PC{k + 1}" for k in range(len(s))
                ],
                "autovalor": autovalores,
                "proporcion_varianza": proporciones,
                "proporcion_acumulada": np.cumsum(
                    proporciones
                ),
                "coordenadas_exportadas": (
                    np.arange(len(s)) < self.n_componentes
                ),
            }
        )

        return ResultadoPCA(
            coordenadas=coordenadas,
            cargas=cargas,
            varianza=varianza,
            exclusiones=exclusiones,
            seleccion_votaciones=seleccion,
            medias=medias,
            rango=rango,
        )

    @staticmethod
    def _validar(matriz):
        if not isinstance(matriz, pd.DataFrame):
            raise TypeError(
                "matriz_ternaria debe ser un DataFrame."
            )

        if matriz.empty:
            raise ErrorPCA("La matriz ternaria está vacía.")

        for eje in (matriz.index, matriz.columns):
            if (
                isinstance(eje, pd.MultiIndex)
                or eje.has_duplicates
                or eje.hasnans
            ):
                raise ErrorPCA(
                    "Los identificadores deben ser únicos y no nulos."
                )

        for tipo in matriz.dtypes:
            if (
                not is_numeric_dtype(tipo)
                or is_bool_dtype(tipo)
                or is_complex_dtype(tipo)
            ):
                raise ErrorPCA(
                    "Los votos deben ser numéricos ternarios o NA."
                )

        if not (
            matriz.isna() | matriz.isin([-1, 0, 1])
        ).all().all():
            raise ErrorPCA(
                "Solo se admiten -1, 0, 1 y NA."
            )


def guardar_resultados(
    resultado: ResultadoPCA,
    directorio: str | Path,
):
    salida = Path(directorio)
    salida.mkdir(parents=True, exist_ok=True)

    tablas = {
        "coordenadas_diputados.csv": resultado.coordenadas,
        "cargas_votaciones.csv": resultado.cargas,
        "varianza_explicada.csv": resultado.varianza,
        "exclusiones_pca.csv": resultado.exclusiones,
        "seleccion_votaciones_pca.csv": (
            resultado.seleccion_votaciones
        ),
        "medias_votaciones.csv": (
            resultado.medias
            .rename("media")
            .rename_axis("votacion_id")
            .reset_index()
        ),
    }

    for nombre, tabla in tablas.items():
        tabla.to_csv(salida / nombre, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--matriz",
        default="F4/data/processed/matriz_ternaria.csv",
    )
    parser.add_argument(
        "--salida",
        default="F4/data/results/individual/pca",
    )
    parser.add_argument(
        "--n-componentes",
        type=int,
        default=2,
    )

    args = parser.parse_args()
    raiz = Path(__file__).resolve().parents[3]

    def resolver(ruta):
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    try:
        tabla = pd.read_csv(resolver(args.matriz))

        if "diputado_id" not in tabla.columns:
            raise ErrorPCA(
                "La matriz debe contener diputado_id."
            )

        resultado = PCASVD(
            n_componentes=args.n_componentes
        ).calcular(
            tabla.set_index("diputado_id")
        )

        guardar_resultados(
            resultado,
            resolver(args.salida),
        )

    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(
        f"Diputados incluidos: {len(resultado.coordenadas)}"
    )
    print(
        f"Diputados excluidos: {len(resultado.exclusiones)}"
    )
    print(
        f"Votaciones informativas: {len(resultado.medias)}"
    )
    print(f"Rango efectivo: {resultado.rango}")

    for fila in resultado.varianza.head(
        args.n_componentes
    ).itertuples(index=False):
        print(
            f"{fila.componente}: "
            f"{100 * fila.proporcion_varianza:.2f}% de varianza"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())