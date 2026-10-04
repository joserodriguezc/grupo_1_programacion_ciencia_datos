"""Port del núcleo bcall() R, commit 4c2e2a98286ceaede7df7d5ffa6a006358b963cd.

Fórmulas y orden equivalentes a BCall$calculate() y Clustering con pivote explícito.
Los diagnósticos adicionales no intervienen en el cálculo. No autoriza publicación.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


@dataclass
class ResultadoBCall:
    diputados: pd.DataFrame
    votaciones: pd.DataFrame
    votos_orientados: pd.DataFrame
    seleccion: pd.DataFrame
    clustering_auto: pd.Series | None = None


class ModeloBCall:
    """Interfaz equivalente a bcall(rollcall, clustering, pivot, threshold=0.1).

    clustering: Series indexada por diputado o DataFrame de una sola columna.
    Exactamente dos etiquetas; el grupo del pivote define el lado positivo.
    Las personas sin asignación se excluyen antes de calcular participación.
    El filtro es > threshold sobre todas las columnas, no >= ni padrón elegible.
    """

    @staticmethod
    def _distancias(a: np.ndarray, b: np.ndarray, metodo: int) -> np.ndarray:
        """Distancias de Clustering$get_distances, incluyendo sin co-votos=0."""
        comunes = ~np.isnan(a[:, None, :]) & ~np.isnan(b[None, :, :])
        diferencia = np.where(comunes, a[:, None, :] - b[None, :, :], 0.0)
        n = comunes.sum(axis=2)
        if metodo == 1:
            suma = np.abs(diferencia).sum(axis=2)
            divisor = 2 * n
        else:
            suma = np.sqrt((diferencia ** 2).sum(axis=2))
            divisor = 2 * np.sqrt(n)
        return np.divide(suma, divisor, out=np.zeros_like(suma), where=divisor > 0)

    def calcular_auto(
        self,
        matriz: pd.DataFrame,
        pivot: Hashable,
        distance_method: int = 1,
        threshold: float = 0.1,
        max_iteraciones: int = 1000,
    ) -> ResultadoBCall:
        """Port de bcall_auto() con pivote explícito y clustering anterior al filtro.

        No recibe clasificación externa. Agrupa TODOS los diputados recibidos,
        después calcular() aplica >threshold. Los empates siguen el orden columna
        de R en la inicialización/asignación; en reclasificación van a 'right'.
        Guardas adicionales ante ciclo, grupo vacío o separación inicial nula:
        se detiene con error, nunca inventa una partición alternativa.
        """
        votos = self._validar_matriz(matriz)
        if pivot not in votos.index:
            raise ValueError("El pivote debe estar en la matriz.")
        if isinstance(distance_method, bool) or distance_method not in (1, 2):
            raise ValueError("distance_method debe ser 1 (Manhattan) o 2 (Euclídea).")
        if isinstance(max_iteraciones, bool) or not isinstance(max_iteraciones, int):
            raise ValueError("max_iteraciones debe ser entero positivo.")
        if max_iteraciones < 1:
            raise ValueError("max_iteraciones debe ser entero positivo.")
        x = votos.to_numpy()
        d = self._distancias(x, x, distance_method)
        # R which(..., arr.ind=TRUE)[1,] recorre matrices por columnas.
        par = np.unravel_index(np.argmax(d.ravel(order="F")), d.shape, order="F")
        if d[par] == 0 or par[0] == par[1]:
            raise ValueError("Clustering automático sin separación inicial positiva.")
        grupos = [[int(par[0])], [int(par[1])]]

        def centro(indices):
            return pd.DataFrame(x[indices]).mean().to_numpy()

        while sum(map(len, grupos)) < len(x):
            pendientes = [i for i in range(len(x)) if i not in grupos[0] + grupos[1]]
            centros = np.stack([centro(g) for g in grupos])
            dist = self._distancias(x[pendientes], centros, distance_method)
            fila, columna = np.unravel_index(
                np.argmin(dist.ravel(order="F")), dist.shape, order="F"
            )
            grupos[columna].append(pendientes[fila])

        p = votos.index.get_loc(pivot)
        izq, der = (grupos[1], grupos[0]) if p in grupos[0] else (grupos[0], grupos[1])
        etiquetas = np.empty(len(x), dtype=object)
        etiquetas[izq], etiquetas[der] = "left", "right"
        visitados = set()
        for _ in range(max_iteraciones):
            estado = tuple(etiquetas)
            if estado in visitados:
                raise ValueError("Clustering automático en ciclo; no se fuerza convergencia.")
            visitados.add(estado)
            if not izq or not der:
                raise ValueError("Clustering automático produce un grupo vacío.")
            centros = np.stack([centro(izq), centro(der)])
            dist = self._distancias(x, centros, distance_method)
            nuevas = np.where(dist[:, 0] < dist[:, 1], "left", "right")
            movidos = np.flatnonzero(nuevas != etiquetas)
            if not len(movidos):
                break
            # Mismo orden de traslado que el bucle R; afecta orden de centroides.
            for i in movidos:
                viejo = izq if etiquetas[i] == "left" else der
                nuevo = izq if nuevas[i] == "left" else der
                viejo.remove(i)
                nuevo.append(int(i))
            etiquetas = nuevas
        else:
            raise ValueError("Clustering automático supera max_iteraciones.")
        clustering = pd.Series(etiquetas, index=votos.index, name="auto_cluster")
        resultado = self.calcular(votos, clustering, pivot=pivot, threshold=threshold)
        resultado.clustering_auto = clustering
        # Interpretación efectiva del eje: el cluster FINAL del pivote define R.
        positivo = clustering.loc[pivot]
        resultado.diputados["grupo_bcall"] = np.where(
            resultado.diputados["cluster"].eq(positivo), "R", "L"
        )
        return resultado

    @staticmethod
    def _validar_matriz(matriz: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(matriz, pd.DataFrame):
            raise TypeError("La matriz debe ser un pandas.DataFrame.")
        if matriz.empty:
            raise ValueError("La matriz requiere diputados y votaciones.")
        for eje in (matriz.index, matriz.columns):
            if isinstance(eje, pd.MultiIndex) or eje.has_duplicates or eje.hasnans:
                raise ValueError("Los identificadores deben ser simples, únicos y no nulos.")
        for tipo in matriz.dtypes:
            if not is_numeric_dtype(tipo) or is_bool_dtype(tipo) or is_complex_dtype(tipo):
                raise ValueError("Los votos deben ser numéricos ternarios o NA.")
        if not (matriz.isna() | matriz.isin([-1, 0, 1])).all().all():
            raise ValueError("Solo se admiten -1, 0, 1 y NA.")
        return matriz.astype(float).copy()

    def calcular(
        self,
        matriz: pd.DataFrame,
        clustering: pd.Series | pd.DataFrame,
        pivot: Hashable,
        threshold: float = 0.1,
    ) -> ResultadoBCall:
        """Filtrar personas, orientar, estandarizar y calcular d1/d2 muestrales.

        Es una traducción del cálculo bcall() con grupos provistos, no bcall_auto().
        Errores estrictos adicionales: índices duplicados/nulos, etiquetas faltantes,
        bool/códigos no numéricos e infinitos. No cambia las fórmulas válidas del R.
        """
        votos = self._validar_matriz(matriz)
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold debe ser numérico entre 0 y 1.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold debe ser finito entre 0 y 1.")
        if isinstance(clustering, pd.DataFrame):
            if clustering.shape[1] != 1:
                raise ValueError("clustering debe tener exactamente una columna.")
            clustering = clustering.iloc[:, 0]
        if not isinstance(clustering, pd.Series):
            raise TypeError("clustering requiere Series o DataFrame de una columna.")
        if isinstance(clustering.index, pd.MultiIndex):
            raise ValueError("clustering requiere identificadores simples.")
        if clustering.index.has_duplicates or clustering.index.hasnans or clustering.isna().any():
            raise ValueError("clustering requiere IDs únicos y etiquetas no nulas.")
        if clustering.nunique() != 2:
            raise ValueError("clustering requiere exactamente dos etiquetas.")
        if pivot not in votos.index or pivot not in clustering.index:
            raise ValueError("El pivote debe estar en matriz y clustering.")

        # Equivalente al intersect de bcall(), conservando el orden de rollcall.
        comunes = votos.index[votos.index.isin(clustering.index)]
        x = votos.loc[comunes]
        grupos = clustering.reindex(comunes)
        participacion = x.notna().sum(axis=1) / x.shape[1]
        conservar = participacion.gt(threshold)
        x = x.loc[conservar]
        grupos = grupos.loc[conservar]
        if pivot not in x.index:
            raise ValueError("El pivote no supera el umbral de participación.")
        lado_r = grupos.eq(grupos.loc[pivot])
        if lado_r.all():
            raise ValueError("Debe permanecer al menos un diputado en el otro grupo.")

        media_l = x.loc[~lado_r].mean()
        media_r = x.loc[lado_r].mean()
        media = x.mean()
        escala = x.std(ddof=1)
        validas = escala.gt(0) & escala.notna()
        if not validas.any():
            raise ValueError("No hay votaciones con varianza positiva para analizar.")
        # En R, comparar una media NA propaga NA; no debe convertirse en -1.
        direccion = pd.Series(
            np.where(media_l < media_r, 1.0, -1.0), index=x.columns
        ).where(media_l.notna() & media_r.notna())
        y = x.loc[:, validas]
        u = y.sub(media[validas], axis=1).div(escala[validas], axis=1)
        u = u.mul(direccion[validas], axis=1)
        m = u.count(axis=1)
        d1 = u.mean(axis=1)
        d2 = u.std(axis=1, ddof=1)
        usadas, excluidas = [], []
        razones = []
        for j in x.columns:
            causas = []
            if not validas[j]:
                causas.append("SIN_VARIANZA_O_MENOS_DE_DOS_OBSERVACIONES")
            if pd.isna(direccion[j]):
                causas.append("MEDIA_DE_GRUPO_NO_ESTIMABLE")
            razones.append(causas)
        for i in x.index:
            usadas.append(u.columns[u.loc[i].notna()].tolist())
            detalle = []
            for j, causas_j in zip(x.columns, razones, strict=True):
                causas = list(causas_j)
                if pd.isna(x.loc[i, j]):
                    causas.append("SIN_DECISION_SUSTANTIVA_INDIVIDUAL")
                if causas:
                    detalle.append({"votacion_id": j, "razones": causas})
            excluidas.append(detalle)
        resumen = pd.DataFrame(
            {"d1": d1, "d2": d2, "m_i": m, "cluster": grupos,
             "participacion": participacion.loc[x.index],
             "votaciones_usadas": usadas, "votaciones_excluidas": excluidas,
             "razon_NA_d1": ["SIN_VOTACIONES_UTILIZABLES" if k == 0 else None for k in m],
             "razon_NA_d2": ["MENOS_DE_DOS_VOTACIONES" if k < 2 else None for k in m],
             "estado": "DESCRIPTIVO_SIN_PADRON"}, index=x.index.copy(),
        )
        resumen.index.name = "diputado_id"
        diagnostico = pd.DataFrame(
            {"n_observados": x.count(), "media": media, "desviacion": escala,
             "media_L": media_l, "media_R": media_r, "orientacion": direccion,
             "incluida_por_varianza": validas,
             "orientacion_definida": direccion.notna(), "razones_exclusion": razones},
            index=x.columns.copy(),
        )
        diagnostico.index.name = "votacion_id"
        tiene_cluster = votos.index.isin(clustering.index)
        seleccion = pd.DataFrame(
            {"tiene_cluster": tiene_cluster,
             "participacion": participacion.reindex(votos.index),
             "incluido": votos.index.isin(x.index),
             "razon_exclusion": [
                 "SIN_CLUSTER" if not asignado else
                 (None if i in x.index else "PARTICIPACION_NO_SUPERA_UMBRAL")
                 for i, asignado in zip(votos.index, tiene_cluster, strict=True)
             ]}, index=votos.index.copy(),
        )
        seleccion.index.name = "diputado_id"
        return ResultadoBCall(resumen, diagnostico, u, seleccion)
