"""Contrato de datos de F4: rutas, productores, columnas y códigos de razón.

Es la referencia única de dónde vive cada salida, quién la produce, qué columnas debe tener y
qué códigos de razón y estado puede contener. Los módulos conservan sus rutas por defecto; los
tests verifican que coincidan con este contrato. La conciliación, el universo por método, el
pipeline y el manifiesto de entrega leen las rutas desde aquí.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]

PROCESADOS = "F4/data/processed"
REPORTES = "F4/data/reports"
BCALL = "F4/data/results/individual/bcall"
PCA = "F4/data/results/individual/pca"
GRUPOS = "F4/data/results/grupos"
PARES = "F4/data/results/pares"
PARTIDOS = "F4/data/results/partidos"
COMPARACION = "F4/data/results/comparacion"


@dataclass(frozen=True, slots=True)
class Salida:
    """Una salida de F4: ruta relativa a la raíz, productor y columnas exigidas."""

    ruta: str
    productor: str
    columnas: tuple[str, ...] = ()
    clave: tuple[str, ...] = ()
    # Columnas cuyos valores deben pertenecer a CODIGOS_RAZON o ESTADOS.
    codigos: tuple[str, ...] = field(default=())

    @property
    def es_tabla(self) -> bool:
        return self.ruta.endswith(".csv")


def _s(ruta, productor, columnas=(), clave=(), codigos=()) -> Salida:
    return Salida(ruta, productor, tuple(columnas), tuple(clave), tuple(codigos))


MATRIZ = ("diputado_id",)
SALIDAS: dict[str, Salida] = {
    # ------------------------------------------------- corte, auditoría y codificación
    "manifiesto_corte": _s(f"{REPORTES}/manifiesto_corte.json", "corte.py"),
    "auditoria": _s(f"{REPORTES}/auditoria_entrada.json", "auditoria.py"),
    "votos": _s(f"{PROCESADOS}/votos_codificados.csv", "codificacion.py",
                ["diputado_id", "votacion_id", "fecha", "partido_id", "partido_alias",
                 "opcion_codigo", "voto_nominal", "voto_ternario", "voto_binario", "observado"],
                ["diputado_id", "votacion_id"]),
    # ----------------------------------------------------- matrices y participación
    "nominal": _s(f"{PROCESADOS}/matriz_nominal.csv", "matrices.py", MATRIZ, MATRIZ),
    "ternaria": _s(f"{PROCESADOS}/matriz_ternaria.csv", "matrices.py", MATRIZ, MATRIZ),
    "binaria": _s(f"{PROCESADOS}/matriz_binaria.csv", "matrices.py", MATRIZ, MATRIZ),
    "mascara": _s(f"{PROCESADOS}/mascara_observacion.csv", "matrices.py", MATRIZ, MATRIZ),
    "afiliacion": _s(f"{PROCESADOS}/afiliacion_por_votacion.csv", "matrices.py",
                     ["diputado_id", "votacion_id", "fecha", "partido_id", "partido_nombre",
                      "partido_alias"], ["diputado_id", "votacion_id"]),
    "participacion": _s(f"{REPORTES}/participacion_observada.csv", "cobertura.py",
                        ["diputado_id", "n_votos_observados", "n_sin_registro",
                         "supera_umbral_bcall"], ["diputado_id"]),
    "celdas_sin_registro": _s(f"{REPORTES}/celdas_sin_registro.csv", "cobertura.py",
                              ["diputado_id", "votacion_id"], ["diputado_id", "votacion_id"]),
    "conteos_participacion": _s(f"{REPORTES}/conteos_partido_votacion.csv", "cobertura.py",
                                ["partido_id", "votacion_id", "Y", "N", "A", "T"],
                                ["partido_id", "votacion_id"]),
    "conciliacion_conteos": _s(f"{REPORTES}/conciliacion_conteos.json", "cobertura.py"),
    # ------------------------------------------------------------------------ B-Call
    "bcall_diputados": _s(f"{BCALL}/bcall_diputados.csv", "F4_02_bcall.ipynb",
                          ["diputado_id", "d1", "d2", "m_i", "participacion", "grupo_bcall",
                           "razon_NA_d1", "razon_NA_d2", "estado"], ["diputado_id"],
                          ["razon_NA_d1", "razon_NA_d2", "estado"]),
    "bcall_seleccion": _s(f"{BCALL}/seleccion.csv", "F4_02_bcall.ipynb",
                          ["diputado_id", "participacion", "incluido", "razon_exclusion"],
                          ["diputado_id"], ["razon_exclusion"]),
    "bcall_votaciones": _s(f"{BCALL}/votaciones.csv", "F4_02_bcall.ipynb",
                           ["votacion_id", "orientacion", "incluida_por_varianza",
                            "razones_exclusion"], ["votacion_id"], ["razones_exclusion"]),
    "votos_orientados": _s(f"{BCALL}/votos_orientados.csv", "F4_02_bcall.ipynb",
                           MATRIZ, MATRIZ),
    "externo_diputados": _s(f"{BCALL}/diputados_externo.csv", "F4_02_bcall.ipynb",
                            ["diputado_id", "d1", "d2", "estado"], ["diputado_id"],
                            ["razon_NA_d1", "razon_NA_d2", "estado"]),
    "externo_seleccion": _s(f"{BCALL}/seleccion_externo.csv", "F4_02_bcall.ipynb",
                            ["diputado_id", "incluido", "razon_exclusion"], ["diputado_id"],
                            ["razon_exclusion"]),
    "perfiles_voto": _s(f"{BCALL}/perfiles_voto.csv", "F4_02_bcall.ipynb",
                        ["patron", "diputados", "d1", "d2", "grupo", "partidos"]),
    "posiciones_revision": _s(f"{BCALL}/posiciones_revision.csv", "F4_02_bcall.ipynb",
                              ["diputado_id", "criterio", "d1", "d2", "patron"]),
    "robustez_pivote": _s(f"{BCALL}/robustez_pivote.csv", "F4_02_bcall.ipynb",
                          ["diputado_id", "mismos_grupos", "max_abs_diferencia_d1"],
                          ["diputado_id"]),
    "ejecucion_bcall": _s(f"{BCALL}/ejecucion_bcall.json", "F4_02_bcall.ipynb"),
    # ---------------------------------------------------- posición y cohesión partidaria
    "posicion": _s(f"{PARTIDOS}/posicion_partidaria.csv", "posicion_partidos.py",
                   ["nivel", "partido_id", "partido_alias", "votacion_id", "b_pj", "n_decisiones",
                    "incluido", "P_p", "n_votaciones", "mediana_d1", "iqr_d1", "tipo_grupo",
                    "publicable", "razon_no_publicable", "estado"], (),
                   ["razon_exclusion", "razon_NA", "razon_no_publicable", "estado"]),
    "cohesion": _s(f"{PARTIDOS}/cohesion_por_votacion.csv", "cohesion.py",
                   ["partido_id", "partido_alias", "votacion_id", "tipo_grupo", "Y", "N", "A",
                    "T", "agreement_index", "rice", "cohesion_entropica",
                    "publicable_ai_entropia", "publicable_rice", "votacion_unanime_sala"],
                   ["partido_id", "votacion_id"],
                   ["razon_na_ai_entropia", "razon_na_rice", "razon_no_publicable"]),
    "resumen_cohesion": _s(f"{PARTIDOS}/resumen_cohesion.csv", "cohesion.py",
                           ["partido_id", "partido_alias", "tipo_grupo", "mediana_ai",
                            "mediana_rice", "mediana_entropia", "publicable_resumen"],
                           ["partido_id"], ["razon_no_publicable_resumen"]),
    # ---------------------------------------------------------------- afinidad (pares)
    "afinidad": _s(f"{PARES}/afinidad_diputados.csv", "afinidad.py",
                   ["diputado_i", "diputado_j", "n_covotos", "acuerdo", "hamming", "incluido",
                    "estado"], ["diputado_i", "diputado_j"], ["razon_exclusion", "estado"]),
    "matriz_hamming": _s(f"{PARES}/matriz_hamming.csv", "afinidad.py", MATRIZ, MATRIZ),
    "hamming_partidos": _s(f"{PARES}/hamming_partidos.csv", "afinidad.py",
                           ["partido_a", "partido_b", "hamming", "n_comparaciones", "estado"],
                           ["partido_a", "partido_b"], ["estado"]),
    # ------------------------------------------------------------- clustering y PCA
    "clusters": _s(f"{GRUPOS}/clusters_diputados.csv", "agrupamiento.py",
                   ["diputado_id", "cluster", "cobertura_corpus", "estado"], ["diputado_id"],
                   ["estado"]),
    "exclusiones_clustering": _s(f"{GRUPOS}/exclusiones_clustering.csv", "agrupamiento.py",
                                 ["diputado_id", "razon_exclusion"], ["diputado_id"],
                                 ["razon_exclusion"]),
    "votaciones_clustering": _s(f"{GRUPOS}/seleccion_votaciones_clustering.csv",
                                "agrupamiento.py", ["votacion_id", "incluida", "razon"],
                                ["votacion_id"], ["razon"]),
    "matriz_hamming_clustering": _s(f"{GRUPOS}/matriz_hamming_clustering.csv",
                                    "agrupamiento.py", MATRIZ, MATRIZ),
    "enlace_clustering": _s(f"{GRUPOS}/enlace_clustering.csv", "agrupamiento.py",
                            ["nodo_izquierdo", "nodo_derecho", "distancia", "n_diputados"]),
    "pca": _s(f"{PCA}/coordenadas_diputados.csv", "pca_svd.py",
              ["diputado_id", "PC1", "PC2", "estado"], ["diputado_id"], ["estado"]),
    "exclusiones_pca": _s(f"{PCA}/exclusiones_pca.csv", "pca_svd.py",
                          ["diputado_id", "razon_exclusion"], ["diputado_id"],
                          ["razon_exclusion"]),
    "votaciones_pca": _s(f"{PCA}/seleccion_votaciones_pca.csv", "pca_svd.py",
                         ["votacion_id", "incluida", "razon"], ["votacion_id"], ["razon"]),
    "cargas_pca": _s(f"{PCA}/cargas_votaciones.csv", "pca_svd.py", ["votacion_id", "PC1"]),
    "varianza_pca": _s(f"{PCA}/varianza_explicada.csv", "pca_svd.py",
                       ["componente", "proporcion_varianza"]),
    "medias_pca": _s(f"{PCA}/medias_votaciones.csv", "pca_svd.py", ["votacion_id", "media"]),
    "comparacion": _s(f"{COMPARACION}/resumen_comparacion.csv", "comparacion_metodos.py",
                      ["ari_clustering_bcall", "pearson_pc1_alineado_d1", "estado"], (),
                      ["estado"]),
    "comparacion_clustering": _s(f"{COMPARACION}/comparacion_clustering_bcall.csv",
                                 "comparacion_metodos.py",
                                 ["diputado_id", "grupo_bcall", "cluster", "coincide"]),
    "comparacion_pca": _s(f"{COMPARACION}/comparacion_pca_bcall.csv", "comparacion_metodos.py",
                          ["diputado_id", "PC1_alineado", "d1"]),
    "correspondencia_clusters": _s(f"{COMPARACION}/correspondencia_clusters.csv",
                                   "comparacion_metodos.py", ["grupo_bcall"]),
    "sensibilidad_pca": _s(f"{COMPARACION}/sensibilidad_pca.csv", "F4_04_clustering_pca.ipynb",
                           ["escenario", "pearson_abs_pc1", "estado"], (), ["estado"]),
    "manifiesto_clustering_pca": _s(f"{COMPARACION}/manifiesto_clustering_pca.json",
                                    "F4_04_clustering_pca.ipynb"),
    # -------------------------------------- sensibilidad, universo y conciliación
    "sensibilidad": _s(f"{REPORTES}/sensibilidad.csv", "sensibilidad.py",
                       ["familia", "metodo", "unidad", "id", "escenario", "votacion_retirada",
                        "valor_base", "valor_alternativo", "diferencia_abs", "estado"], (),
                       ["razon_NA_base", "estado"]),
    "sensibilidad_clustering": _s(f"{REPORTES}/sensibilidad_clustering.csv", "sensibilidad.py",
                                  ["escenario", "ari", "n_cambios_grupo", "estado"], (),
                                  ["estado"]),
    "sensibilidad_clustering_diputados": _s(f"{REPORTES}/sensibilidad_clustering_diputados.csv",
                                            "sensibilidad.py",
                                            ["escenario", "diputado_id", "cambio_grupo"]),
    "universo_por_metodo": _s(f"{REPORTES}/universo_por_metodo.csv", "universo.py",
                              ["metodo", "rol", "unidad", "n_corpus", "n_incluidos",
                               "n_excluidos", "estado"], ["metodo"], ["estado"]),
    "conciliacion_resultados": _s(f"{REPORTES}/conciliacion_resultados.json", "conciliacion.py"),
    "decisiones": _s(f"{REPORTES}/decisiones_metodologicas.json", "protocolo metodológico"),
    "manifiesto_entrega": _s(f"{REPORTES}/manifiesto_entrega.json", "exportacion.py"),
}

# Códigos de razón: catálogo normativo del protocolo y los códigos implementados que lo concretan.
CATALOGO_NORMATIVO = frozenset({
    "SIN_DECISION_SUSTANTIVA", "SIN_VOTACIONES_UTILIZABLES", "SIN_VARIANZA",
    "ORIENTACION_INDETERMINADA", "REFERENCIA_EXTERNA_PENDIENTE", "DENOMINADOR_CERO",
    "DENOMINADOR_NO_VERIFICABLE", "COBERTURA_INSUFICIENTE", "MINIMO_ABSOLUTO_INSUFICIENTE",
    "D2_MENOS_DE_DOS_VOTACIONES", "SIN_VOTOS_BINARIOS", "AFILIACION_NO_RESUELTA",
    "SIN_COVOTOS", "METODO_CONDICIONAL_NO_HABILITADO", "PROTOCOLO_NO_APROBADO",
})
EQUIVALENCIAS = {
    "MENOS_DE_DOS_VOTACIONES": "D2_MENOS_DE_DOS_VOTACIONES",
    "SIN_VARIANZA_O_MENOS_DE_DOS_OBSERVACIONES": "SIN_VARIANZA",
    "SIN_DECISION_SUSTANTIVA_INDIVIDUAL": "SIN_DECISION_SUSTANTIVA",
    "DECISIONES_INSUFICIENTES_PARTIDO_VOTACION": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "VOTACIONES_INSUFICIENTES_PARA_PERFIL_PARTIDARIO": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "COVOTOS_INSUFICIENTES": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "votaciones_validas_bajo_minimo": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "T_bajo_minimo": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "binarios_bajo_minimo_rice": "MINIMO_ABSOLUTO_INSUFICIENTE",
    "T_cero": "SIN_DECISION_SUSTANTIVA",
    "sin_si_ni_no": "SIN_VOTOS_BINARIOS",
    "CASO_INCOMPLETO": "SIN_DECISION_SUSTANTIVA",
}
# Códigos propios de un método, sin equivalente en el catálogo normativo.
ADICIONALES = frozenset({
    "PARTICIPACION_NO_SUPERA_UMBRAL", "SIN_CLUSTER", "MEDIA_DE_GRUPO_NO_ESTIMABLE",
    "SIN_ASIGNACION_EXTERNA", "SIN_CLASIFICACION_EXTERNA", "SIN_VARIACION_OBSERVADA",
    "ESCENARIO_NO_ESTIMABLE", "MENOS_DE_DOS_DIPUTADOS_COMUNES", "SIN_RESULTADO_BCALL",
    "UNIDAD_NO_PRESENTE_EN_ESCENARIO", "SIN_COMPARACIONES", "GRUPO_INDEPENDIENTES",
    "grupo_independientes", "INFORMATIVA",
})
CODIGOS_RAZON = CATALOGO_NORMATIVO | frozenset(EQUIVALENCIAS) | ADICIONALES
# Todos los resultados son descriptivos (puerta de clasificación de resultados): no hay
# estado de publicación principal.
ESTADOS = frozenset({
    "DESCRIPTIVO", "DESCRIPTIVO_SIN_PADRON", "DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS",
    "DESCRIPTIVO_CASOS_COMPLETOS", "DESCRIPTIVO_COMPLEMENTARIO", "ESCENARIO_NO_ESTIMABLE",
})
CODIGOS_VALIDOS = CODIGOS_RAZON | ESTADOS


def ruta(nombre: str, raiz: Path | str = RAIZ_REPOSITORIO) -> Path:
    """Ruta absoluta de una salida del contrato."""
    return Path(raiz) / SALIDAS[nombre].ruta


def _valores_codigo(serie: pd.Series) -> set[str]:
    """Códigos de una columna; acepta listas JSON (["A", "B"]) y valores separados por «;»."""
    valores: set[str] = set()
    for valor in serie.dropna().astype(str):
        if valor.startswith("["):
            valores.update(str(v) for v in json.loads(valor))
        else:
            valores.update(v.strip() for v in valor.split(";") if v.strip())
    return valores


def validar_tabla(nombre: str, tabla: pd.DataFrame) -> list[str]:
    """Problemas de una tabla frente a su contrato (lista vacía si cumple)."""
    salida = SALIDAS[nombre]
    problemas = [f"{nombre}: falta la columna {c}" for c in salida.columnas
                 if c not in tabla.columns]
    if salida.clave and set(salida.clave) <= set(tabla.columns):
        if tabla.duplicated(list(salida.clave)).any():
            problemas.append(f"{nombre}: clave {salida.clave} duplicada")
    for columna in salida.codigos:
        if columna in tabla.columns:
            desconocidos = sorted(_valores_codigo(tabla[columna]) - CODIGOS_VALIDOS)
            if desconocidos:
                problemas.append(f"{nombre}.{columna}: códigos fuera del contrato {desconocidos}")
    return problemas


def validar_repositorio(raiz: Path | str = RAIZ_REPOSITORIO,
                        omitir: tuple[str, ...] = ()) -> list[str]:
    """Problemas de todas las salidas del contrato: faltantes, vacías, columnas y códigos."""
    problemas = []
    for nombre, salida in SALIDAS.items():
        if nombre in omitir:
            continue
        archivo = Path(raiz) / salida.ruta
        if not archivo.is_file() or archivo.stat().st_size == 0:
            problemas.append(f"{nombre}: falta o está vacío {salida.ruta}")
        elif salida.es_tabla:
            problemas += validar_tabla(nombre, pd.read_csv(archivo, low_memory=False))
    return problemas


def main(raiz: Path | str = RAIZ_REPOSITORIO) -> int:
    problemas = validar_repositorio(raiz)
    for problema in problemas:
        print(problema, file=sys.stderr)
    print(f"Contrato: {len(SALIDAS)} salidas, {len(problemas)} problemas.")
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
