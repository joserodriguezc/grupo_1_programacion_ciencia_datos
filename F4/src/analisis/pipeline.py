"""Pipeline reproducible de F4: regenera la cadena de módulos en orden.

Cada paso lee las salidas del anterior desde el contrato de datos (contratos.py) y escribe
las suyas en su ubicación oficial. Los parámetros vienen de F4/config/analisis.toml. Al final
se valida el repositorio contra el contrato.

Las salidas individuales de B-Call (F4/data/results/individual/bcall) las produce el notebook
F4_02, que también documenta el cotejo con la referencia externa. El pipeline no las
reemplaza: exige que existan antes de los pasos que las usan (comparación y universo).

Uso: python -m F4.src.analisis.pipeline [--pasos auditoria codificacion ...]
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from F4.src.analisis import (
    afinidad,
    auditoria,
    cobertura,
    codificacion,
    cohesion,
    conciliacion,
    contratos,
    exportacion,
    matrices,
    posicion_partidos,
    sensibilidad,
)
from F4.src.analisis.agrupamiento import AgrupamientoDiputados
from F4.src.analisis.agrupamiento import guardar_resultados as guardar_agrupamiento
from F4.src.analisis.comparacion_metodos import comparar
from F4.src.analisis.pca_svd import PCASVD
from F4.src.analisis.pca_svd import guardar_resultados as guardar_pca
from F4.src.analisis.universo import desde_repositorio as universo_desde_repositorio

RAIZ_REPOSITORIO = contratos.RAIZ_REPOSITORIO
# Parámetros del clustering y del PCA (los mismos de F4_04 y de [clustering] en el TOML).
COBERTURA_CLUSTERING = 0.80
N_CLUSTERS = 2
N_COMPONENTES = 2
# Salidas de F4_02 que los pasos posteriores necesitan.
SALIDAS_BCALL = ("bcall_diputados", "bcall_seleccion", "bcall_votaciones",
                 "externo_seleccion")


class ErrorPipeline(RuntimeError):
    """El pipeline no puede continuar o termina con salidas fuera del contrato."""


def _config(raiz: Path) -> dict:
    with (raiz / "F4/config/analisis.toml").open("rb") as archivo:
        return tomllib.load(archivo)


def _exigir_bcall(raiz: Path) -> None:
    faltan = [contratos.SALIDAS[n].ruta for n in SALIDAS_BCALL
              if not contratos.ruta(n, raiz).is_file()]
    if faltan:
        raise ErrorPipeline("Faltan salidas de B-Call; ejecute primero el notebook F4_02: "
                            + ", ".join(faltan))


def _clustering(raiz: Path) -> None:
    config = _config(raiz)
    sensibilidad.ejecutar_clustering(
        cobertura_base=COBERTURA_CLUSTERING,
        umbrales_cobertura=tuple(config["sensibilidad"]["clustering_umbrales_cobertura"]),
        n_clusters=N_CLUSTERS,
        bloques=config["sensibilidad"]["clustering_bloques"],
        raiz=raiz,
    )
    nominal = afinidad.cargar_matriz_nominal(contratos.ruta("nominal", raiz))
    resultado = AgrupamientoDiputados(cobertura_minima=COBERTURA_CLUSTERING,
                                      n_clusters=N_CLUSTERS).calcular(nominal)
    guardar_agrupamiento(resultado, contratos.ruta("clusters", raiz).parent)


def _pca_y_comparacion(raiz: Path) -> None:
    _exigir_bcall(raiz)
    ternaria = pd.read_csv(contratos.ruta("ternaria", raiz))
    pca = PCASVD(n_componentes=N_COMPONENTES).calcular(ternaria.set_index("diputado_id"))
    guardar_pca(pca, contratos.ruta("pca", raiz).parent)
    # Como F4_04: coordenadas PCA en memoria (releerlas del CSV redondea el último decimal).
    tablas = comparar(
        pd.read_csv(contratos.ruta("bcall_diputados", raiz)),
        pd.read_csv(contratos.ruta("clusters", raiz)),
        pca.coordenadas,
    )
    destino = contratos.ruta("comparacion", raiz).parent
    for nombre, tabla in tablas.items():
        tabla.to_csv(destino / nombre, index=False, lineterminator="\n")


def _universo(raiz: Path) -> None:
    _exigir_bcall(raiz)
    universo_desde_repositorio(raiz).calcular().to_csv(
        contratos.ruta("universo_por_metodo", raiz), index=False, lineterminator="\n")


# Orden de dependencia: cada paso usa las salidas de los anteriores.
PASOS: dict[str, Callable[[Path], object]] = {
    "auditoria": auditoria.main,
    "codificacion": lambda raiz: codificacion.codificar_archivo(raiz=raiz),
    "matrices": lambda raiz: matrices.ejecutar(raiz=raiz),
    "cobertura": cobertura.main,
    "cohesion": cohesion.main,
    "posicion": lambda raiz: posicion_partidos.ejecutar(raiz=raiz),
    "afinidad": lambda raiz: afinidad.ejecutar(raiz=raiz),
    "sensibilidad": lambda raiz: sensibilidad.ejecutar(raiz=raiz),
    "clustering": _clustering,
    "pca_comparacion": _pca_y_comparacion,
    "universo": _universo,
    "conciliacion": conciliacion.main,
    # Manifiesto de entrega; valida el contrato y exige el protocolo aprobado.
    "exportacion": exportacion.main,
}


def ejecutar(raiz: Path | str = RAIZ_REPOSITORIO, pasos: Iterable[str] | None = None,
             validar: bool = True) -> list[str]:
    """Ejecuta los pasos indicados (todos por defecto) en el orden de PASOS.

    Devuelve los pasos ejecutados. Con validar=True, termina con ErrorPipeline si alguna
    salida queda fuera del contrato.
    """
    raiz = Path(raiz)
    pedidos = list(PASOS) if pasos is None else list(pasos)
    desconocidos = sorted(set(pedidos) - set(PASOS))
    if desconocidos:
        raise ErrorPipeline(f"Pasos desconocidos: {desconocidos}. Disponibles: {list(PASOS)}")
    ejecutados = [paso for paso in PASOS if paso in pedidos]
    for paso in ejecutados:
        PASOS[paso](raiz)
    if validar:
        omitir = () if "exportacion" in ejecutados else ("manifiesto_entrega",)
        problemas = contratos.validar_repositorio(raiz, omitir=omitir)
        if problemas:
            raise ErrorPipeline("Salidas fuera del contrato:\n" + "\n".join(problemas))
    return ejecutados


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pasos", nargs="+", choices=list(PASOS),
                        help="Pasos a ejecutar (por defecto, todos en orden).")
    args = parser.parse_args(argv)
    try:
        ejecutados = ejecutar(pasos=args.pasos)
    except ErrorPipeline as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Pipeline F4: {len(ejecutados)} pasos ejecutados ({', '.join(ejecutados)}); "
          "salidas conformes al contrato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
