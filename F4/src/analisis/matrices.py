from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RUTA_ENTRADA_POR_DEFECTO = Path("F4/data/processed/votos_codificados.csv")
RUTA_AUDITORIA_POR_DEFECTO = Path("F4/data/reports/auditoria_entrada.json")
DIRECTORIO_SALIDA_POR_DEFECTO = Path("F4/data/processed")

COLUMNAS_REQUERIDAS = (
    "diputado_id",
    "votacion_id",
    "fecha",
    "partido_id",
    "partido_nombre",
    "partido_alias",
    "voto_binario",
    "voto_ternario",
    "voto_nominal",
    "observado",
)

COLUMNAS_AFILIACION = (
    "diputado_id",
    "votacion_id",
    "fecha",
    "partido_id",
    "partido_nombre",
    "partido_alias",
)


class ErrorMatrices(ValueError):
    """La tabla no permite construir matrices sin violar el contrato."""


@dataclass(frozen=True, slots=True)
class ResultadoMatrices:
    """Productos tabulares derivados de los votos codificados."""

    binaria: pd.DataFrame
    ternaria: pd.DataFrame
    nominal: pd.DataFrame
    mascara: pd.DataFrame
    afiliacion: pd.DataFrame

    @property
    def n_diputados(self) -> int:
        return int(len(self.mascara))

    @property
    def n_votaciones(self) -> int:
        return int(len(self.mascara.columns) - 1)

    @property
    def n_decisiones_observadas(self) -> int:
        columnas_voto = [c for c in self.mascara.columns if c != "diputado_id"]
        return int(self.mascara[columnas_voto].sum().sum())

    @property
    def n_celdas_sin_registro(self) -> int:
        total = self.n_diputados * self.n_votaciones
        return int(total - self.n_decisiones_observadas)


def _raiz_repositorio() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolver_desde_raiz(raiz: Path, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else raiz / ruta


class ConstructorMatrices:
    """Construye matrices y extrae afiliación sin recodificar ni imputar."""

    def validar_entrada(self, votos: pd.DataFrame) -> None:
        if not isinstance(votos, pd.DataFrame):
            raise TypeError("votos debe ser un pandas.DataFrame.")

        faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in votos.columns]
        if faltantes:
            raise ErrorMatrices(f"Faltan columnas requeridas: {faltantes}.")

        if votos.empty:
            raise ErrorMatrices("La tabla de votos codificados está vacía.")

        if votos[["diputado_id", "votacion_id"]].isna().any(axis=None):
            raise ErrorMatrices(
                "diputado_id y votacion_id no pueden contener valores nulos."
            )

        duplicadas = votos.duplicated(["diputado_id", "votacion_id"], keep=False)
        if duplicadas.any():
            ejemplos = (
                votos.loc[duplicadas, ["diputado_id", "votacion_id"]]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise ErrorMatrices(
                "La clave diputado_id × votacion_id debe ser única. "
                f"Ejemplos duplicados: {ejemplos}."
            )

        observado = votos["observado"]
        if observado.isna().any():
            raise ErrorMatrices("observado no puede contener NA.")

        if not observado.dropna().isin([True, False]).all():
            raise ErrorMatrices("observado debe contener valores booleanos.")

        abstenciones = votos["voto_nominal"].eq("Abstención")
        if abstenciones.any():
            if votos.loc[abstenciones, "voto_ternario"].ne(0).any():
                raise ErrorMatrices(
                    "Una abstención observada debe conservar valor ternario 0."
                )
            if votos.loc[abstenciones, "voto_binario"].notna().any():
                raise ErrorMatrices(
                    "Una abstención no debe transformarse en voto binario."
                )

        no_observadas = ~observado.astype(bool)
        if no_observadas.any():
            columnas_vistas = ["voto_binario", "voto_ternario", "voto_nominal"]
            if votos.loc[no_observadas, columnas_vistas].notna().any(axis=None):
                raise ErrorMatrices(
                    "Las decisiones no observadas deben permanecer como NA."
                )

    def construir(self, votos: pd.DataFrame) -> ResultadoMatrices:
        self.validar_entrada(votos)

        diputados = self._orden_diputados(votos)
        votaciones = self._orden_votaciones(votos)

        binaria = self._pivotear(votos, "voto_binario", diputados, votaciones)
        ternaria = self._pivotear(votos, "voto_ternario", diputados, votaciones)
        nominal = self._pivotear(votos, "voto_nominal", diputados, votaciones)
        mascara = self._construir_mascara(votos, diputados, votaciones)
        afiliacion = self._extraer_afiliacion(votos)

        for columna in binaria.columns[1:]:
            binaria[columna] = binaria[columna].astype("Int8")

        for columna in ternaria.columns[1:]:
            ternaria[columna] = ternaria[columna].astype("Int8")

        for columna in mascara.columns[1:]:
            mascara[columna] = mascara[columna].astype("boolean")

        self._validar_resultados(
            votos=votos,
            binaria=binaria,
            ternaria=ternaria,
            nominal=nominal,
            mascara=mascara,
            afiliacion=afiliacion,
        )

        return ResultadoMatrices(
            binaria=binaria,
            ternaria=ternaria,
            nominal=nominal,
            mascara=mascara,
            afiliacion=afiliacion,
        )

    @staticmethod
    def _orden_diputados(votos: pd.DataFrame) -> list[Any]:
        return sorted(votos["diputado_id"].dropna().unique().tolist())

    @staticmethod
    def _orden_votaciones(votos: pd.DataFrame) -> list[Any]:
        orden = (
            votos[["votacion_id", "fecha"]]
            .drop_duplicates()
            .assign(fecha_orden=lambda d: pd.to_datetime(d["fecha"], errors="coerce"))
            .sort_values(["fecha_orden", "votacion_id"], kind="stable")
        )
        return orden["votacion_id"].tolist()

    @staticmethod
    def _pivotear(
        votos: pd.DataFrame,
        columna_valor: str,
        diputados: list[Any],
        votaciones: list[Any],
    ) -> pd.DataFrame:
        matriz = votos.pivot(
            index="diputado_id",
            columns="votacion_id",
            values=columna_valor,
        )
        matriz = matriz.reindex(index=diputados, columns=votaciones)
        matriz.columns.name = None
        matriz.index.name = "diputado_id"
        return matriz.reset_index()

    @staticmethod
    def _construir_mascara(
        votos: pd.DataFrame,
        diputados: list[Any],
        votaciones: list[Any],
    ) -> pd.DataFrame:
        mascara = votos.pivot(
            index="diputado_id",
            columns="votacion_id",
            values="observado",
        )
        mascara = mascara.reindex(index=diputados, columns=votaciones)
        mascara = mascara.astype("boolean").fillna(False)
        mascara.columns.name = None
        mascara.index.name = "diputado_id"
        return mascara.reset_index()

    @staticmethod
    def _extraer_afiliacion(votos: pd.DataFrame) -> pd.DataFrame:
        """Extrae la afiliación histórica ya resuelta en F3.

        Solo conserva filas que existen en la tabla de votos. No completa el
        producto cartesiano diputado × votación y no infiere afiliaciones para
        combinaciones sin registro.
        """
        afiliacion = votos.loc[:, COLUMNAS_AFILIACION].copy()

        afiliacion = afiliacion.sort_values(
            ["fecha", "votacion_id", "diputado_id"],
            kind="stable",
        ).reset_index(drop=True)

        return afiliacion

    @staticmethod
    def _validar_resultados(
        *,
        votos: pd.DataFrame,
        binaria: pd.DataFrame,
        ternaria: pd.DataFrame,
        nominal: pd.DataFrame,
        mascara: pd.DataFrame,
        afiliacion: pd.DataFrame,
    ) -> None:
        columnas_matriz = [c for c in mascara.columns if c != "diputado_id"]

        n_observadas = int(mascara[columnas_matriz].sum().sum())
        n_esperadas = int(votos["observado"].astype(bool).sum())

        if n_observadas != n_esperadas:
            raise ErrorMatrices(
                f"La máscara contiene {n_observadas} observaciones, "
                f"pero la tabla contiene {n_esperadas}."
            )

        if int(ternaria[columnas_matriz].notna().sum().sum()) != n_esperadas:
            raise ErrorMatrices(
                "La matriz ternaria no conserva todas las decisiones observadas."
            )

        if int(nominal[columnas_matriz].notna().sum().sum()) != n_esperadas:
            raise ErrorMatrices(
                "La matriz nominal no conserva todas las decisiones observadas."
            )

        abstenciones = int(votos["voto_nominal"].eq("Abstención").sum())
        esperados_binaria = n_esperadas - abstenciones
        reales_binaria = int(binaria[columnas_matriz].notna().sum().sum())

        if reales_binaria != esperados_binaria:
            raise ErrorMatrices(
                "La matriz binaria no conserva el universo Sí/No esperado."
            )

        no_observada = ~mascara.set_index("diputado_id")[columnas_matriz].astype(bool)
        bin_idx = binaria.set_index("diputado_id")[columnas_matriz]
        ter_idx = ternaria.set_index("diputado_id")[columnas_matriz]
        nom_idx = nominal.set_index("diputado_id")[columnas_matriz]

        if bin_idx.mask(~no_observada).notna().any(axis=None):
            raise ErrorMatrices("Hay valores binarios en celdas no observadas.")
        if ter_idx.mask(~no_observada).notna().any(axis=None):
            raise ErrorMatrices("Hay valores ternarios en celdas no observadas.")
        if nom_idx.mask(~no_observada).notna().any(axis=None):
            raise ErrorMatrices("Hay valores nominales en celdas no observadas.")

        if len(afiliacion) != len(votos):
            raise ErrorMatrices(
                "La tabla de afiliación debe conservar exactamente las filas observadas."
            )

        if afiliacion.duplicated(["diputado_id", "votacion_id"]).any():
            raise ErrorMatrices(
                "La tabla de afiliación contiene claves diputado_id × votacion_id duplicadas."
            )


def validar_contra_auditoria(
    votos: pd.DataFrame,
    ruta_auditoria: str | Path,
) -> None:
    """Concilia los conteos nominales con la auditoría de entrada."""
    ruta = Path(ruta_auditoria)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe la auditoría de entrada: {ruta}")

    with ruta.open(encoding="utf-8") as archivo:
        auditoria = json.load(archivo)

    esperados = auditoria.get("perfil", {}).get("votos_por_categoria")
    if not isinstance(esperados, dict):
        raise ErrorMatrices(
            "La auditoría no contiene perfil.votos_por_categoria."
        )

    traduccion = {
        "Sí": "Afirmativo",
        "No": "En Contra",
        "Abstención": "Abstención",
    }

    observados = (
        votos["voto_nominal"]
        .value_counts(dropna=False)
        .rename(index=traduccion)
        .to_dict()
    )

    diferencias = {
        categoria: {
            "auditoria": int(esperado),
            "codificados": int(observados.get(categoria, 0)),
        }
        for categoria, esperado in esperados.items()
        if int(observados.get(categoria, 0)) != int(esperado)
    }

    if diferencias:
        raise ErrorMatrices(
            "Los conteos de votos no coinciden con la auditoría: "
            f"{diferencias}."
        )


def guardar_resultados(
    resultado: ResultadoMatrices,
    directorio: str | Path,
) -> None:
    """Exporta matrices, máscara y afiliación histórica en CSV."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    resultado.binaria.to_csv(directorio / "matriz_binaria.csv", index=False, lineterminator="\n")
    resultado.ternaria.to_csv(directorio / "matriz_ternaria.csv", index=False, lineterminator="\n")
    resultado.nominal.to_csv(directorio / "matriz_nominal.csv", index=False, lineterminator="\n")
    resultado.mascara.to_csv(
        directorio / "mascara_observacion.csv",
        index=False,
        lineterminator="\n",
    )
    resultado.afiliacion.to_csv(
        directorio / "afiliacion_por_votacion.csv",
        index=False,
        lineterminator="\n",
    )


def ejecutar(
    *,
    ruta_entrada: str | Path = RUTA_ENTRADA_POR_DEFECTO,
    ruta_auditoria: str | Path = RUTA_AUDITORIA_POR_DEFECTO,
    directorio_salida: str | Path = DIRECTORIO_SALIDA_POR_DEFECTO,
    raiz: Path | None = None,
) -> ResultadoMatrices:
    raiz = (raiz or _raiz_repositorio()).resolve()

    entrada = _resolver_desde_raiz(raiz, ruta_entrada)
    auditoria = _resolver_desde_raiz(raiz, ruta_auditoria)
    salida = _resolver_desde_raiz(raiz, directorio_salida)

    if not entrada.exists():
        raise FileNotFoundError(f"No existe la tabla codificada: {entrada}")

    votos = pd.read_csv(entrada)
    validar_contra_auditoria(votos, auditoria)

    resultado = ConstructorMatrices().construir(votos)
    guardar_resultados(resultado, salida)

    return resultado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye matrices diputado × votación desde los votos codificados."
    )
    parser.add_argument(
        "--entrada",
        default=str(RUTA_ENTRADA_POR_DEFECTO),
        help="CSV de votos codificados relativo a la raíz del repositorio.",
    )
    parser.add_argument(
        "--auditoria",
        default=str(RUTA_AUDITORIA_POR_DEFECTO),
        help="JSON de auditoría usado para conciliar conteos.",
    )
    parser.add_argument(
        "--salida",
        default=str(DIRECTORIO_SALIDA_POR_DEFECTO),
        help="Directorio de salida relativo a la raíz del repositorio.",
    )
    return parser


def main() -> int:
    args = _crear_parser().parse_args()

    try:
        resultado = ejecutar(
            ruta_entrada=args.entrada,
            ruta_auditoria=args.auditoria,
            directorio_salida=args.salida,
        )
    except (ErrorMatrices, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Matrices generadas: "
        f"{resultado.n_diputados} diputados, "
        f"{resultado.n_votaciones} votaciones, "
        f"{resultado.n_decisiones_observadas} decisiones observadas, "
        f"{resultado.n_celdas_sin_registro} celdas sin registro."
    )
    print(
        "Afiliaciones extraídas: "
        f"{len(resultado.afiliacion)} registros observados."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())