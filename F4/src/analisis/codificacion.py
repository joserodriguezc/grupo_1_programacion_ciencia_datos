from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


RUTA_ENTRADA_POR_DEFECTO = Path("F3/data/processed/big_table_analitica.csv")
RUTA_DICCIONARIO_POR_DEFECTO = Path("F4/config/diccionario_votos.toml")
RUTA_SALIDA_POR_DEFECTO = Path("F4/data/processed/votos_codificados.csv")

COLUMNAS_CLAVE = ("diputado_id", "votacion_id")
COLUMNAS_SALIDA = (
    "voto_binario",
    "voto_ternario",
    "voto_nominal",
    "observado",
    "estado_observacion",
)


class ErrorCodificacion(ValueError):
    """La tabla no puede codificarse sin violar las reglas declaradas."""


class ErrorDiccionarioVotos(ErrorCodificacion):
    """El archivo de configuración de votos es inválido o incompleto."""


@dataclass(frozen=True, slots=True)
class ReglaVoto:
    """Representaciones analíticas asociadas a un código observado."""

    codigo: str
    etiqueta_fuente: str
    binario: int | None
    ternario: int
    nominal: str


@dataclass(frozen=True, slots=True)
class DiccionarioVotos:
    """Configuración validada de códigos y nombres de columnas de origen."""

    version: str
    columna_codigo: str
    columna_texto: str
    reglas: Mapping[str, ReglaVoto]
    etiqueta_no_observado: str = "desconocido"


def _raiz_repositorio() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolver_desde_raiz(raiz: Path, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else raiz / ruta


def _normalizar_codigo(valor: Any) -> str | None:
    """Normaliza códigos de CSV sin convertir NA en una categoría válida."""
    if pd.isna(valor):
        return None

    if isinstance(valor, bool):
        return str(valor)

    if isinstance(valor, int):
        return str(valor)

    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))

    texto = str(valor).strip()
    if not texto:
        return None

    try:
        numero = float(texto)
    except ValueError:
        return texto

    return str(int(numero)) if numero.is_integer() else texto


def cargar_diccionario(ruta: str | Path) -> DiccionarioVotos:
    """Carga y valida el diccionario TOML de códigos del voto."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el diccionario de votos: {ruta}")

    with ruta.open("rb") as archivo:
        configuracion = tomllib.load(archivo)

    metadatos = configuracion.get("metadatos", {})
    codigos = configuracion.get("codigos", {})
    no_observado = configuracion.get("no_observado", {})

    version = str(metadatos.get("version", "")).strip()
    columna_codigo = str(metadatos.get("columna_codigo", "")).strip()
    columna_texto = str(metadatos.get("columna_texto", "")).strip()

    if not version or not columna_codigo or not columna_texto:
        raise ErrorDiccionarioVotos(
            "El diccionario requiere version, columna_codigo y columna_texto."
        )
    if not isinstance(codigos, dict) or not codigos:
        raise ErrorDiccionarioVotos("El diccionario no contiene códigos de voto.")

    reglas: dict[str, ReglaVoto] = {}
    for codigo, definicion in codigos.items():
        if not isinstance(definicion, dict):
            raise ErrorDiccionarioVotos(f"Definición inválida para código {codigo!r}.")

        codigo_normalizado = _normalizar_codigo(codigo)
        etiqueta = str(definicion.get("etiqueta_fuente", "")).strip()
        nominal = str(definicion.get("nominal", "")).strip()
        ternario = definicion.get("ternario")
        binario = definicion.get("binario")

        if codigo_normalizado is None or not etiqueta or not nominal:
            raise ErrorDiccionarioVotos(f"Definición incompleta para código {codigo!r}.")
        if ternario not in (-1, 0, 1):
            raise ErrorDiccionarioVotos(
                f"ternario del código {codigo!r} debe pertenecer a -1, 0, 1."
            )
        if binario is not None and binario not in (0, 1):
            raise ErrorDiccionarioVotos(
                f"binario del código {codigo!r} debe ser 0, 1 o estar ausente."
            )
        if codigo_normalizado in reglas:
            raise ErrorDiccionarioVotos(
                f"Código duplicado tras normalización: {codigo_normalizado!r}."
            )

        reglas[codigo_normalizado] = ReglaVoto(
            codigo=codigo_normalizado,
            etiqueta_fuente=etiqueta,
            binario=binario,
            ternario=int(ternario),
            nominal=nominal,
        )

    etiqueta_no_observado = str(
        no_observado.get("etiqueta", "desconocido")
    ).strip()
    if not etiqueta_no_observado:
        raise ErrorDiccionarioVotos("La etiqueta no observada no puede estar vacía.")

    return DiccionarioVotos(
        version=version,
        columna_codigo=columna_codigo,
        columna_texto=columna_texto,
        reglas=reglas,
        etiqueta_no_observado=etiqueta_no_observado,
    )


class CodificadorVotos:
    """Agrega tres vistas del voto sin alterar las columnas de origen."""

    def __init__(self, diccionario: DiccionarioVotos) -> None:
        self.diccionario = diccionario

    def transformar(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Codifica una tabla larga y devuelve una copia con columnas derivadas."""
        if not isinstance(tabla, pd.DataFrame):
            raise TypeError("tabla debe ser un pandas.DataFrame.")

        requeridas = {
            *COLUMNAS_CLAVE,
            self.diccionario.columna_codigo,
            self.diccionario.columna_texto,
        }
        faltantes = sorted(requeridas.difference(tabla.columns))
        if faltantes:
            raise ErrorCodificacion(f"Faltan columnas obligatorias: {faltantes}.")

        claves_nulas = tabla[list(COLUMNAS_CLAVE)].isna().any(axis=1)
        if claves_nulas.any():
            raise ErrorCodificacion(
                f"Hay {int(claves_nulas.sum())} filas con clave diputado-votación nula."
            )

        duplicadas = tabla.duplicated(list(COLUMNAS_CLAVE), keep=False)
        if duplicadas.any():
            ejemplos = (
                tabla.loc[duplicadas, list(COLUMNAS_CLAVE)]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise ErrorCodificacion(
                "La clave diputado-votación debe ser única. "
                f"Ejemplos duplicados: {ejemplos}."
            )

        salida = tabla.copy()
        codigos = salida[self.diccionario.columna_codigo].map(_normalizar_codigo)

        codigos_no_nulos = set(codigos.dropna().unique())
        desconocidos = sorted(codigos_no_nulos.difference(self.diccionario.reglas))
        if desconocidos:
            raise ErrorCodificacion(
                "Hay códigos de voto no declarados en el diccionario: "
                f"{desconocidos}."
            )

        self._validar_etiquetas_fuente(salida, codigos)

        mapa_binario = {
            codigo: regla.binario for codigo, regla in self.diccionario.reglas.items()
        }
        mapa_ternario = {
            codigo: regla.ternario for codigo, regla in self.diccionario.reglas.items()
        }
        mapa_nominal = {
            codigo: regla.nominal for codigo, regla in self.diccionario.reglas.items()
        }

        salida["voto_binario"] = codigos.map(mapa_binario).astype("Int64")
        salida["voto_ternario"] = codigos.map(mapa_ternario).astype("Int64")
        salida["voto_nominal"] = codigos.map(mapa_nominal).astype("string")

        observados = codigos.isin(self.diccionario.reglas)
        salida["observado"] = observados.astype("boolean")
        salida["estado_observacion"] = pd.Series(
            pd.NA,
            index=salida.index,
            dtype="string",
        )
        salida.loc[observados, "estado_observacion"] = "observado"
        salida.loc[~observados, "estado_observacion"] = (
            self.diccionario.etiqueta_no_observado
        )

        self._validar_invariantes(salida)
        return salida

    def _validar_etiquetas_fuente(
        self,
        tabla: pd.DataFrame,
        codigos: pd.Series,
    ) -> None:
        """Impide que un código conocido cambie silenciosamente de significado."""
        texto = tabla[self.diccionario.columna_texto].astype("string").str.strip()
        esperadas = codigos.map(
            {codigo: regla.etiqueta_fuente for codigo, regla in self.diccionario.reglas.items()}
        ).astype("string")

        observados = codigos.isin(self.diccionario.reglas)
        inconsistentes = observados & (texto.isna() | texto.ne(esperadas))
        if inconsistentes.any():
            columnas = [
                *COLUMNAS_CLAVE,
                self.diccionario.columna_codigo,
                self.diccionario.columna_texto,
            ]
            ejemplos = tabla.loc[inconsistentes, columnas].head(5).to_dict("records")
            raise ErrorCodificacion(
                "El texto del voto no coincide con el significado declarado para su código. "
                f"Ejemplos: {ejemplos}."
            )

    @staticmethod
    def _validar_invariantes(tabla: pd.DataFrame) -> None:
        """Verifica las reglas que no deben romper los métodos posteriores."""
        no_observados = ~tabla["observado"].fillna(False)
        if tabla.loc[no_observados, "voto_ternario"].notna().any():
            raise ErrorCodificacion("Un voto no observado recibió valor ternario.")
        if tabla.loc[no_observados, "voto_binario"].notna().any():
            raise ErrorCodificacion("Un voto no observado recibió valor binario.")
        if tabla.loc[no_observados, "voto_nominal"].notna().any():
            raise ErrorCodificacion("Un voto no observado recibió valor nominal.")

        abstenciones = tabla["voto_nominal"].eq("Abstención").fillna(False)
        if not tabla.loc[abstenciones, "observado"].all():
            raise ErrorCodificacion("Una abstención válida debe estar marcada como observada.")
        if not tabla.loc[abstenciones, "voto_ternario"].eq(0).all():
            raise ErrorCodificacion("Una abstención válida debe valer 0 en la vista ternaria.")
        if tabla.loc[abstenciones, "voto_binario"].notna().any():
            raise ErrorCodificacion("Una abstención no debe entrar en la vista binaria.")


def codificar_archivo(
    *,
    ruta_entrada: str | Path = RUTA_ENTRADA_POR_DEFECTO,
    ruta_diccionario: str | Path = RUTA_DICCIONARIO_POR_DEFECTO,
    ruta_salida: str | Path = RUTA_SALIDA_POR_DEFECTO,
    raiz: Path | None = None,
) -> pd.DataFrame:
    """Carga la entrada, aplica el diccionario y exporta la tabla codificada."""
    raiz = (raiz or _raiz_repositorio()).resolve()
    entrada = _resolver_desde_raiz(raiz, ruta_entrada)
    diccionario = _resolver_desde_raiz(raiz, ruta_diccionario)
    salida = _resolver_desde_raiz(raiz, ruta_salida)

    if not entrada.exists():
        raise FileNotFoundError(f"No existe la entrada: {entrada}")

    tabla = pd.read_csv(entrada)
    codificador = CodificadorVotos(cargar_diccionario(diccionario))
    resultado = codificador.transformar(tabla)

    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False)
    return resultado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera las vistas binaria, ternaria y nominal del voto."
    )
    parser.add_argument("--entrada", default=str(RUTA_ENTRADA_POR_DEFECTO))
    parser.add_argument("--diccionario", default=str(RUTA_DICCIONARIO_POR_DEFECTO))
    parser.add_argument("--salida", default=str(RUTA_SALIDA_POR_DEFECTO))
    return parser


def main() -> int:
    args = _crear_parser().parse_args()
    try:
        resultado = codificar_archivo(
            ruta_entrada=args.entrada,
            ruta_diccionario=args.diccionario,
            ruta_salida=args.salida,
        )
    except (ErrorCodificacion, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    conteos = resultado["voto_nominal"].value_counts(dropna=False).to_dict()
    print(f"Codificación completada: {len(resultado)} filas.")
    print(f"Conteos nominales: {conteos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
