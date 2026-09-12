from pathlib import Path
import requests
import pandas as pd


BASE_URL = (
    "https://opendata.camara.cl/"
    "camaradiputados/WServices/WSLegislativo.asmx/"
    "retornarVotacionesXProyectoLey"
)

OUTPUT_DIR = Path("data")
XML_FILE = OUTPUT_DIR / "raw"/ "votaciones.xml"
CSV_FILE = OUTPUT_DIR / "interim" /"votaciones.csv"


def extraer_votaciones(numero_boletin: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}?prmNumeroBoletin={numero_boletin}"

    print(f"Consultando boletín: {numero_boletin}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # 1. Guardar XML crudo
    XML_FILE.write_bytes(response.content)

    print(f"XML guardado en: {XML_FILE}")

    # 2. Leer el XML guardado con Pandas
    df = pd.read_xml(
        XML_FILE,
        xpath=".//*[local-name()='VotacionProyectoLey']"
    )

    if df.empty:
        print("No se encontraron votaciones.")
        return df

    df.insert(0, "numero_boletin", numero_boletin)

    # 3. Guardar CSV
    df.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"CSV guardado en: {CSV_FILE}")
    print(f"Registros extraídos: {len(df)}")

    return df


def main():
    numero_boletin = "11092-07"

    df_votaciones = extraer_votaciones(numero_boletin)

    if not df_votaciones.empty:
        print("\nVista previa:")
        print(df_votaciones.head())


if __name__ == "__main__":
    main()