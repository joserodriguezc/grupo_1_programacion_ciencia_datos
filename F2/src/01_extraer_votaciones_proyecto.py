from pathlib import Path
import requests
import pandas as pd

#URL de camara de diputados y diputadas (transparencia datos abiertos)
BASE_URL = (
    "https://opendata.camara.cl/"
    "camaradiputados/WServices/WSLegislativo.asmx/"
    "retornarVotacionesXProyectoLey"
)

# Direccion de destino de los archivos generados
OUTPUT_DIR = Path("data")
XML_FILE = OUTPUT_DIR / "raw" / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"
CSV_FILE = OUTPUT_DIR / "interim" / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"

"11092-07:  este es el numero de boletin a escribir en la funcion de extraer_votaciones"
def extraer_votaciones(numero_boletin: str) -> pd.DataFrame:
    XML_FILE.parent.mkdir(parents=True, exist_ok=True)
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}?prmNumeroBoletin={numero_boletin}"

    print(f"Consultando boletín: {numero_boletin}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Guardar XML crudo
    XML_FILE.write_bytes(response.content)

    print(f"XML guardado en: {XML_FILE}")

    # Leer XML con Pandas
    df = pd.read_xml(XML_FILE, xpath=".//*[local-name()='VotacionProyectoLey']")

    if df.empty:
        print("No se encontraron votaciones.")
        return df

    df.insert(0, "numero_boletin", numero_boletin)

    # Guardar CSV
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

    print(f"CSV guardado en: {CSV_FILE}")
    print(f"Registros extraídos: {len(df)}")

    return df