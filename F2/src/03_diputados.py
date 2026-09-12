"""
Extrae el XML crudo (envelope SOAP) de retornarDiputadosXPeriodo
para un período legislativo y lo guarda en data/raw/diputados/.

TODO: por ahora periodo_id queda fijo en 10. Reemplazar más adelante
por la lectura del CSV que generará el script de proyecto legislativo.
"""

from pathlib import Path
from zeep import Client
from zeep.plugins import HistoryPlugin
from lxml import etree

# Configuración para extraer el XML crudo de la llamada a retornarDiputadosXPeriodo
WSDL_DIPUTADO = "https://opendata.camara.cl/camaradiputados/WServices/WSDiputado.asmx?WSDL"
RAW_DIR = Path("data/raw/diputados")
periodo_id = 10

# Conexión y llamada al servicio web
history = HistoryPlugin()
client_diputado = Client(WSDL_DIPUTADO, plugins=[history])

resultado = client_diputado.service.retornarDiputadosXPeriodo(prmPeriodoID=str(periodo_id))

# Extraer el envelope SOAP crudo
ultimo = history.last_received
envelope = ultimo["envelope"]

xml_crudo = etree.tostring(
    envelope, pretty_print=True, xml_declaration=True, encoding="UTF-8"
).decode("utf-8")

# Guardar el XML crudo en un archivo

RAW_DIR.mkdir(parents=True, exist_ok=True)
ruta_archivo = RAW_DIR / f"diputados_periodo_{periodo_id}.xml"
ruta_archivo.write_text(xml_crudo, encoding="utf-8")

print(f"Guardado en: {ruta_archivo.resolve()}")