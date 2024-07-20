import pandas as pd
import os

# Ruta al archivo .xls
file_path = '/teamspace/studios/this_studio/2000_sec_2023_12_listo.xls'  # Reemplaza con la ruta de tu archivo

# Crear la carpeta 'data' si no existe
output_dir = os.path.join(os.path.dirname(file_path), 'data')
os.makedirs(output_dir, exist_ok=True)

# Leer todas las hojas del archivo XLS en un diccionario de dataframes
xls = pd.ExcelFile(file_path)
sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

# Exportar cada hoja a un archivo CSV dentro de la carpeta 'data'
for sheet_name, df in sheets.items():
    csv_file_path = os.path.join(output_dir, f"{sheet_name}.csv")
    df.to_csv(csv_file_path, index=False)

# Mostrar los nombres de las hojas cargadas
print(sheets.keys())
