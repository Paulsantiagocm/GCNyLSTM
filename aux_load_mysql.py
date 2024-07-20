import pandas as pd
import mysql.connector
from mysql.connector import Error
import re
import os

def clean_column_name(column_name):
    # Reemplazar caracteres no permitidos en nombres de columnas por un guion bajo
    return re.sub(r'\W+', '_', column_name)

def load_csv_to_mysql(csv_file_path, host, user, password, database, table_name):
    connection = None
    try:
        # Conectar al servidor MySQL (sin especificar la base de datos)
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Crear la base de datos si no existe
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            connection.commit()

        # Reconectar especificando la base de datos
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Leer el archivo CSV en un DataFrame de pandas
            data = pd.read_csv(csv_file_path)

            # Limpiar los nombres de las columnas
            data.columns = [clean_column_name(col) for col in data.columns]

            # Crear la tabla si no existe
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join([f'`{col}` VARCHAR(255)' for col in data.columns])}
            );
            """
            cursor.execute(create_table_query)
            connection.commit()

            # Reemplazar NaN por None
            data = data.where(pd.notnull(data), None)

            # Insertar los datos del DataFrame en la tabla MySQL
            for _, row in data.iterrows():
                # Convertir fila a una lista y manejar None para MySQL
                row_values = [None if pd.isna(val) else val for val in row]
                insert_row_query = f"""
                INSERT INTO {table_name} ({', '.join([f'`{col}`' for col in data.columns])})
                VALUES ({', '.join(['%s'] * len(row))});
                """
                cursor.execute(insert_row_query, tuple(row_values))
            
            connection.commit()
            print(f"Datos cargados exitosamente en la tabla {table_name}")

    except Error as e:
        print(f"Error al conectar a MySQL: {e}")
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("Conexión a MySQL cerrada")

# Configuración
host = 'localhost'  # Reemplaza con tu host de MySQL
user = 'root'  # Reemplaza con tu usuario de MySQL
password = '12345678'  # Reemplaza con tu contraseña de MySQL

# Llamada a la función para cada archivo CSV
csv_files = {
    'data_stations': '/teamspace/studios/this_studio/data/Stations.csv',
    'data_flow': '/teamspace/studios/this_studio/data/Flow.csv',
    'data_occ': '/teamspace/studios/this_studio/data/Occ.csv',
    'data_speed': '/teamspace/studios/this_studio/data/Speed.csv'
}

for table_name, csv_file_path in csv_files.items():
    load_csv_to_mysql(csv_file_path, host, user, password, table_name, table_name)
