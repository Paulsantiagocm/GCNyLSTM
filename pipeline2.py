import pandas as pd
import numpy as np
from prefect import task, flow
from sqlalchemy import create_engine
import networkx as nx
import matplotlib.pyplot as plt
import torch  

# Parte 1: Extracción de datos

@task
def extraer_data_stations():
    # Crear la cadena de conexión utilizando SQLAlchemy
    engine = create_engine('mysql+pymysql://root:12345678@localhost/data_stations')
    
    # Realizar la consulta y leer los datos en un DataFrame de pandas
    consulta = "SELECT * FROM data_stations"
    df = pd.read_sql(consulta, engine)
    return df

@task
def extraer_data_flow():
    # Crear la cadena de conexión utilizando SQLAlchemy
    engine = create_engine('mysql+pymysql://root:12345678@localhost/data_flow')
    
    # Realizar la consulta y leer los datos en un DataFrame de pandas
    consulta = "SELECT * FROM data_flow"
    df = pd.read_sql(consulta, engine)
    return df

@task
def extraer_data_occ():
    # Crear la cadena de conexión utilizando SQLAlchemy
    engine = create_engine('mysql+pymysql://root:12345678@localhost/data_occ')
    
    # Realizar la consulta y leer los datos en un DataFrame de pandas
    consulta = "SELECT * FROM data_occ"
    df = pd.read_sql(consulta, engine)
    return df

@task
def extraer_data_speed():
    # Crear la cadena de conexión utilizando SQLAlchemy
    engine = create_engine('mysql+pymysql://root:12345678@localhost/data_speed')
    
    # Realizar la consulta y leer los datos en un DataFrame de pandas
    consulta = "SELECT * FROM data_speed"
    df = pd.read_sql(consulta, engine)
    return df

@task
def transformar_a_numeros(df):
    # Convertir las columnas a tipo numérico cuando sea posible
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

@task
def graficar_grafo(data_stations):
    # Cargar el archivo CSV
    data = data_stations
    # Crear un grafo
    G = nx.Graph()
    # Agregar nodos al grafo con las coordenadas lat y long
    for index, row in data.iterrows():
        G.add_node(index, pos=(row['Long'], row['Lat']))
    # Agregar aristas entre nodos consecutivos
    for i in range(len(data) - 1):
        G.add_edge(i, i+1)
     # Visualizar y guardar el grafo
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color="blue", edge_color="gray")
    plt.title("Grafo basado en Latitud y Longitud con Conexiones Consecutivas")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig('/teamspace/studios/this_studio/grafo_lat_long_con_conexiones.png')  # Guardar la imagen
    return 

@task
def features_nodes(data_flow, data_occ, data_speed):
    data_n1 = pd.DataFrame()
    data_n1['flow'] = data_flow.iloc[:, 1]
    data_n1['occ'] = data_occ.iloc[:, 1]
    data_n1['speed'] = data_speed.iloc[:, 1]

    data_n2 = pd.DataFrame()
    data_n2['flow'] = data_flow.iloc[:, 2]
    data_n2['occ'] = data_occ.iloc[:, 2]
    data_n2['speed'] = data_speed.iloc[:, 2]

    data_n3 = pd.DataFrame()
    data_n3['flow'] = data_flow.iloc[:, 3]
    data_n3['occ'] = data_occ.iloc[:, 3]
    data_n3['speed'] = data_speed.iloc[:, 3]
    return data_n1, data_n2, data_n3

@task
def transformar_a_tensor(data_n1, data_n2, data_n3):
    # Convert DataFrames to tensors
    tensor_n1 = torch.tensor(data_n1.values, dtype=torch.float32)
    tensor_n2 = torch.tensor(data_n2.values, dtype=torch.float32)
    tensor_n3 = torch.tensor(data_n3.values, dtype=torch.float32)
    # Create a list of tensors
    node_features = [tensor_n1, tensor_n2, tensor_n3]
    # Stack the tensors along a new dimension
    x = torch.stack(node_features, dim=0)
    return x

@task
def crear_edge_index(data_stations):
    # Crear un grafo
    G = nx.Graph()
    # Agregar nodos al grafo con las coordenadas lat y long
    for index, row in data_stations.head(3).iterrows():
        G.add_node(index, pos=(row['Long'], row['Lat']))
    # Agregar aristas entre los primeros tres nodos
    for i in range(2):
        G.add_edge(i, i+1)
    # Crear el edge_index
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    return edge_index

@task
def exportar_tensores(x, edge_index):
    torch.save(x, '/teamspace/studios/this_studio/x_tensor.pt')
    torch.save(edge_index, '/teamspace/studios/this_studio/edge_index_tensor.pt')

@flow(name="GCN+LSTM")
def flujo_completo():
    data_stations = extraer_data_stations()
    data_flow = extraer_data_flow()
    data_occ = extraer_data_occ()
    data_speed = extraer_data_speed()

    data_stations = transformar_a_numeros(data_stations)
    data_flow = transformar_a_numeros(data_flow)
    data_occ = transformar_a_numeros(data_occ)
    data_speed = transformar_a_numeros(data_speed)

    graficar_grafo(data_stations) 

    data_n1, data_n2, data_n3 = features_nodes(data_flow, data_occ, data_speed)
    
    x = transformar_a_tensor(data_n1, data_n2, data_n3)
   
    edge_index = crear_edge_index(data_stations)
    
    exportar_tensores(x, edge_index)
        
    return x, edge_index

if __name__ == "__main__":
    flujo_completo()
