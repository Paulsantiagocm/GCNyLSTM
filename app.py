import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

# Definir el modelo GCN-LSTM mejorado
class GCN_LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(GCN_LSTM, self).__init__()
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lstm = torch.nn.LSTM(hidden_channels, out_channels, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data_list):
        x_seq = []
        for data in data_list:
            x, edge_index = data.x, data.edge_index
            for i, gcn in enumerate(self.gcn_layers):
                x = F.relu(gcn(x, edge_index))
                if i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = self.dropout(x)
            x_seq.append(x.unsqueeze(0))
        
        x_seq = torch.cat(x_seq, dim=0)
        x_seq = x_seq.transpose(0, 1)
        x, (h_n, c_n) = self.lstm(x_seq)
        return x[:, -1, :]

# Cargar los tensores y el modelo
x_tensor_path = 'x_tensor.pt'  # Asegúrate de que esta ruta sea correcta
edge_index_tensor_path = 'edge_index_tensor.pt'  # Asegúrate de que esta ruta sea correcta
model_path = 'gcn_lstm_model.pth'  # Asegúrate de que esta ruta sea correcta

x = torch.load(x_tensor_path)
edge_index = torch.load(edge_index_tensor_path)

# Verificar los datos cargados
print(f"x shape: {x.shape}")
print(f"edge_index shape: {edge_index.shape}")

# Verificar si el tensor x tiene NaN o valores infinitos
nan_in_x = torch.isnan(x).any()
inf_in_x = torch.isinf(x).any()
print(f"NaN in x: {nan_in_x}, Inf in x: {inf_in_x}")

# Calcular la media de la columna ignorando los valores NaN
mean_values = torch.nanmean(x, dim=(0, 1), keepdim=True)

# Reemplazar los valores NaN con la media de la columna correspondiente
for i in range(x.size(2)):  # Iterar sobre cada característica
    nan_mask = torch.isnan(x[:, :, i])
    x[:, :, i][nan_mask] = mean_values[:, :, i]

# Verificar de nuevo para asegurarnos de que no hay valores NaN
nan_in_x_after_replacement = torch.isnan(x).any()
print(f"NaN in x after replacement: {nan_in_x_after_replacement}")

# Normalizar el tensor x
mean = x.mean(dim=(0, 1), keepdim=True)
std = x.std(dim=(0, 1), keepdim=True)
x_normalized = (x - mean) / (std + 1e-6)  # Añadir un pequeño valor para evitar la división por cero

# Verificar el tensor normalizado
print(f"x_normalized shape: {x_normalized.shape}")
print(f"mean: {mean}")
print(f"std: {std}")

# Crear una lista de Data para PyTorch Geometric
data_list = [Data(x=x_normalized[:, t, :], edge_index=edge_index) for t in range(x_normalized.size(1))]

# Verificar la lista de datos
print(f"data_list length: {len(data_list)}")
print(f"data_list[0] x shape: {data_list[0].x.shape}")
print(f"data_list[0] edge_index shape: {data_list[0].edge_index.shape}")

# Cargar el modelo
model = GCN_LSTM(in_channels=x.size(2), hidden_channels=64, out_channels=x.size(2), num_layers=3, dropout=0.3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Verificar que el modelo esté cargado correctamente
print(model)

# Funciones para normalizar y desnormalizar los datos
def normalize(tensor, mean, std):
    return (tensor - mean) / (std + 1e-6)

def denormalize(tensor, mean, std):
    return tensor * std + mean

# Crear la interfaz de Streamlit
st.title("GCN-LSTM Prediction App")

# Botón para predecir
if st.button("Predict"):
    with torch.no_grad():
        out = model(data_list)
        
        # Verificar salida antes de desnormalizar
        st.write(f"Model output (raw): {out}")

        out_denormalized = denormalize(out, mean[:, -1, :], std[:, -1, :])
        target_denormalized = denormalize(x_normalized[:, -1, :], mean[:, -1, :], std[:, -1, :])

        st.write("Predicted (denormalized):")
        st.write(out_denormalized.numpy())
        st.write("Real (denormalized):")
        st.write(target_denormalized.numpy())

# Instrucciones para correr el archivo Streamlit usando: streamlit run app.py
