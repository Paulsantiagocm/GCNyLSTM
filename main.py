import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Cargar los tensores desde los archivos proporcionados
x_tensor_path = '/teamspace/studios/this_studio/x_tensor.pt'
edge_index_tensor_path = '/teamspace/studios/this_studio/edge_index_tensor.pt'

x = torch.load(x_tensor_path)
edge_index = torch.load(edge_index_tensor_path)

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

# Convertimos los datos para adaptarlos a la estructura esperada por PyTorch Geometric
# Reorganizamos x_normalized para que cada paso de tiempo sea una entrada diferente con su grafo correspondiente
data_list = []
for t in range(x_normalized.size(1)):
    data_list.append(Data(x=x_normalized[:, t, :], edge_index=edge_index))

# Visualizar el grafo
G = to_networkx(data_list[0])
nx.draw(G, with_labels=True)
plt.show()

# Definir el modelo GCN-LSTM mejorado
class GCN_LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(GCN_LSTM, self).__init__()
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms = torch.nn.ModuleList()  # Añadir la lista de BatchNorm
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))  # Añadir capa de BatchNorm
        self.lstm = torch.nn.LSTM(hidden_channels, out_channels, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data_list):
        x_seq = []
        for data in data_list:
            x, edge_index = data.x, data.edge_index
            for i, gcn in enumerate(self.gcn_layers):
                x = F.relu(gcn(x, edge_index))
                if i < len(self.batch_norms):
                    x = self.batch_norms[i](x)  # Aplicar BatchNorm solo a x
                x = self.dropout(x)
            x_seq.append(x.unsqueeze(0))
        
        x_seq = torch.cat(x_seq, dim=0)
        x_seq = x_seq.transpose(0, 1)
        x, (h_n, c_n) = self.lstm(x_seq)
        return x[:, -1, :]

# Definir el modelo, el optimizador y la función de pérdida
model = GCN_LSTM(in_channels=x_normalized.size(2), hidden_channels=64, out_channels=x_normalized.size(2), num_layers=3, dropout=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
criterion = torch.nn.MSELoss()

# Función para calcular la precisión
def accuracy(pred, y, threshold=0.1):
    correct = torch.abs(pred - y) < (threshold * y)
    return correct.sum().item() / torch.numel(correct)

# Crear un DataLoader
loader = DataLoader(data_list, batch_size=1, shuffle=True)

# Entrenar el modelo
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_list)
    target = x_normalized[:, -1, :]  # Usamos el último paso temporal como target
    loss = criterion(out, target)  # Comparamos la predicción con el último paso temporal

    # Verificar si el valor de pérdida es nan
    if torch.isnan(loss):
        print(f"NaN loss detected at epoch {epoch+1}. Exiting training loop.")
        break

    loss.backward()
    optimizer.step()
    scheduler.step()
    acc = accuracy(out, target)  # Calculamos la precisión
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {acc * 100:.2f}%')

# Evaluar el modelo
model.eval()
with torch.no_grad():
    out = model(data_list)
    target = x_normalized[:, -1, :]
    loss = criterion(out, target)
    acc = accuracy(out, target)
    print(f'Test Loss: {loss.item()}, Test Accuracy: {acc * 100:.2f}%')

    # Desnormalizar las predicciones
    out_denormalized = out * std[:, -1, :] + mean[:, -1, :]
    target_denormalized = target * std[:, -1, :] + mean[:, -1, :]

    print(f'Predicted (denormalized): {out_denormalized}')
    print(f'Real (denormalized): {target_denormalized}')


# Guardar el modelo entrenado
model_path = '/teamspace/studios/this_studio/gcn_lstm_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
