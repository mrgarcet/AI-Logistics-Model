#----------------------------------------------------
#           Path: src/gnn/train_forecast.py
#----------------------------------------------------
'''
Train a simple temporal GNN to forecast next‑period demand.
Reads : models/gnn_dataset.pt   (created by preprocess.py)
Saves : models/gnn_forecaster/
Run   : python -m src.gnn.train_forecast
'''

#----------------------------------------------------
#       Import libraries (torch, PyG, helpers)
#----------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.serialization as ts
from pathlib import Path

#----------------------------------------------------
#              Hyper‑parameters & paths
#----------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
DATA_PATH  = ROOT / "models" / "gnn_dataset.pt"
MODEL_DIR  = ROOT / "models" / "gnn_forecaster"
HIDDEN     = 32
EPOCHS     = 30
LR         = 1e-3                                       # learning‑rate 0.001

#----------------------------------------------------
#                Load dataset from disk
#----------------------------------------------------
data: Data = torch.load(DATA_PATH, weights_only=False)  # allow custom PyG classes

x_tr = data.x                                           # [N, T_train]
y_tr = data.y                                           # [N, T_val]
edge_index = data.edge_index

num_nodes, train_len = x_tr.shape
val_len = y_tr.shape[1]

#----------------------------------------------------
#         Model definition  (GCN + GRU + Linear)
#----------------------------------------------------
class GCN_GRU(nn.Module):
    def __init__(self, t_in: int, hidden: int):
        super().__init__()
        self.gcn = GCNConv(t_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)                 # 1‑day‑ahead forecast

    def forward(self, x_seq, edge_idx):
        g = self.gcn(x_seq, edge_idx)                   # [N, hidden]
        g = g.unsqueeze(1)                              # [N, 1, hidden]
        out, _ = self.gru(g)                            # [N, 1, hidden]
        return self.fc(out.squeeze(1)).squeeze()        # [N]

model = GCN_GRU(train_len, HIDDEN)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()                                  # MAE ≈ SMAPE scale‑free

#----------------------------------------------------
#          Training loop (recursive 1‑step forecast)
#----------------------------------------------------
model.train()
for epoch in range(1, EPOCHS + 1):
    optim.zero_grad()
    pred = model(x_tr, edge_index)
    loss = loss_fn(pred, y_tr[:, 0])                   # predict first step in val window
    loss.backward()
    optim.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch:02d}/{EPOCHS}  train‑MAE = {loss.item():.4f}")

#----------------------------------------------------
#                    Save artefacts
#----------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "weights.pt")
torch.save(edge_index,             MODEL_DIR / "edge_index.pt")
torch.save(x_tr,                   MODEL_DIR / "train_window.pt")
print(f"✅  Saved model to {MODEL_DIR}")

#----------------------------------------------------
#                 End of train_forecast.py
#----------------------------------------------------

## Instructions ##
# 1. Activate your venv, cd to project root.
# 2. Run:  python -m src.gnn.train_forecast
# 3. Expect ~30 lines of epoch logs and a success message.
