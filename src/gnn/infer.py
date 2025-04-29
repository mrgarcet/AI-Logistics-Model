#----------------------------------------------------
#              Path: src/gnn/infer.py
#----------------------------------------------------
'''
Inference helper for demand forecasting.

Usage:
    from src.gnn.infer import forecast
    demand = forecast(item_id=1001, horizon=30)
'''

import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from pathlib import Path

#----------------------------------------------------
#             Load saved model & artefacts
#----------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
MODEL_DIR  = ROOT / "models" / "gnn_forecaster"

edge_index = torch.load(MODEL_DIR / "edge_index.pt")
train_x    = torch.load(MODEL_DIR / "train_window.pt")   # [N, T]
num_nodes, t_len = train_x.shape

HIDDEN = 32

class GCN_GRU(nn.Module):
    def __init__(self, t_in, hidden):
        super().__init__()
        self.gcn = GCNConv(t_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x_seq, edge_idx):
        g = self.gcn(x_seq, edge_idx)
        g = g.unsqueeze(1)
        out, _ = self.gru(g)
        return self.fc(out.squeeze(1)).squeeze()

model = GCN_GRU(t_len, HIDDEN)
model.load_state_dict(torch.load(MODEL_DIR / "weights.pt"))
model.eval()

#----------------------------------------------------
#            Public function: forecast()
#----------------------------------------------------
def forecast(item_id: int, horizon: int = 30) -> float:
    """
    Returns cumulative demand forecast for <item_id> over <horizon> days.
    Very simple: multiplies 1‑day prediction by horizon.
    """
    try:
        row = int(item_id) - 1                   # our demo maps item_id N → row N‑1
        one_day = float(model(train_x, edge_index)[row])
        return max(one_day, 0) * horizon
    except IndexError:
        raise ValueError(f"item_id {item_id} is out of range 1..{num_nodes}")

# quick CLI test
if __name__ == "__main__":
    for i in (1, 10, 25):
        print(f"30‑day forecast for item {i:02d}: {forecast(i):.1f}")
