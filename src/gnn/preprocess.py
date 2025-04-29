#----------------------------------------------------
#            Path: src/gnn/preprocess.py
#----------------------------------------------------
'''
Builds a PyTorch‑Geometric dataset from:

  - data/inventory.csv   (item_id, name, barcode, current_stock)
  - data/sales.csv       (date, item_id, qty_sold)

Output: models/gnn_dataset.pt
Run   : python -m src.gnn.preprocess
'''

#----------------------------------------------------
#       Import libraries (pandas, PyTorch, PyG)
#----------------------------------------------------
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np

#----------------------------------------------------
#         Project paths and split parameters
#----------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]              # project root
INV_PATH   = ROOT / "data" / "inventory.csv"
SALES_PATH = ROOT / "data" / "sales.csv"
OUT_PATH   = ROOT / "models" / "gnn_dataset.pt"

SPLIT_FRAC = 0.8                                        # 80 % train, 20 % val

#----------------------------------------------------
# 1) Load inventory and build item_id → row‑index map
#----------------------------------------------------
inv_df = pd.read_csv(INV_PATH)
item_ids = inv_df["item_id"].tolist()
id2idx  = {item_id: i for i, item_id in enumerate(item_ids)}
num_items = len(item_ids)

#----------------------------------------------------
# 2) Load sales and build tensor  [items × days]
#----------------------------------------------------
sales_df = pd.read_csv(SALES_PATH, parse_dates=["date"])
sales_df["idx"] = sales_df["item_id"].map(id2idx)       # convert to tensor row

# Normalise date to integer day offset
min_day = sales_df["date"].min()
sales_df["t"] = (sales_df["date"] - min_day).dt.days
num_days = sales_df["t"].max() + 1

qty = torch.zeros((num_items, num_days), dtype=torch.float32)

for _, row in sales_df.iterrows():                      # scatter add qty
    qty[row["idx"], int(row["t"])] += row["qty_sold"]

#----------------------------------------------------
# 3) Split columns into train / validation windows
#----------------------------------------------------
split_col = int(num_days * SPLIT_FRAC)
qty_train = qty[:, :split_col]                          # feature window
qty_val   = qty[:, split_col:]                          # label window

#----------------------------------------------------
# 4) Build a simple fully connected undirected graph
#----------------------------------------------------
row_idx, col_idx = torch.triu_indices(num_items, num_items, offset=1)
edge_index = torch.stack([torch.cat([row_idx, col_idx]),
                          torch.cat([col_idx, row_idx])], dim=0)

#----------------------------------------------------
# 5) Assemble PyG Data object and save
#----------------------------------------------------
data = Data(
    x=qty_train,            # [num_items, train_days]
    y=qty_val,              # [num_items, val_days]
    edge_index=edge_index   # [2, num_edges]
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, OUT_PATH)
print(f"✅  Saved dataset with {num_items} items × {num_days} days → {OUT_PATH}")

#----------------------------------------------------
#                   End of file
#----------------------------------------------------

## Instructions ##
# 1. Ensure data/inventory.csv   and data/sales.csv exist.
# 2. Activate your virtual‑env, cd to project root.
# 3. Run:  python -m src.gnn.preprocess
# 4. Verify models/gnn_dataset.pt is created.
