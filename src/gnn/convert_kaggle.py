#----------------------------------------------------
#      Path: src/gnn/convert_kaggle.py
#----------------------------------------------------
"""
Convert Kaggle Store‑Item dataset (train.csv) into:

  - data/sales.csv      (date, item_id, qty_sold)
  - data/inventory.csv  (item_id, name, barcode, current_stock)

Run: python -m src.gnn.convert_kaggle
"""

#------------------ standard libs -------------------
import pathlib
import pandas as pd
import numpy as np

#----------------------------------------------------
#        project‑root & source / destination paths
#----------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_TRAIN = ROOT / "data" / "kaggle" / "train.csv"    # Kaggle file location
OUT_SALES = ROOT / "data" / "sales.csv"
OUT_INV   = ROOT / "data" / "inventory.csv"

#----------------------------------------------------
#   1) load Kaggle train.csv (5 years of daily sales)
#----------------------------------------------------
df = pd.read_csv(RAW_TRAIN, parse_dates=["date"])     # 913 k rows

#----------------------------------------------------
#   2) build sales.csv  (drop store, rename columns)
#----------------------------------------------------
sales = (
    df.rename(columns={"sales": "qty_sold"})          # qty_sold is our column name
      .loc[:, ["date", "item", "qty_sold"]]           # keep date, item, qty
      .rename(columns={"item": "item_id"})            # item → item_id
)
sales.to_csv(OUT_SALES, index=False)

#----------------------------------------------------
#   3) derive inventory.csv
#----------------------------------------------------
# last 30‑day sum → naive starting stock level
last_30 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=29)]
stock_series = last_30.groupby("item")["sales"].sum()   # Series[item] = qty

inv_rows = []
for item_id in sorted(df["item"].unique()):
    name = f"Product {item_id:02d}"                     # human‑readable name
    barcode = f"000000000{item_id:03d}"                 # 12‑digit dummy barcode
    current_stock = int(stock_series.get(item_id, 0))   # fallback 0 if no sales
    inv_rows.append((item_id, name, barcode, current_stock))

inv = pd.DataFrame(inv_rows,
                   columns=["item_id", "name", "barcode", "current_stock"])
inv.to_csv(OUT_INV, index=False)

#----------------------------------------------------
#                     logging
#----------------------------------------------------
print(f"✅ wrote {OUT_SALES.name} with {len(sales):,} rows")
print(f"✅ wrote {OUT_INV.name} with {len(inv)} rows")

#----------------------------------------------------
#              end of convert_kaggle.py
#----------------------------------------------------

## Instructions ##
# 1.  Download Kaggle train.csv (Store‑Item Demand Forecasting).
# 2.  Place it at   data/kaggle/train.csv
# 3.  Activate your venv and run:
#       python -m src.gnn.convert_kaggle
