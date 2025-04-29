#----------------------------------------------------
#  Path: src/app/inventory.py
#----------------------------------------------------
'''
Tiny helper around data/inventory.csv.
Usage:
    from app.inventory import get_stock, update_stock
'''
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INV  = ROOT / "data" / "inventory.csv"

def _load():
    with open(INV, newline='') as f:
        return {row['item_id']: row for row in csv.DictReader(f)}

def _save(rows):
    with open(INV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item_id', 'name', 'barcode', 'current_stock'])
        for r in rows.values():
            w.writerow([r['item_id'], r['name'], r['barcode'], r['current_stock']])

def get_stock(item_id: int) -> int:
    return int(_load()[str(item_id)]['current_stock'])

def update_stock(item_id: int, delta: int):
    rows = _load()
    rec  = rows[str(item_id)]
    rec['current_stock'] = str(max(int(rec['current_stock']) + delta, 0))
    _save(rows)
