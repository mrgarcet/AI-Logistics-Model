# --------------------------------------------------
#  Path: src/app/chatbot.py
# --------------------------------------------------
"""
NLP‑driven helper that understands simple inventory queries:

    • “what’s the stock of item A?”
    • “forecast product 12”
    • “add 5 to item B”
    • “remove 2 product id 07”
    • “change product 3 count to 99”

The DistilBERT intent‑classifier lives in  src/nlp/infer.py
"""

import re
from src.nlp.infer import predict_intent          # DistilBERT classifier
from src.gnn.infer import forecast                # 30‑day demand forecast
from .inventory    import get_stock, update_stock # CSV helpers

# ---------- regexes ----------------------------------------------------
# numeric ids:  product 7,  item id 023, …
_item_pat = re.compile(
    r"(?:item|product)\s*(?:id\s*)?0*([1-9]\d{0,2})\b",  # → 1‑999
    re.IGNORECASE,
)

# letter ids:  item A, product b, …
_item_letter_pat = re.compile(
    r"(?:item|product)\s*([A-Z])\b",
    re.IGNORECASE,
)

# ---------- public API -------------------------------------------------
def handle_user_query(text: str) -> str:
    """
    Main entry point called from Streamlit.
    """
    intent = predict_intent(text)

    # ── stock lookup / forecast ────────────────────────────────────────
    if intent in {"CheckStock", "ForecastQuery"}:
        item = _extract_item(text)
        if item is None:
            return "Sorry, I couldn't find an item number."

        if intent == "CheckStock":
            return f"Current stock for Item {item}: {get_stock(item)}"

        val = forecast(item)
        return f"Projected 30‑day demand for Item {item}: {val:.1f}"

    # ── add / remove / update quantities ───────────────────────────────
    if intent in {"AddItem", "RemoveItem", "UpdateStock"}:
        item, qty = _extract_item_qty(text, intent)
        if item is None or qty is None:
            return "Sorry, I couldn't understand the item / quantity."

        # UpdateStock → absolute value → convert to delta
        delta = qty - get_stock(item) if intent == "UpdateStock" else qty
        update_stock(item, delta)
        return f"✅  New stock for Item {item}: {get_stock(item)}"

    # ── fallback ───────────────────────────────────────────────────────
    return "Sorry, I didn't understand."


# ----------------------------------------------------------------------
#                           helper functions
# ----------------------------------------------------------------------
def _extract_item(text: str) -> int | None:
    """
    Return numeric item‑id (1‑999) or None.

        “item 7”        → 7
        “product id 007”→ 7
        “item B”        → 2   (A‑Z → 1‑26)
    """
    m = _item_pat.search(text)
    if m:
        return int(m.group(1))

    m = _item_letter_pat.search(text)
    if m:
        return ord(m.group(1).upper()) - ord("A") + 1

    return None


def _extract_item_qty(text: str, intent: str) -> tuple[int | None, int | None]:
    """
    Return (item_id, qty) or (None, None).

    * For UpdateStock we look for “… to 99”
    * Otherwise we take the first integer as the delta.
    """
    item = _extract_item(text)
    if item is None:
        return None, None

    if intent == "UpdateStock":
        m = re.search(r"\bto\s+(-?\d+)\b", text, re.I)      # “… to 99”
        qty_s = m.group(1) if m else None
    else:
        m = re.search(r"-?\d+", text)                       # first integer
        qty_s = m.group(0) if m else None

    if qty_s is None:
        return None, None

    return item, int(qty_s)
