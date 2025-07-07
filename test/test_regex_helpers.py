# --------------------------------------------------
#  tests/test_regex_helpers.py
# --------------------------------------------------
"""
Smoke‑tests for the helper functions in src/app/chatbot.py

Run from project root (inside the venv):

    pytest -q                       # quiet
    pytest -vv                      # verbose
"""

from src.app.chatbot import _extract_item, _extract_item_qty


# ------------------------------------------------------------------
# _extract_item()  – single‑argument helper
# ------------------------------------------------------------------
def test_digit_ids():
    assert _extract_item("What’s the stock of item 7?") == 7
    assert _extract_item("Check product 012") == 12
    assert _extract_item("product id 99") == 99


def test_letter_ids():
    assert _extract_item("Inventory for item A") == 1
    assert _extract_item("product C current stock") == 3
    assert _extract_item("Need item Z forecast") == 26


def test_no_id_returns_none():
    assert _extract_item("How’s the weather today?") is None


# ------------------------------------------------------------------
# _extract_item_qty()  – two‑argument helper
# ------------------------------------------------------------------
def _iq(text: str, intent: str):
    """shorthand wrapper – returns (item, qty)"""
    return _extract_item_qty(text, intent)


# --- AddItem / RemoveItem (quantity may come first or last) ----------
def test_add_remove_patterns():
    assert _iq("Add 5 to item 10", "AddItem") == (10, 5)
    assert _iq("remove 3 item 8", "RemoveItem") == (8, -3)
    assert _iq("Add fifteen product B", "AddItem") == (2, 15)  # B→2


# --- UpdateStock (“…to 99” pattern) ----------------------------------
def test_update_patterns():
    assert _iq("update stock of item 4 to 100", "UpdateStock") == (4, 100)
    assert _iq("Set product 7 to 22 units", "UpdateStock") == (7, 22)
    assert _iq("change item C to 17", "UpdateStock") == (3, 17)


def test_bad_strings_return_none():
    assert _iq("update nothing here", "UpdateStock") == (None, None)
    assert _iq("add something weird", "AddItem") == (None, None)
