# --------------------------------------------------
# tests/test_regex_helpers.py
# --------------------------------------------------
"""
Smoke‑tests for the private helpers in  src/app/chatbot.py

Run from project root (same place you type  streamlit run ui/dashboard.py):
    pytest -vv
"""

from pathlib import Path
import sys
import re
import pytest

# ── 1) make sure the *project‑root* is on sys.path  ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/AI_Logistics_Model
sys.path.insert(0, str(PROJECT_ROOT))                # ← single, simple line

# now normal imports work
from src.app.chatbot import _extract_item, _extract_item_qty


# ─────────────────────────────────────────────────────────────────────────
# 2)  _extract_item  –– numeric  &  letter IDs
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "q, expected",
    [
        ("show stock of item 7",     7),
        ("product 23 inventory?",    23),
        ("product id 005",           5),
        (" ITEM   42 ",             42),
    ],
)
def test_digit_ids(q, expected):
    assert _extract_item(q) == expected


@pytest.mark.parametrize(
    "q, expected",
    [
        ("what about item A?",      1),
        ("Stock product   C now",   3),
        ("Product   z qty?",       26),
    ],
)
def test_letter_ids(q, expected):
    assert _extract_item(q) == expected


def test_no_id_returns_none():
    assert _extract_item("how are you today?") is None


# ─────────────────────────────────────────────────────────────────────────
# 3)  _extract_item_qty  –– add / remove / update
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "q, intent, exp",
    [
        ("add 5 to item 10",              "AddItem",     (10, 5)),
        ("ADD  -3  product B",            "AddItem",     (2, -3)),
        ("remove 8  item  4",             "RemoveItem",  (4, 8)),
        ("subtract   -6 product Z",       "RemoveItem",  (26, -6)),
        ("update stock of item 3 to 99",  "UpdateStock", (3, 99)),
        ("set product 12 count to 100",   "UpdateStock", (12, 100)),
    ],
)
def test_item_qty_pairs(q, intent, exp):
    assert _extract_item_qty(q, intent) == exp


def test_bad_string_returns_none_tuple():
    assert _extract_item_qty("completely unrelated text", "AddItem") == (
        None,
        None,
    )
