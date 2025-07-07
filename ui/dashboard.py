# --------------------------------------------------
# ui/dashboard.py – Streamlit Inventory‑AI (v2.2)
# --------------------------------------------------
from __future__ import annotations

import sys, threading
from pathlib import Path
from queue import Queue, Empty

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ── project root on sys.path so “src.*” imports work ────────────────────
ROOT = Path(__file__).resolve().parents[1]      # project /
sys.path.append(str(ROOT))

# ── local project imports ───────────────────────────────────────────────
from src.app.chatbot   import handle_user_query
from src.app.inventory import update_stock
from src.gnn.infer     import forecast
from src.cv.ai_scanner import listen                     # YOLO v8n + ZBar

INV_CSV   = ROOT / "data" / "inventory.csv"
WEIGHTS_P = ROOT / "src" / "cv" / "weights" / "yolov8n.pt"

# ── quick sanity‑check for the CV weights (non‑fatal) ───────────────────
if not WEIGHTS_P.exists():
    st.warning(
        "⚠️  AI‑scanner disabled – "
        f"model weights not found at **{WEIGHTS_P.relative_to(ROOT)}**"
    )

# ── helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_inventory() -> pd.DataFrame:
    return pd.read_csv(INV_CSV, dtype={"barcode": str}) #DEBUG - force ‘barcode’ to string so “000…023” keeps its leading zeros

def save_inventory(df: pd.DataFrame):
    df.to_csv(INV_CSV, index=False)

# ---- Streamlit <1.30 / >=1.30 compatibility ---------------------------
def _safe_rerun() -> None:
    """
    Streamlit ≥ 1.30 -> st.rerun()
    older versions  -> st.experimental_rerun()
    """
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        _safe_rerun()


# ── page & KPI header ───────────────────────────────────────────────────
st.set_page_config("Inventory‑AI", layout="wide")

df = load_inventory()
c1, c2, c3 = st.columns(3)
c1.metric("📦 SKUs",        len(df))
c2.metric("📊 Total units", int(df["current_stock"].sum()))
delta_today = st.session_state.get("delta_today", 0)
c3.metric("🆕 Net today",   delta_today, delta=f"{delta_today:+}")

st.divider()

# ── inventory grid (AG‑Grid) ────────────────────────────────────────────
st.subheader("Inventory")

df["30‑day forecast"] = df["item_id"].apply(lambda x: round(forecast(int(x))))

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(enabled=True)
gb.configure_side_bar()                         # columns / filters
gb.configure_column("current_stock", editable=True)

grid = AgGrid(
    df,
    gridOptions=gb.build(),
    height=400,
    theme="balham",
    update_mode=GridUpdateMode.MODEL_CHANGED,
    fit_columns_on_grid_load=True,
    columns_auto_size_mode="FIT_CONTENTS",
)
edited_df: pd.DataFrame = grid["data"]

if st.button("💾 Save table changes"):
    save_inventory(edited_df)
    st.success("Inventory saved ✔")

st.divider()

# ── quick manual adjust ────────────────────────────────────────────────
st.subheader("Quick Adjust")
col_sel, col_qty, col_btn = st.columns([2, 1, 1])
item_sel = col_sel.selectbox("Item ID", edited_df["item_id"])
qty_val  = col_qty.number_input("Quantity (±)", step=1, value=1)

if col_btn.button("Apply"):
    update_stock(int(item_sel), int(qty_val))
    st.session_state.delta_today = (
        st.session_state.get("delta_today", 0) + int(qty_val)
    )
    st.rerun()

# ── chat box ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Chat with Inventory‑AI")

if "chat_log" not in st.session_state:
    st.session_state.chat_log: list[tuple[str, str]] = []

for role, msg in st.session_state.chat_log:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask me…")
if prompt:
    with st.chat_message("user"):       st.markdown(prompt)
    reply = handle_user_query(prompt)
    with st.chat_message("assistant"):  st.markdown(reply)
    st.session_state.chat_log += [("user", prompt), ("assistant", reply)]

# ── AI barcode scanner (sidebar) ───────────────────────────────────────
with st.sidebar:
    st.header("Barcode scanner 🤖")

    # -------------------------------------------------
    # 1) – ONE queue  +  ONE background thread per session
    # -------------------------------------------------
    if "scan_q" not in st.session_state:
        st.session_state.scan_q = Queue()       # will carry bar‑codes
    if "scanner_thread" not in st.session_state:
        st.session_state.scanner_thread = None  # not started yet

    # -------------------------------------------------
    # 2) – start button (creates the thread only once)
    # -------------------------------------------------
    if st.button("📷 Start AI scan", disabled=not WEIGHTS_P.exists()):
        if st.session_state.scanner_thread is None:            # first click
            st.session_state.scanner_thread = threading.Thread(
                target=listen, args=(0, st.session_state.scan_q), daemon=True
            )
            st.session_state.scanner_thread.start()
            st.success("Scanner started ✔")
        else:
            st.info("Scanner already running – point a barcode at the camera!")

    # -------------------------------------------------
    # 3) – poll queue every Streamlit run‑loop tick
    # -------------------------------------------------
    try:
        bc = st.session_state.scan_q.get_nowait()          # <- NEW queue
        bc = bc.strip()                                    # keep leading zeros
        st.session_state["last_bc"] = bc                   # remember
    except Empty:
        bc = st.session_state.get("last_bc")               # nothing new

    # -------------------------------------------------
    # 4) – show result or unknown‑barcode error
    # -------------------------------------------------
    if bc:
        norm_bc = bc.lstrip("0")                           # accept 00023 / 23
        row = edited_df[
            (edited_df["barcode"] == bc)
            | (edited_df["barcode"] == norm_bc)
        ]

        if row.empty:
            st.error(f"Unknown barcode: **{bc}**")
        else:
            iid  = int(row["item_id"].iloc[0])
            name = row["name"].iloc[0]

            st.success(f"Detected **Item {iid} – {name}**")
            qty = st.number_input("Quantity (±)", step=1, value=1, key="ai_qty")

            if st.button("Update stock"):
                update_stock(iid, int(qty))
                st.session_state.delta_today = (
                    st.session_state.get("delta_today", 0) + int(qty)
                )
                load_inventory.clear() # NEW - clears invalidate cache
                st.session_state.pop("last_bc", None)      # reset
                st.rerun()
