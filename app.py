"""
AML Transaction Screening
GraphSAGE edge classifier — waltertaya/aml-gnn-ibm-baseline-medium
"""

import json
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID = "waltertaya/aml-gnn-ibm-baseline-medium"

# ── MUST match inference.py and the training notebook c07-graph exactly ───────
# Wrong values / wrong order = one-hot bits land in wrong positions = bad scores
CURRENCIES = [
    "US Dollar", "Euro", "Bitcoin", "UK Pound", "Yen",
    "Ruble", "Yuan", "Swiss Franc", "Canadian Dollar", "Australian Dollar",
    "Mexican Peso", "Brazil Real", "Indian Rupee", "Saudi Riyal",
]

PAYMENT_TYPES = [
    "Wire", "Credit Card", "ACH", "Cheque", "Cash",
    "Crypto", "Reinvestment", "Bills",
]

# Pre-built fixed column order — mirrors inference.py EDGE_FEATURE_COLS
# NEVER use pd.get_dummies alone: it only creates columns for values present
# in the current batch and sorts alphabetically — both wrong for inference.
def _build_edge_columns():
    cols = ["amount", "hour", "dayofweek"]
    for c in CURRENCIES:
        cols.append(f"currency_{c}")
    cols.append("currency_nan")
    for p in PAYMENT_TYPES:
        cols.append(f"payment_type_{p}")
    cols.append("payment_type_nan")
    return cols

EDGE_FEATURE_COLS = _build_edge_columns()  # 3 + 14 + 1 + 8 + 1 = 27

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Transaction Screening",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: 14px;
        }
        h1 { font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em; }
        h2 { font-size: 1.05rem; font-weight: 600; margin-top: 1.5rem; }
        div[data-testid="metric-container"] {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 0.75rem 1rem;
        }
        .alert-high {
            background: #fff5f5;
            border-left: 3px solid #e53e3e;
            padding: 0.6rem 0.9rem;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
        .alert-clear {
            background: #f0fff4;
            border-left: 3px solid #38a169;
            padding: 0.6rem 0.9rem;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model definition ──────────────────────────────────────────────────────────
class EdgeGNN(nn.Module):
    """2-layer GraphSAGE node encoder + MLP edge classifier."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

    def edge_logits(self, node_emb, edge_index, edge_attr, local_idx=None):
        if local_idx is not None:
            s, d, ea = edge_index[0, local_idx], edge_index[1, local_idx], edge_attr[local_idx]
        else:
            s, d, ea = edge_index[0], edge_index[1], edge_attr
        return self.edge_mlp(torch.cat([node_emb[s], node_emb[d], ea], dim=1)).squeeze(-1)

    def forward(self, x, edge_index, edge_attr):
        return self.edge_logits(self.encode_nodes(x, edge_index), edge_index, edge_attr)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    cfg_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin")
    model = EdgeGNN(
        in_dim=cfg["in_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.2),
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    try:
        m_path = hf_hub_download(repo_id=HF_REPO_ID, filename="metrics.json")
        with open(m_path) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    return model, cfg, metrics


# ── Timestamp parser: datetime string → hour + dayofweek ─────────────────────
def parse_timestamp_column(series: pd.Series):
    """
    Returns (hour_series, dayofweek_series) from a raw timestamp column.
    Handles: datetime strings, numeric step values (IBM dataset uses step=hour).
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() > len(series) * 0.5:
        return parsed.dt.hour.fillna(12).astype(float), parsed.dt.dayofweek.fillna(0).astype(float)
    # IBM dataset uses integer "step" = hours since simulation start
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric % 24).astype(float), ((numeric // 24) % 7).astype(float)


# ── Feature engineering — must match inference.py and c07-graph exactly ───────
def build_features(df: pd.DataFrame):
    df = df.copy()

    all_accounts  = pd.Index(df["src"]).append(pd.Index(df["dst"])).unique()
    account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes     = len(all_accounts)

    df["src_id"] = df["src"].map(account_to_id).astype("int32")
    df["dst_id"] = df["dst"].map(account_to_id).astype("int32")

    edge_index = torch.tensor(
        df[["src_id", "dst_id"]].to_numpy().T.astype("int64"), dtype=torch.long
    )

    # ── Edge features: fixed 27-column vector (BUG FIX) ──────────────────────
    # OLD code used pd.get_dummies — only creates columns for values PRESENT in
    # this batch and sorts them alphabetically. On a single transaction that gives
    # just 4 columns; pad_edge_attr then zero-pads at the END, so every categorical
    # bit lands in the wrong position vs what the model was trained on.
    # FIX: build the full fixed-order vector manually using EDGE_FEATURE_COLS.
    n    = len(df)
    feat = np.zeros((n, len(EDGE_FEATURE_COLS)), dtype=np.float32)

    feat[:, 0] = np.log1p(pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0).values)
    feat[:, 1] = (pd.to_numeric(df["hour"],      errors="coerce").fillna(12).values / 23.0) if "hour"      in df.columns else 0.5
    feat[:, 2] = (pd.to_numeric(df["dayofweek"], errors="coerce").fillna(0).values  / 6.0)  if "dayofweek" in df.columns else 0.0

    if "currency" in df.columns:
        for row_i, val in enumerate(df["currency"].astype(str)):
            key = f"currency_{val}"
            col = EDGE_FEATURE_COLS.index(key) if key in EDGE_FEATURE_COLS else EDGE_FEATURE_COLS.index("currency_nan")
            feat[row_i, col] = 1.0
    else:
        feat[:, EDGE_FEATURE_COLS.index("currency_nan")] = 1.0

    if "payment_type" in df.columns:
        for row_i, val in enumerate(df["payment_type"].astype(str)):
            key = f"payment_type_{val}"
            col = EDGE_FEATURE_COLS.index(key) if key in EDGE_FEATURE_COLS else EDGE_FEATURE_COLS.index("payment_type_nan")
            feat[row_i, col] = 1.0
    else:
        feat[:, EDGE_FEATURE_COLS.index("payment_type_nan")] = 1.0

    edge_attr = torch.tensor(feat, dtype=torch.float32)

    # ── Node features: 10 dims matching c07-graph exactly (BUG FIX) ──────────
    # OLD code had 7 wrong features: missing out_hour_mean, in_hour_mean,
    # out_night_ratio, in_night_ratio — and had out_in_ratio which was never
    # a training feature. pad_edge_attr masked this by zero-filling, but those
    # zeros flow through trained weight slots that expect real values.
    node_df       = pd.DataFrame(index=np.arange(num_nodes))
    amt_log       = np.log1p(pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0))
    df["_amt_log"] = amt_log

    node_df["out_count"]    = df.groupby("src_id").size().reindex(node_df.index, fill_value=0)
    node_df["in_count"]     = df.groupby("dst_id").size().reindex(node_df.index, fill_value=0)
    node_df["out_amt_sum"]  = df.groupby("src_id")["_amt_log"].sum().reindex(node_df.index,  fill_value=0)
    node_df["in_amt_sum"]   = df.groupby("dst_id")["_amt_log"].sum().reindex(node_df.index,  fill_value=0)
    node_df["out_amt_mean"] = df.groupby("src_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)
    node_df["in_amt_mean"]  = df.groupby("dst_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)

    if "hour" in df.columns:
        df["_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype("float32")
        node_df["out_hour_mean"]   = df.groupby("src_id")["hour"].mean().reindex(node_df.index,   fill_value=12.0)
        node_df["in_hour_mean"]    = df.groupby("dst_id")["hour"].mean().reindex(node_df.index,   fill_value=12.0)
        node_df["out_night_ratio"] = df.groupby("src_id")["_night"].mean().reindex(node_df.index, fill_value=0.0)
        node_df["in_night_ratio"]  = df.groupby("dst_id")["_night"].mean().reindex(node_df.index, fill_value=0.0)
    else:
        node_df["out_hour_mean"]   = 12.0
        node_df["in_hour_mean"]    = 12.0
        node_df["out_night_ratio"] = 0.0
        node_df["in_night_ratio"]  = 0.0

    node_x = torch.tensor(node_df.fillna(0.0).values.astype("float32"), dtype=torch.float32)
    return edge_index, edge_attr, node_x


def pad_edge_attr(edge_attr: torch.Tensor, expected_dim: int) -> torch.Tensor:
    cur = edge_attr.shape[1]
    if cur == expected_dim:
        return edge_attr
    if cur < expected_dim:
        return torch.cat([edge_attr, torch.zeros(edge_attr.shape[0], expected_dim - cur)], dim=1)
    return edge_attr[:, :expected_dim]


@torch.no_grad()
def score_transactions(model, cfg, edge_index, edge_attr, node_x):
    edge_attr = pad_edge_attr(edge_attr, cfg["edge_dim"])
    in_dim = cfg["in_dim"]
    if node_x.shape[1] < in_dim:
        node_x = torch.cat([node_x, torch.zeros(node_x.shape[0], in_dim - node_x.shape[1])], dim=1)
    elif node_x.shape[1] > in_dim:
        node_x = node_x[:, :in_dim]
    logits = model(node_x, edge_index, edge_attr)
    return torch.sigmoid(logits).cpu().numpy()


# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, cfg, stored_metrics = load_model()
except Exception as e:
    st.error(f"Failed to load model from `{HF_REPO_ID}`: {e}")
    st.stop()

model_threshold = cfg.get("best_threshold", 0.5)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Decision threshold")
    use_model_thr = st.checkbox(
        "Use model's saved threshold",
        value=True,
        help=f"Threshold optimised on the validation set: {model_threshold * 100:.1f}%",
    )
    manual_thr = st.slider(
        "Manual threshold",
        min_value=1,
        max_value=99,
        value=int(round(model_threshold * 100)),
        step=1,
        format="%d%%",
        disabled=use_model_thr,
        help="Increase to reduce false positives; decrease to increase recall.",
    )
    threshold = model_threshold if use_model_thr else manual_thr / 100.0
    st.caption(f"Active threshold: **{threshold * 100:.1f}%**")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## AML Transaction Screening")
st.caption(
    "GraphSAGE edge classifier trained on the IBM HI-Medium AML dataset. "
    "Upload a CSV or enter transactions manually to score them for suspicious activity."
)

with st.expander("Model information", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "GraphSAGE")
    c2.metric("Hidden dim", cfg.get("hidden_dim", "—"))
    c3.metric("Decision threshold", f"{threshold * 100:.1f}%")
    c4.metric("Trained on", "IBM HI-Medium")
    if stored_metrics.get("test_metrics"):
        tm = stored_metrics["test_metrics"]
        st.markdown("**Test set performance**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1", f"{tm.get('f1', 0):.4f}")
        m2.metric("Precision", f"{tm.get('precision', 0):.4f}")
        m3.metric("Recall", f"{tm.get('recall', 0):.4f}")
        m4.metric("PR-AUC", f"{tm.get('pr_auc', 0):.4f}")

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs(["Upload CSV", "Manual entry"])
df_input = None

with tab_upload:
    st.caption(
        "Required columns: `From Account` (or `src`), `To Account` (or `dst`), `Amount Paid` (or `amount`). "
        "Optional: `Timestamp` (datetime or numeric step), `Payment Currency`, `Payment Format`."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            df_input.columns = [c.strip().lower().replace(" ", "_") for c in df_input.columns]

            alias_map = {
                "from_account": "src", "originator": "src", "sender": "src",
                "to_account": "dst", "beneficiary": "dst", "receiver": "dst",
                "amount_paid": "amount", "transaction_amount": "amount", "amt": "amount",
                "payment_currency": "currency", "receiving_currency": "currency",
                "payment_format": "payment_type",
            }
            df_input = df_input.rename(
                columns={k: v for k, v in alias_map.items() if k in df_input.columns}
            )

            # Parse timestamp → hour + dayofweek (keep raw timestamp for display)
            ts_col = next(
                (c for c in ["timestamp", "step", "time", "date", "tran_date"] if c in df_input.columns),
                None,
            )
            if ts_col:
                df_input["hour"], df_input["dayofweek"] = parse_timestamp_column(df_input[ts_col])

            missing = {"src", "dst", "amount"} - set(df_input.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
                df_input = None
            else:
                st.success(f"{len(df_input):,} transactions loaded.")
                st.dataframe(df_input.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse file: {e}")

with tab_manual:
    st.caption("Enter one transaction per row.")

    n_rows = st.number_input("Number of transactions", min_value=1, max_value=50, value=4, step=1)

    col_src, col_dst, col_amt, col_ts, col_cur, col_pay = st.columns([2, 2, 1.5, 2, 1.5, 1.5])
    col_src.markdown("**Source account**")
    col_dst.markdown("**Destination account**")
    col_amt.markdown("**Amount**")
    col_ts.markdown("**Timestamp**")
    col_cur.markdown("**Currency**")
    col_pay.markdown("**Payment type**")

    rows = []
    defaults_src = ["ACC001", "ACC002", "ACC003", "ACC001"]
    defaults_dst = ["ACC002", "ACC003", "ACC004", "ACC004"]
    defaults_amt = [15000.0, 9500.0, 800.0, 250000.0]
    defaults_cur = ["US Dollar", "US Dollar", "Euro", "US Dollar"]
    defaults_pay = ["Wire", "Wire", "ACH", "Wire"]

    for i in range(int(n_rows)):
        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1.5, 2, 1.5, 1.5])
        src = c1.text_input(
            f"src_{i}", value=defaults_src[i] if i < len(defaults_src) else f"ACC{i+1:03d}",
            label_visibility="collapsed", key=f"src_{i}",
        )
        dst = c2.text_input(
            f"dst_{i}", value=defaults_dst[i] if i < len(defaults_dst) else f"ACC{i+2:03d}",
            label_visibility="collapsed", key=f"dst_{i}",
        )
        amt = c3.number_input(
            f"amt_{i}", value=defaults_amt[i] if i < len(defaults_amt) else 1000.0,
            min_value=0.0, format="%.2f", label_visibility="collapsed", key=f"amt_{i}",
        )
        ts = c4.datetime_input(
            f"ts_{i}",
            value=datetime.now().replace(second=0, microsecond=0),
            label_visibility="collapsed", key=f"ts_{i}",
        ) if hasattr(st, "datetime_input") else c4.text_input(
            f"ts_{i}", value=datetime.now().strftime("%Y-%m-%d %H:%M"),
            label_visibility="collapsed", key=f"ts_{i}",
        )
        cur = c5.selectbox(
            f"cur_{i}",
            options=CURRENCIES,
            index=CURRENCIES.index(defaults_cur[i]) if i < len(defaults_cur) else 0,
            label_visibility="collapsed", key=f"cur_{i}",
        )
        pay = c6.selectbox(
            f"pay_{i}",
            options=PAYMENT_TYPES,
            index=PAYMENT_TYPES.index(defaults_pay[i]) if i < len(defaults_pay) else 0,
            label_visibility="collapsed", key=f"pay_{i}",
        )

        # Parse timestamp to hour + dayofweek
        if isinstance(ts, datetime):
            hour_val = float(ts.hour)
            dow_val = float(ts.weekday())
        else:
            try:
                dt = datetime.fromisoformat(str(ts))
                hour_val = float(dt.hour)
                dow_val = float(dt.weekday())
            except Exception:
                hour_val = 12.0
                dow_val = 0.0

        rows.append({
            "src": src, "dst": dst, "amount": amt,
            "hour": hour_val, "dayofweek": dow_val,
            "currency": cur, "payment_type": pay,
        })

    if st.button("Score transactions", type="primary"):
        df_input = pd.DataFrame(rows)

# ── Inference ─────────────────────────────────────────────────────────────────
if df_input is not None and len(df_input) > 0:
    st.markdown("---")
    st.markdown("## Results")

    with st.spinner("Scoring..."):
        try:
            edge_index, edge_attr, node_x = build_features(df_input)
            probs = score_transactions(model, cfg, edge_index, edge_attr, node_x)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    result_df = df_input[["src", "dst", "amount"]].copy()
    for col in ["currency", "payment_type"]:
        if col in df_input.columns:
            result_df[col] = df_input[col].values

    result_df["risk_score"] = probs          # raw float kept for threshold comparison
    result_df["flagged"] = (probs >= threshold).astype(bool)
    result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    n_total = len(result_df)
    n_flagged = int(result_df["flagged"].sum())
    n_clear = n_total - n_flagged

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions scored", f"{n_total:,}")
    m2.metric("Flagged", f"{n_flagged:,}")
    m3.metric("Clear", f"{n_clear:,}")
    m4.metric("Threshold", f"{threshold * 100:.1f}%")

    if n_flagged > 0:
        st.markdown(
            f'<div class="alert-high"><strong>{n_flagged} transaction{"s" if n_flagged != 1 else ""} '
            f'flagged</strong> above the {threshold * 100:.1f}% risk threshold.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-clear">No transactions flagged at the current threshold.</div>',
            unsafe_allow_html=True,
        )

    def render_table(df):
        d = df.copy()
        d["amount"] = pd.to_numeric(d["amount"], errors="coerce").map("{:,.2f}".format)
        d["risk_score"] = d["risk_score"].map(lambda v: f"{v * 100:.2f}%")
        d["flagged"] = d["flagged"].map(lambda x: "Yes" if x else "No")
        st.dataframe(d, use_container_width=True, hide_index=True)

    tab_all, tab_flagged = st.tabs(
        [f"All transactions ({n_total})", f"Flagged ({n_flagged})"]
    )
    with tab_all:
        render_table(result_df)
    with tab_flagged:
        if n_flagged > 0:
            render_table(result_df[result_df["flagged"]])
        else:
            st.info("No transactions were flagged at the current threshold.")

    # Risk distribution + top accounts
    st.markdown("### Risk distribution")
    hist_col, acct_col = st.columns([2, 1])

    with hist_col:
        counts, edges = np.histogram(probs, bins=20, range=(0, 1))
        hist_df = pd.DataFrame({
            "bin": [f"{edges[i]*100:.0f}–{edges[i+1]*100:.0f}%" for i in range(len(counts))],
            "count": counts,
        })
        st.bar_chart(hist_df.set_index("bin"), use_container_width=True, height=220)

    with acct_col:
        st.markdown("**Top accounts by max risk**")
        acct = pd.concat([
            result_df[["src", "risk_score"]].rename(columns={"src": "account"}),
            result_df[["dst", "risk_score"]].rename(columns={"dst": "account"}),
        ])
        acct_summary = (
            acct.groupby("account")["risk_score"]
            .max().sort_values(ascending=False)
            .head(10).reset_index()
        )
        acct_summary.columns = ["Account", "Max risk"]
        acct_summary["Max risk"] = acct_summary["Max risk"].map(lambda v: f"{v * 100:.2f}%")
        st.dataframe(acct_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    # Export with percentage in CSV too
    export_df = result_df.copy()
    export_df["risk_score"] = export_df["risk_score"].map(lambda v: f"{v * 100:.2f}%")
    export_df["flagged"] = export_df["flagged"].map(lambda x: "Yes" if x else "No")
    st.download_button(
        label="Download results as CSV",
        data=export_df.to_csv(index=False).encode(),
        file_name="aml_scored_transactions.csv",
        mime="text/csv",
    )
