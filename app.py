"""
AML Transaction Screening
GraphSAGE edge classifier — waltertaya/aml-gnn-ibm-baseline-medium
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID = "waltertaya/aml-gnn-ibm-baseline-medium"

# Exact values present in the IBM AML dataset (used during training one-hot encoding)
CURRENCIES = [
    "US Dollar", "Euro", "UK Pound", "Swiss Franc", "Yen",
    "Ruble", "Yuan", "Bitcoin", "Australian Dollar", "Canadian Dollar",
    "Brazilian Real", "Mexican Peso", "Saudi Riyal", "South Korean Won",
    "Swedish Krona",
]

PAYMENT_TYPES = [
    "Reinvestment", "Wire", "Credit Card", "ACH", "Cheque",
    "Cash", "Bitcoin", "Cross-border Wire",
]

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


# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def build_features(df: pd.DataFrame):
    df = df.copy()

    all_accounts = pd.Index(df["src"]).append(pd.Index(df["dst"])).unique()
    account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    df["src_id"] = df["src"].map(account_to_id).astype("int32")
    df["dst_id"] = df["dst"].map(account_to_id).astype("int32")

    edge_index = torch.tensor(
        df[["src_id", "dst_id"]].to_numpy().T.astype("int64"), dtype=torch.long
    )

    edge_cont = pd.DataFrame(index=df.index)
    edge_cont["amount"] = np.log1p(
        pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0.0)
    )
    edge_cont["hour"] = (
        pd.to_numeric(df["hour"], errors="coerce").fillna(12) / 23.0
        if "hour" in df.columns else 0.5
    )
    edge_cont["dayofweek"] = (
        pd.to_numeric(df["dayofweek"], errors="coerce").fillna(0) / 6.0
        if "dayofweek" in df.columns else 0.0
    )

    cat_cols = [c for c in ["currency", "payment_type"] if c in df.columns]
    if cat_cols:
        edge_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True).astype("float32")
        edge_feat_df = pd.concat([edge_cont, edge_cat], axis=1)
    else:
        edge_feat_df = edge_cont

    edge_feat_df = edge_feat_df.fillna(0.0).astype("float32")
    edge_attr = torch.tensor(edge_feat_df.values, dtype=torch.float32)

    node_df = pd.DataFrame(index=np.arange(num_nodes))
    node_df["out_count"] = df.groupby("src_id").size().reindex(node_df.index, fill_value=0)
    node_df["in_count"] = df.groupby("dst_id").size().reindex(node_df.index, fill_value=0)
    amt_log = np.log1p(pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0))
    df["_amt_log"] = amt_log
    node_df["out_amt_sum"] = df.groupby("src_id")["_amt_log"].sum().reindex(node_df.index, fill_value=0)
    node_df["in_amt_sum"] = df.groupby("dst_id")["_amt_log"].sum().reindex(node_df.index, fill_value=0)
    node_df["out_amt_mean"] = df.groupby("src_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)
    node_df["in_amt_mean"] = df.groupby("dst_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)
    node_df["out_in_ratio"] = (node_df["out_count"] / (node_df["in_count"] + 1e-6)).clip(upper=100.0)

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


# ── Load model (runs once, cached) ───────────────────────────────────────────
try:
    model, cfg, stored_metrics = load_model()
except Exception as e:
    st.error(f"Failed to load model from `{HF_REPO_ID}`: {e}")
    st.stop()

model_threshold = cfg.get("best_threshold", 0.5)

# ── Sidebar — threshold controls ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Decision threshold")
    use_model_thr = st.checkbox(
        "Use model's saved threshold",
        value=True,
        help=f"Threshold optimised on the validation set: {model_threshold:.3f}",
    )
    manual_thr = st.slider(
        "Manual threshold",
        min_value=0.01,
        max_value=0.99,
        value=float(round(model_threshold, 2)),
        step=0.01,
        disabled=use_model_thr,
        help="Increase to reduce false positives; decrease to increase recall.",
    )
    threshold = model_threshold if use_model_thr else manual_thr
    st.caption(f"Active threshold: **{threshold:.3f}**")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## AML Transaction Screening")
st.caption(
    "GraphSAGE edge classifier trained on the IBM HI-Medium AML dataset. "
    "Upload a CSV or enter transactions manually to score them for suspicious activity."
)

# Model info
with st.expander("Model information", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "GraphSAGE")
    c2.metric("Hidden dim", cfg.get("hidden_dim", "—"))
    c3.metric("Decision threshold", f"{threshold:.3f}")
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
        "Optional: `Payment Currency`, `Payment Format`, `Timestamp`."
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
                "timestamp": "hour", "step": "hour",
            }
            df_input = df_input.rename(
                columns={k: v for k, v in alias_map.items() if k in df_input.columns}
            )
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
    st.caption("Enter one transaction per row. Currency and payment type must match the training data values.")

    n_rows = st.number_input("Number of transactions", min_value=1, max_value=50, value=4, step=1)

    col_src, col_dst, col_amt, col_cur, col_pay = st.columns([2, 2, 1.5, 1.5, 1.5])
    col_src.markdown("**Source account**")
    col_dst.markdown("**Destination account**")
    col_amt.markdown("**Amount**")
    col_cur.markdown("**Currency**")
    col_pay.markdown("**Payment type**")

    rows = []
    defaults_src = ["ACC001", "ACC002", "ACC003", "ACC001"]
    defaults_dst = ["ACC002", "ACC003", "ACC004", "ACC004"]
    defaults_amt = [15000.0, 9500.0, 800.0, 250000.0]
    defaults_cur = ["US Dollar", "US Dollar", "Euro", "US Dollar"]
    defaults_pay = ["Wire", "Wire", "ACH", "Wire"]

    for i in range(int(n_rows)):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 1.5, 1.5])
        src = c1.text_input(
            f"src_{i}", value=defaults_src[i] if i < len(defaults_src) else f"ACC{i+1:03d}",
            label_visibility="collapsed", key=f"src_{i}"
        )
        dst = c2.text_input(
            f"dst_{i}", value=defaults_dst[i] if i < len(defaults_dst) else f"ACC{i+2:03d}",
            label_visibility="collapsed", key=f"dst_{i}"
        )
        amt = c3.number_input(
            f"amt_{i}", value=defaults_amt[i] if i < len(defaults_amt) else 1000.0,
            min_value=0.0, format="%.2f", label_visibility="collapsed", key=f"amt_{i}"
        )
        cur = c4.selectbox(
            f"cur_{i}",
            options=CURRENCIES,
            index=CURRENCIES.index(defaults_cur[i]) if i < len(defaults_cur) else 0,
            label_visibility="collapsed", key=f"cur_{i}"
        )
        pay = c5.selectbox(
            f"pay_{i}",
            options=PAYMENT_TYPES,
            index=PAYMENT_TYPES.index(defaults_pay[i]) if i < len(defaults_pay) else 0,
            label_visibility="collapsed", key=f"pay_{i}"
        )
        rows.append({"src": src, "dst": dst, "amount": amt, "currency": cur, "payment_type": pay})

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

    result_df["risk_score"] = probs
    result_df["flagged"] = (probs >= threshold).astype(bool)
    result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    n_total = len(result_df)
    n_flagged = int(result_df["flagged"].sum())
    n_clear = n_total - n_flagged

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions scored", f"{n_total:,}")
    m2.metric("Flagged", f"{n_flagged:,}")
    m3.metric("Clear", f"{n_clear:,}")
    m4.metric("Threshold", f"{threshold:.3f}")

    if n_flagged > 0:
        st.markdown(
            f'<div class="alert-high"><strong>{n_flagged} transaction{"s" if n_flagged != 1 else ""} '
            f'flagged</strong> above the {threshold:.3f} risk threshold.</div>',
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
        d["risk_score"] = d["risk_score"].map("{:.4f}".format)
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
            "bin": [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))],
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
        acct_summary["Max risk"] = acct_summary["Max risk"].map("{:.4f}".format)
        st.dataframe(acct_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.download_button(
        label="Download results as CSV",
        data=result_df.to_csv(index=False).encode(),
        file_name="aml_scored_transactions.csv",
        mime="text/csv",
    )
