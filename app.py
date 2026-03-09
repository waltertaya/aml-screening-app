"""
AML Detection — Streamlit inference app
Uses the EdgeGNN model pushed to Hugging Face Hub.
"""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Transaction Screening",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — clean, professional ────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Hide Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        /* Typography */
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: 14px;
        }

        h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; }
        h2 { font-size: 1.1rem; font-weight: 600; margin-top: 1.5rem; }
        h3 { font-size: 0.95rem; font-weight: 600; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 0.75rem 1rem;
        }

        /* Table */
        .dataframe td, .dataframe th {
            font-size: 0.82rem !important;
        }

        /* Alert banner */
        .alert-high {
            background: #fff5f5;
            border-left: 3px solid #e53e3e;
            padding: 0.6rem 0.9rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        .alert-clear {
            background: #f0fff4;
            border-left: 3px solid #38a169;
            padding: 0.6rem 0.9rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model definition (must match training code exactly) ───────────────────────
class EdgeGNN(nn.Module):
    """2-layer GraphSAGE node encoder + MLP edge classifier."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError:
            raise ImportError(
                "torch-geometric is required. "
                "Install with: pip install torch-geometric"
            )
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
            s = edge_index[0, local_idx]
            d = edge_index[1, local_idx]
            ea = edge_attr[local_idx]
        else:
            s = edge_index[0]
            d = edge_index[1]
            ea = edge_attr
        return self.edge_mlp(
            torch.cat([node_emb[s], node_emb[d], ea], dim=1)
        ).squeeze(-1)

    def forward(self, x, edge_index, edge_attr):
        return self.edge_logits(
            self.encode_nodes(x, edge_index), edge_index, edge_attr
        )


# ── HF model loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_model(repo_id: str, hf_token: str | None):
    """Download config + weights from HF and return (model, config)."""
    token = hf_token or None

    cfg_path = hf_hub_download(
        repo_id=repo_id, filename="config.json", token=token
    )
    with open(cfg_path) as f:
        cfg = json.load(f)

    weights_path = hf_hub_download(
        repo_id=repo_id, filename="pytorch_model.bin", token=token
    )

    device = torch.device("cpu")
    model = EdgeGNN(
        in_dim=cfg["in_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Try to load stored metrics
    try:
        metrics_path = hf_hub_download(
            repo_id=repo_id, filename="metrics.json", token=token
        )
        with open(metrics_path) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    return model, cfg, metrics


# ── Feature engineering (mirrors the notebook exactly) ────────────────────────
def build_features(df: pd.DataFrame):
    """
    df must have columns: src, dst, amount
    Optional: hour, dayofweek, currency, payment_type

    Returns (edge_index, edge_attr, node_x, account_to_id)
    """
    df = df.copy()

    # Node index
    all_accounts = pd.Index(df["src"]).append(pd.Index(df["dst"])).unique()
    account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    df["src_id"] = df["src"].map(account_to_id).astype("int32")
    df["dst_id"] = df["dst"].map(account_to_id).astype("int32")

    # Edge index
    edge_index = torch.tensor(
        df[["src_id", "dst_id"]].to_numpy().T.astype("int64"), dtype=torch.long
    )

    # Continuous edge features
    edge_cont = pd.DataFrame()
    edge_cont["amount"] = np.log1p(
        pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0.0)
    )

    if "hour" in df.columns:
        edge_cont["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(12) / 23.0
    else:
        edge_cont["hour"] = 0.5

    if "dayofweek" in df.columns:
        edge_cont["dayofweek"] = (
            pd.to_numeric(df["dayofweek"], errors="coerce").fillna(0) / 6.0
        )
    else:
        edge_cont["dayofweek"] = 0.0

    # Categorical edge features (one-hot)
    cat_cols = [c for c in ["currency", "payment_type"] if c in df.columns]
    if cat_cols:
        edge_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True).astype(
            "float32"
        )
        edge_feat_df = pd.concat([edge_cont, edge_cat], axis=1)
    else:
        edge_feat_df = edge_cont

    edge_feat_df = edge_feat_df.fillna(0.0).astype("float32")
    edge_attr = torch.tensor(edge_feat_df.values, dtype=torch.float32)

    # Node features (aggregate stats)
    node_df = pd.DataFrame(index=np.arange(num_nodes))
    node_df["out_count"] = (
        df.groupby("src_id").size().reindex(node_df.index, fill_value=0)
    )
    node_df["in_count"] = (
        df.groupby("dst_id").size().reindex(node_df.index, fill_value=0)
    )
    amt_log = np.log1p(
        pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0)
    )
    df["_amt_log"] = amt_log
    node_df["out_amt_sum"] = (
        df.groupby("src_id")["_amt_log"].sum().reindex(node_df.index, fill_value=0)
    )
    node_df["in_amt_sum"] = (
        df.groupby("dst_id")["_amt_log"].sum().reindex(node_df.index, fill_value=0)
    )
    node_df["out_amt_mean"] = (
        df.groupby("src_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)
    )
    node_df["in_amt_mean"] = (
        df.groupby("dst_id")["_amt_log"].mean().reindex(node_df.index, fill_value=0)
    )
    # Degree ratio
    node_df["out_in_ratio"] = (
        node_df["out_count"] / (node_df["in_count"] + 1e-6)
    ).clip(upper=100.0)

    node_x = torch.tensor(
        node_df.fillna(0.0).values.astype("float32"), dtype=torch.float32
    )

    return edge_index, edge_attr, node_x, account_to_id, df


def pad_edge_attr(edge_attr: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Zero-pad or truncate edge features to match the trained model dimension."""
    current = edge_attr.shape[1]
    if current == expected_dim:
        return edge_attr
    if current < expected_dim:
        pad = torch.zeros(edge_attr.shape[0], expected_dim - current)
        return torch.cat([edge_attr, pad], dim=1)
    return edge_attr[:, :expected_dim]


@torch.no_grad()
def score_transactions(model, cfg, edge_index, edge_attr, node_x):
    """Run full-graph inference. Returns probability scores (numpy array)."""
    edge_attr = pad_edge_attr(edge_attr, cfg["edge_dim"])

    # Pad node features if needed
    in_dim = cfg["in_dim"]
    if node_x.shape[1] < in_dim:
        pad = torch.zeros(node_x.shape[0], in_dim - node_x.shape[1])
        node_x = torch.cat([node_x, pad], dim=1)
    elif node_x.shape[1] > in_dim:
        node_x = node_x[:, :in_dim]

    logits = model(node_x, edge_index, edge_attr)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    repo_id = st.text_input(
        "Hugging Face repo",
        value="",
        placeholder="waltertaya/aml-gnn-ibm-baseline-medium",
        help="Full repo ID as it appears on huggingface.co/models",
    )
    # Auto-read from Streamlit Cloud secrets if available
    _secret_token = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
    hf_token = st.text_input(
        "HF token (private repos)",
        type="password",
        placeholder="hf_... (or set via Streamlit secrets)",
        value=_secret_token,
        help="Leave blank for public repos. Set HF_TOKEN in Streamlit Cloud secrets to avoid entering it here.",
    )

    st.markdown("---")
    st.markdown("### Threshold")
    use_model_thr = st.checkbox("Use model's saved threshold", value=True)
    manual_thr = st.slider(
        "Manual threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        disabled=use_model_thr,
    )

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "GraphSAGE edge classifier trained on the IBM AML dataset. "
        "Each transaction is modelled as an edge in a financial graph. "
        "The model propagates 1-hop neighbourhood features to score each edge."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## AML Transaction Screening")
st.caption(
    "Upload a CSV of transactions or enter them manually. "
    "The model scores each transaction and flags potential suspicious activity."
)

if not repo_id:
    st.info("Enter a Hugging Face repo ID in the sidebar to get started.")
    st.stop()

# Load model
try:
    model, cfg, stored_metrics = load_model(repo_id, hf_token or None)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Determine threshold
threshold = cfg.get("best_threshold", 0.5) if use_model_thr else manual_thr

# ── Model info strip ──────────────────────────────────────────────────────────
with st.expander("Model details", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hidden dim", cfg.get("hidden_dim", "—"))
    col2.metric("Node feature dim", cfg.get("in_dim", "—"))
    col3.metric("Edge feature dim", cfg.get("edge_dim", "—"))
    col4.metric("Decision threshold", f"{threshold:.3f}")

    if stored_metrics.get("test_metrics"):
        tm = stored_metrics["test_metrics"]
        st.markdown("**Stored test metrics**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("F1", f"{tm.get('f1', 0):.4f}")
        mc2.metric("Precision", f"{tm.get('precision', 0):.4f}")
        mc3.metric("Recall", f"{tm.get('recall', 0):.4f}")
        mc4.metric("PR-AUC", f"{tm.get('pr_auc', 0):.4f}")

st.markdown("---")

# ── Input section ─────────────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs(["Upload CSV", "Manual entry"])

df_input = None

with tab_upload:
    st.markdown(
        "CSV must contain at minimum: **src** (sender account), **dst** (receiver account), "
        "**amount**. Optional columns: `hour`, `dayofweek`, `currency`, `payment_type`."
    )
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            # Normalise column names
            df_input.columns = [c.strip().lower().replace(" ", "_") for c in df_input.columns]

            # Column aliases → canonical names
            alias_map = {
                "from_account": "src", "originator": "src", "sender": "src",
                "to_account": "dst", "beneficiary": "dst", "receiver": "dst",
                "amount_paid": "amount", "transaction_amount": "amount", "amt": "amount",
                "timestamp": "hour", "step": "hour",
                "payment_format": "payment_type",
                "receiving_currency": "currency", "payment_currency": "currency",
            }
            df_input = df_input.rename(columns={k: v for k, v in alias_map.items() if k in df_input.columns})

            required = {"src", "dst", "amount"}
            missing = required - set(df_input.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
                df_input = None
            else:
                st.success(f"{len(df_input):,} transactions loaded.")
                st.dataframe(df_input.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

with tab_manual:
    st.markdown("Enter one transaction per row.")
    default_data = pd.DataFrame(
        {
            "src": ["ACC001", "ACC002", "ACC003", "ACC001"],
            "dst": ["ACC002", "ACC003", "ACC004", "ACC004"],
            "amount": [15000.0, 9500.0, 800.0, 250000.0],
            "currency": ["USD", "USD", "EUR", "USD"],
            "payment_type": ["WIRE", "WIRE", "CREDIT", "WIRE"],
        }
    )
    edited = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "src": st.column_config.TextColumn("Source account"),
            "dst": st.column_config.TextColumn("Destination account"),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "currency": st.column_config.TextColumn("Currency"),
            "payment_type": st.column_config.TextColumn("Payment type"),
        },
    )
    if st.button("Use this data", type="primary"):
        df_input = edited.copy()

# ── Inference ─────────────────────────────────────────────────────────────────
if df_input is not None and len(df_input) > 0:
    st.markdown("---")
    st.markdown("## Results")

    with st.spinner("Scoring transactions..."):
        try:
            edge_index, edge_attr, node_x, account_to_id, df_feat = build_features(df_input)
            probs = score_transactions(model, cfg, edge_index, edge_attr, node_x)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    result_df = df_input[["src", "dst", "amount"]].copy()
    for col in ["currency", "payment_type", "hour", "dayofweek"]:
        if col in df_input.columns:
            result_df[col] = df_input[col].values

    result_df["risk_score"] = probs
    result_df["flagged"] = (probs >= threshold).astype(bool)
    result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    n_total = len(result_df)
    n_flagged = int(result_df["flagged"].sum())
    n_clear = n_total - n_flagged
    max_score = float(result_df["risk_score"].max())

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{n_total:,}")
    m2.metric("Flagged", f"{n_flagged:,}", delta=None)
    m3.metric("Clear", f"{n_clear:,}")
    m4.metric("Highest risk score", f"{max_score:.4f}")

    # Status banner
    if n_flagged > 0:
        st.markdown(
            f'<div class="alert-high">'
            f"<strong>{n_flagged} transaction{'s' if n_flagged != 1 else ''} flagged</strong> "
            f"above the {threshold:.3f} threshold. Review the table below."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-clear">'
            "No transactions flagged at the current threshold."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Tabs: all vs flagged
    view_all, view_flagged = st.tabs(
        [f"All transactions ({n_total})", f"Flagged ({n_flagged})"]
    )

    def style_risk(val):
        if isinstance(val, float):
            if val >= threshold:
                return "background-color: #fff5f5; color: #c53030; font-weight: 600"
            elif val >= threshold * 0.7:
                return "background-color: #fffbeb; color: #b7791f"
        return ""

    def render_table(df):
        display = df.copy()
        display["risk_score"] = display["risk_score"].map("{:.4f}".format)
        display["flagged"] = display["flagged"].map(
            lambda x: "Yes" if x else "No"
        )
        display["amount"] = pd.to_numeric(display["amount"], errors="coerce").map(
            "{:,.2f}".format
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    with view_all:
        render_table(result_df)

    with view_flagged:
        if n_flagged > 0:
            render_table(result_df[result_df["flagged"]])
        else:
            st.info("No transactions were flagged at the current threshold.")

    # Risk distribution
    st.markdown("### Risk score distribution")
    hist_col, acct_col = st.columns([2, 1])

    with hist_col:
        import math
        bins = 20
        counts, edges = np.histogram(probs, bins=bins, range=(0, 1))
        hist_df = pd.DataFrame(
            {
                "bin": [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))],
                "count": counts,
            }
        )
        st.bar_chart(hist_df.set_index("bin"), use_container_width=True, height=220)

    with acct_col:
        st.markdown("**Top accounts by max risk**")
        acct = pd.concat(
            [
                result_df[["src", "risk_score"]].rename(columns={"src": "account"}),
                result_df[["dst", "risk_score"]].rename(columns={"dst": "account"}),
            ]
        )
        acct_summary = (
            acct.groupby("account")["risk_score"]
            .max()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        acct_summary.columns = ["Account", "Max risk"]
        acct_summary["Max risk"] = acct_summary["Max risk"].map("{:.4f}".format)
        st.dataframe(acct_summary, use_container_width=True, hide_index=True)

    # Download
    st.markdown("---")
    csv_bytes = result_df.to_csv(index=False).encode()
    st.download_button(
        label="Download scored transactions as CSV",
        data=csv_bytes,
        file_name="aml_scored_transactions.csv",
        mime="text/csv",
    )
