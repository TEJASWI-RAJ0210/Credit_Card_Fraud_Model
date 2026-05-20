import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1.5rem 2rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .fraud-box {
        background: #fff5f5;
        border: 1.5px solid #fc8181;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .legit-box {
        background: #f0fff4;
        border: 1.5px solid #68d391;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Feature config ─────────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "V14": {
        "label":   "Authentication Pattern",
        "help":    "Derived from transaction authentication behaviour. "
                   "Lower values indicate unusual auth patterns linked to fraud.",
        "default": 0.0, "min": -20.0, "max": 10.0,
    },
    "V17": {
        "label":   "Spending Velocity",
        "help":    "Reflects how rapidly money is being spent relative to account history. "
                   "Sudden spikes are a strong fraud signal.",
        "default": 0.0, "min": -20.0, "max": 10.0,
    },
    "V12": {
        "label":   "Merchant Category Deviation",
        "help":    "How much this transaction deviates from the cardholder's "
                   "typical merchant categories.",
        "default": 0.0, "min": -20.0, "max": 10.0,
    },
    "V10": {
        "label":   "Geographic Anomaly Score",
        "help":    "Captures geographic inconsistencies — e.g. a card used in "
                   "two distant locations within minutes.",
        "default": 0.0, "min": -20.0, "max": 10.0,
    },
    "V4": {
        "label":   "Card Usage Frequency",
        "help":    "Reflects how frequently the card is being used compared to "
                   "historical patterns. Unusually high = suspicious.",
        "default": 0.0, "min": -10.0, "max": 20.0,
    },
    "V11": {
        "label":   "Time-of-Day Behaviour",
        "help":    "Captures whether the transaction time matches the cardholder's "
                   "normal activity hours.",
        "default": 0.0, "min": -10.0, "max": 20.0,
    },
    "Amount_scaled": {
        "label":   "Transaction Amount (€)",
        "help":    "The actual euro value of the transaction.",
        "default": 50.0, "min": 0.0, "max": 25000.0,
    },
}

MODEL_FEATURES = list(FEATURE_CONFIG.keys())
LABEL_MAP      = {k: v["label"] for k, v in FEATURE_CONFIG.items()}

# ── Load model & explainer ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("xgb_credit_fraud_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model     = load_model()
explainer = load_explainer(model)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detector")
st.markdown(
    "Fill in the transaction details below. The model analyses **6 behavioural signals** "
    "and returns a real-time fraud risk score with a full SHAP explanation."
)
st.caption(
    "ℹ️ Features are derived from anonymised PCA components of real bank transaction data "
    "(Kaggle ULB dataset). Renamed for readability — not official bank terminology."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
inputs = {}

for i, (feat, cfg) in enumerate(FEATURE_CONFIG.items()):
    target_col = [col1, col2, col3][i % 3]
    with target_col:
        if feat == "Amount_scaled":
            inputs[feat] = st.number_input(
                label=cfg["label"],
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=0.01,
                format="%.2f",
                help=cfg["help"],
            )
        else:
            inputs[feat] = st.slider(
                label=cfg["label"],
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=0.01,
                help=cfg["help"],
            )

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([inputs])[MODEL_FEATURES]
    prob     = model.predict_proba(input_df)[0][1]
    is_fraud = prob >= 0.5

    st.divider()

    res_col, gauge_col, summary_col = st.columns([1.2, 1.5, 1.3])

    # Verdict card
    with res_col:
        st.markdown('<p class="section-header">Verdict</p>', unsafe_allow_html=True)
        if is_fraud:
            st.markdown(f"""
            <div class="fraud-box">
                <h2 style="color:#c53030;margin:0">🚨 FRAUD</h2>
                <p style="font-size:2rem;font-weight:700;color:#c53030;margin:0.3rem 0">
                    {prob:.1%}</p>
                <p style="color:#742a2a;margin:0">fraud probability</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-box">
                <h2 style="color:#276749;margin:0">✅ LEGITIMATE</h2>
                <p style="font-size:2rem;font-weight:700;color:#276749;margin:0.3rem 0">
                    {prob:.1%}</p>
                <p style="color:#22543d;margin:0">fraud probability</p>
            </div>""", unsafe_allow_html=True)

    # Risk meter
    with gauge_col:
        st.markdown('<p class="section-header">Risk Level</p>', unsafe_allow_html=True)
        st.progress(float(prob))

        if prob < 0.2:
            st.info("🟢 **Low risk** — Transaction appears normal.")
        elif prob < 0.5:
            st.warning("🟡 **Moderate risk** — Some unusual signals detected.")
        elif prob < 0.8:
            st.warning("🟠 **High risk** — Flag for manual review.")
        else:
            st.error("🔴 **Critical risk** — Block this transaction.")

    # Input summary table
    with summary_col:
        st.markdown('<p class="section-header">Input Summary</p>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Feature": [FEATURE_CONFIG[f]["label"] for f in MODEL_FEATURES],
            "Value":   [f"€{inputs[f]:.2f}" if f == "Amount_scaled"
                        else f"{inputs[f]:.3f}" for f in MODEL_FEATURES],
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ── SHAP explanation ───────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Why did the model decide this?</p>',
                unsafe_allow_html=True)
    st.caption(
        "🔴 Red bars push toward **fraud** &nbsp;|&nbsp; "
        "🔵 Blue bars push toward **legitimate** &nbsp;|&nbsp; "
        "Longer bar = stronger influence on the decision."
    )

    with st.spinner("Computing SHAP explanation..."):
        shap_values = explainer(input_df)

    shap_values.feature_names = [LABEL_MAP[f] for f in MODEL_FEATURES]

    tab1, tab2 = st.tabs(["📊 Waterfall Plot", "📋 Contributions Table"])

    with tab1:
        fig, _ = plt.subplots(figsize=(9, 4))
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            f"Base value = model's average fraud rate across all training transactions. "
            f"Output value = fraud probability for this transaction = **{prob:.4f}**"
        )

    with tab2:
        contrib_df = pd.DataFrame({
            "Feature":           [LABEL_MAP[f] for f in MODEL_FEATURES],
            "Your Input":        [f"€{inputs[f]:.2f}" if f == "Amount_scaled"
                                   else f"{inputs[f]:.4f}" for f in MODEL_FEATURES],
            "SHAP Contribution": [f"{v:+.4f}" for v in shap_values.values[0]],
            "Effect":            ["↑ Toward Fraud" if v > 0 else "↓ Toward Legit"
                                   for v in shap_values.values[0]],
            "Abs Impact":        [abs(v) for v in shap_values.values[0]],
        })
        contrib_df = (contrib_df
                      .sort_values("Abs Impact", ascending=False)
                      .reset_index(drop=True)
                      .drop(columns="Abs Impact"))
        st.dataframe(contrib_df, hide_index=True, use_container_width=True)

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#a0aec0">
        <p style="font-size:3rem;margin:0">🛡️</p>
        <p style="font-size:1.1rem;margin:0.5rem 0">
            Fill in the transaction details above and click
            <strong>Analyse Transaction</strong>
        </p>
        <p style="font-size:0.85rem">
            You'll get a fraud probability score + full SHAP explanation
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost retrained on top 6 features (V14, V17, V12, V10, V4, V11) + Amount_scaled  |  "
    "PR-AUC: 0.78  |  Explanations: SHAP TreeExplainer  |  "
    "Dataset: Kaggle Credit Card Fraud Detection (ULB, 284,807 transactions)"
)