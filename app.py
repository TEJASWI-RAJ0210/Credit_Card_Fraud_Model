import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("xgb_credit_fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Build SHAP explainer once (cached) ────────────────────────────────────────
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# ── Feature list (30 features: V1-V28 + Amount_scaled + Time_scaled) ──────────
V_FEATURES   = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = V_FEATURES + ["Amount_scaled", "Time_scaled"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detector")
st.markdown(
    "Enter transaction details below. The model will predict the **fraud probability** "
    "and explain *why* using SHAP."
)
st.divider()

# ── Sidebar — V feature inputs ─────────────────────────────────────────────────
st.sidebar.header("V Features (PCA components)")
st.sidebar.caption("These are anonymised PCA features from the original dataset.")

v_values = {}
for feat in V_FEATURES:
    v_values[feat] = st.sidebar.number_input(
        label=feat,
        value=0.0,
        format="%.4f",
        step=0.0001,
        key=feat,
    )

# ── Main — Amount and Time inputs ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction details")
    amount_raw = st.number_input(
        "Transaction amount (€)",
        min_value=0.0,
        max_value=25000.0,
        value=50.0,
        step=0.01,
        format="%.2f",
    )
    time_raw = st.number_input(
        "Time (seconds since first transaction)",
        min_value=0,
        max_value=200000,
        value=50000,
        step=1,
    )

    # Scale Amount and Time the same way training did (StandardScaler μ/σ from dataset)
    # Creditcard.csv known statistics:
    AMOUNT_MEAN, AMOUNT_STD = 88.35, 250.12
    TIME_MEAN,   TIME_STD   = 94813.86, 47488.15

    amount_scaled = (amount_raw - AMOUNT_MEAN) / AMOUNT_STD
    time_scaled   = (time_raw   - TIME_MEAN)   / TIME_STD

    st.caption(
        f"Scaled → Amount: `{amount_scaled:.4f}`  |  Time: `{time_scaled:.4f}`"
    )

# ── Build input dataframe ──────────────────────────────────────────────────────
input_dict = {**v_values, "Amount_scaled": amount_scaled, "Time_scaled": time_scaled}
input_df   = pd.DataFrame([input_dict])[ALL_FEATURES]

# ── Predict button ─────────────────────────────────────────────────────────────
with col2:
    st.subheader("Prediction")
    predict_btn = st.button("Analyse transaction", type="primary", use_container_width=True)

    if predict_btn:
        prob      = model.predict_proba(input_df)[0][1]
        threshold = 0.5
        is_fraud  = prob >= threshold

        # Score gauge
        if is_fraud:
            st.error(f"🚨 **FRAUD DETECTED**  —  Probability: `{prob:.2%}`")
        else:
            st.success(f"✅ **LEGITIMATE**  —  Fraud probability: `{prob:.2%}`")

        st.progress(float(prob), text=f"Fraud risk: {prob:.2%}")

        st.markdown("---")

        # Risk breakdown
        st.markdown("**Risk level**")
        if prob < 0.2:
            st.info("🟢 Low risk — very likely legitimate")
        elif prob < 0.5:
            st.warning("🟡 Moderate risk — review recommended")
        elif prob < 0.8:
            st.warning("🟠 High risk — flag for manual review")
        else:
            st.error("🔴 Critical risk — block transaction")

# ── SHAP explanation ───────────────────────────────────────────────────────────
st.divider()
st.subheader("SHAP Explanation — why did the model decide this?")

if not predict_btn:
    st.info("Click **Analyse transaction** above to see the SHAP explanation.")
else:
    with st.spinner("Computing SHAP values..."):
        shap_values = explainer(input_df)

    tab1, tab2 = st.tabs(["Waterfall plot", "Feature contributions table"])

    with tab1:
        st.caption(
            "Each bar shows how much a feature **pushed the prediction** toward fraud (red ↑) "
            "or toward legitimate (blue ↓). The bottom value is the model baseline; "
            "the top is the final fraud probability."
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        shap_df = pd.DataFrame({
            "Feature":          ALL_FEATURES,
            "Input value":      input_df.iloc[0].values,
            "SHAP value":       shap_values.values[0],
            "Direction":        ["↑ Fraud" if v > 0 else "↓ Legit"
                                 for v in shap_values.values[0]],
        })
        shap_df["Abs impact"] = shap_df["SHAP value"].abs()
        shap_df = shap_df.sort_values("Abs impact", ascending=False).reset_index(drop=True)

        st.dataframe(
            shap_df[["Feature", "Input value", "SHAP value", "Direction", "Abs impact"]],
            use_container_width=True,
            hide_index=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost trained on the Kaggle Credit Card Fraud dataset (ULB).  "
    "Explanations powered by SHAP TreeExplainer.  "
    "Threshold: 0.5 — adjust based on business risk tolerance."
)