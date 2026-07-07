import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Fraud Investigation Dashboard", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model_base.joblib")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer

@st.cache_data
def load_scored():
    return pd.read_csv("scored_transactions.csv")

model, feature_cols, explainer = load_artifacts()
df = load_scored()

st.title("🔍 Smart Credit Card Fraud Investigation Dashboard")
st.caption("Risk-ranked transactions with per-transaction explanations")

st.sidebar.header("Filters")
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 50)
show_only_flagged = st.sidebar.checkbox("Show only predicted fraud", value=False)
merchant_filter = st.sidebar.multiselect(
    "Merchant category",
    options=sorted(df["merchant_category"].unique()),
    default=[]
)

filtered = df[df["risk_score"] >= min_risk]
if show_only_flagged:
    filtered = filtered[filtered["predicted_fraud"] == 1]
if merchant_filter:
    filtered = filtered[filtered["merchant_category"].isin(merchant_filter)]

filtered = filtered.sort_values("risk_score", ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Transactions shown", len(filtered))
col2.metric("Flagged as fraud", int(filtered["predicted_fraud"].sum()))
col3.metric("Avg risk score", f"{filtered['risk_score'].mean():.1f}" if len(filtered) else "N/A")

st.subheader("Ranked Transactions")
display_cols = ["transaction_id", "amount", "merchant_category", "foreign_transaction",
                 "location_mismatch", "risk_score", "predicted_fraud", "top_reasons"]
st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    width="stretch",
    height=350
)

st.subheader("Investigate a Transaction")
row_options = filtered.index.tolist()

if row_options:
    selected_idx = st.selectbox("Select transaction row index", row_options)
    row = df.loc[selected_idx]

    st.write(f"**Transaction ID:** {row['transaction_id']}  |  **Amount:** ${row['amount']:.2f}")
    st.write(f"**Risk Score:** {row['risk_score']}/100  |  "
             f"**Predicted:** {'FRAUD' if row['predicted_fraud'] == 1 else 'Legitimate'}")
    st.write(f"**Merchant category:** {row['merchant_category']}  |  "
             f"**Foreign:** {row['foreign_transaction']}  |  "
             f"**Location mismatch:** {row['location_mismatch']}")
    st.write(f"**Top reasons:** {row['top_reasons']}")

    le = joblib.load("merchant_category_encoder.joblib")
    merchant_enc = le.transform([row["merchant_category"]])[0]

    X_row = pd.DataFrame([{
        "amount": row["amount"],
        "log_amount": np.log1p(row["amount"]),
        "transaction_hour": row["transaction_hour"],
        "merchant_category_enc": merchant_enc,
        "foreign_transaction": row["foreign_transaction"],
        "location_mismatch": row["location_mismatch"],
        "device_trust_score": row["device_trust_score"],
        "velocity_last_24h": row["velocity_last_24h"],
        "cardholder_age": row["cardholder_age"],
    }])[feature_cols]

    shap_values_row = explainer.shap_values(X_row)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values_row, X_row, feature_names=feature_cols,
                       plot_type="bar", show=False)
    st.pyplot(fig)
else:
    st.info("No transactions match the current filters.")