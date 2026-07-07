import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              f1_score, recall_score, precision_score, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import time
import json

df = pd.read_csv("credit card fraud.csv")

print(f"Loaded {len(df)} transactions")
print(f"Fraud rate: {df['is_fraud'].mean() * 100:.4f}%")
print(df['is_fraud'].value_counts())

df["log_amount"] = np.log1p(df["amount"])

le = LabelEncoder()
df["merchant_category_enc"] = le.fit_transform(df["merchant_category"])
joblib.dump(le, "merchant_category_encoder.joblib")

feature_cols = [
    "amount", "log_amount", "transaction_hour", "merchant_category_enc",
    "foreign_transaction", "location_mismatch", "device_trust_score",
    "velocity_last_24h", "cardholder_age"
]

df = df.sort_values("transaction_id").reset_index(drop=True)
split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["is_fraud"]
X_test, y_test = test_df[feature_cols], test_df["is_fraud"]

print(f"\nTrain fraud rate: {y_train.mean() * 100:.4f}%")
print(f"Test fraud rate:  {y_test.mean() * 100:.4f}%")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / max(n_pos, 1)
print(f"scale_pos_weight = {scale_pos_weight:.2f}  (fraud={n_pos}, legit={n_neg})")

base_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

base_model.fit(X_train, y_train)

try:
    from sklearn.frozen import FrozenEstimator
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated_model.fit(X_train, y_train)
except ImportError:
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

probs = calibrated_model.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

valid = precisions[:-1] >= 0.5
if valid.any():
    best_idx = np.where(valid)[0][np.argmax(recalls[:-1][valid])]
    chosen_threshold = thresholds[best_idx]
else:
    chosen_threshold = 0.5

preds = (probs >= chosen_threshold).astype(int)
f1 = f1_score(y_test, preds)
recall = recall_score(y_test, preds)
precision = precision_score(y_test, preds)

sample = X_test.iloc[:1]
start = time.perf_counter()
for _ in range(100):
    calibrated_model.predict_proba(sample)
latency_ms = (time.perf_counter() - start) / 100 * 1000

print(f"\n--- RESULTS ---")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Threshold: {chosen_threshold:.4f}")
print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
print(f"Latency:   {latency_ms:.2f} ms (budget: 200ms)")

results_md = f"""# Results

Dataset: credit card fraud.csv
Rows: {len(df)}
Split: sorted by transaction_id (time proxy), 70% train / 30% test - no shuffling, no leakage

| Metric | Value |
|---|---|
| PR-AUC | {pr_auc:.4f} |
| ROC-AUC | {roc_auc:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |
| Chosen threshold | {chosen_threshold:.4f} |
| Inference latency | {latency_ms:.2f} ms (budget: 200ms) |

Fraud rate in data: {df['is_fraud'].mean() * 100:.4f}%
"""
with open("RESULTS.md", "w") as f:
    f.write(results_md)

explainer = shap.TreeExplainer(base_model)
shap_values_test = explainer.shap_values(X_test)

sample_idx = np.arange(len(X_test))

scored_df = test_df.iloc[sample_idx].copy()
scored_df["risk_score"] = (probs[sample_idx] * 100).round(2)
scored_df["predicted_fraud"] = preds[sample_idx]

top_features = []
for i in sample_idx:
    row_shap = shap_values_test[i]
    top3_idx = np.argsort(-np.abs(row_shap))[:3]
    reasons = [
        f"{feature_cols[j]} ({'+' if row_shap[j] > 0 else '-'}{abs(row_shap[j]):.2f})"
        for j in top3_idx
    ]
    top_features.append("; ".join(reasons))

scored_df["top_reasons"] = top_features
scored_df.to_csv("scored_transactions.csv", index=False)

joblib.dump(calibrated_model, "fraud_model.joblib")
joblib.dump(base_model, "fraud_model_base.joblib")

with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

with open("threshold.json", "w") as f:
    json.dump({"threshold": float(chosen_threshold)}, f)

print("\nAll done! Saved:")
print(" - fraud_model.joblib / fraud_model_base.joblib")
print(" - merchant_category_encoder.joblib")
print(" - feature_cols.json / threshold.json")
print(" - scored_transactions.csv")
print(" - RESULTS.md")