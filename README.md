# 🔍 Smart Credit Card Fraud Investigation Dashboard

An end-to-end machine learning system designed to detect fraudulent credit card transactions and provide transparent, explainable risk scores for financial investigators. 

Instead of treating the machine learning model as a "black box," this project pairs a highly optimized **XGBoost** classification pipeline with a live **Streamlit dashboard** that uses **SHAP** values to break down exactly *why* any given transaction was flagged.

## 🚀 Live Demo
🔗 **[View the Live Interactive Dashboard Here](YOUR_STREAMLIT_CLOUD_LINK_HERE)**

---

## 🛠️ The Tech Stack
* **Data Processing & Analytics:** Python, Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn (CalibratedClassifierCV)
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Frontend Dashboard:** Streamlit, Matplotlib

---

## 💡 System Architecture & Core Features

### 1. The Machine Learning Engine (`train.py`)
* **Time-Proxy Splitting:** To mimic a realistic production environment, data is sorted chronologically via transaction IDs and split (70/30) without shuffling, completely preventing future data leakage.
* **Imbalance Management:** Addresses severe fraud class imbalances using calculated dynamic positive instance weighting (`scale_pos_weight`).
* **Probability Calibration:** Smoothes out raw XGBoost output distortions using Sigmoid calibration, transforming raw outputs into reliable 0-100% risk probabilities.
* **Optimized Thresholding:** Uses Precision-Recall curves to mathematically isolate the ideal decision threshold—maximizing fraud catch rates (Recall) while minimizing expensive false alarms (Precision).

### 2. The Investigator UI (`app.py`)
* **Risk-Ranked Feeds:** Automatically bubbles the most critical, high-risk anomalies straight to the top of the investigator's queue.
* **Custom Drill-Downs:** Allows real-time filtering by minimum risk thresholds, specific merchants, or fraud-only flags.
* **Per-Transaction Deep Dives:** Selecting a transaction dynamically generates a local SHAP bar chart, showing the top 3 drivers (e.g., location mismatches, high transaction velocity, or unusual hours) behind that specific risk score.

---

## 📊 Performance Report Card
The pipeline automatically logs its performance metrics into `RESULTS.md` upon training. Our current benchmark:

* **PR-AUC:** ~0.87+ (High precision across fraud detection)
* **Inference Latency:** ~1.2ms (Well within standard production SLAs of <200ms)

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the repository and install dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
