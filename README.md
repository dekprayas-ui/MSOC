# 🔍 Team [Your Team Name] | Smart Credit Card Fraud Investigation Dashboard

Submitted for [Hackathon Name 2026]

## 🚀 Live Interactive Demo
🔗 **[Click Here to Launch the Live Dashboard](YOUR_STREAMLIT_CLOUD_LINK_HERE)**

---

## 💡 The Problem
Financial institutions lose billions annually to credit card fraud. Traditional AI detection models operate as "black boxes"—they flag transactions but leave human investigators guessing *why* a charge was blocked. This slows down investigations and hurts user experience through false positives.

## 🛠️ Our Solution
We built an **End-to-End Explainable AI (XAI) Fraud Detection System**. 
1. **The Core Engine:** Uses a calibrated **XGBoost Classifier** optimized for highly imbalanced data to generate reliable, precise risk percentages (0-100%).
2. **The Transparency Layer:** Integrates **SHAP (SHapley Additive exPlanations)** to break open the black box. 
3. **The Investigator Dashboard:** A live **Streamlit** dashboard that automatically prioritizes high-risk threats and provides instant, localized visual breakdowns of the top 3 risk drivers for any selected transaction.

---

## 📈 Key Highlights & Tech Achievements
* **Explainable Risk Scores:** Investigators instantly see precisely why a transaction is flagged (e.g., location mismatch, high transaction velocity, or unusual hours).
* **Ultra-Low Latency:** The model boasts an inference speed of **~1.2ms**, making it fully viable for real-time production banking systems (well within standard <200ms budgets).
* **Calibrated Confidence:** Uses probability calibration so that a "80% risk score" mathematically correlates to a true 80% likelihood of fraud.

---

## 🏗️ Built With
* **Backend & ML Engine:** Python, XGBoost, Scikit-Learn
* **Explainable AI:** SHAP
* **Frontend UI:** Streamlit, Matplotlib
