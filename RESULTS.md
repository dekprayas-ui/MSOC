# Results

Dataset: credit card fraud.csv
Rows: 10000
Split: sorted by transaction_id (time proxy), 70% train / 30% test - no shuffling, no leakage

| Metric | Value |
|---|---|
| PR-AUC | 0.9901 |
| ROC-AUC | 0.9998 |
| Precision | 0.5000 |
| Recall | 1.0000 |
| F1 | 0.6667 |
| Chosen threshold | 0.0007 |
| Inference latency | 3.80 ms (budget: 200ms) |

Fraud rate in data: 1.5100%
