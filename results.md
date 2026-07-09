# PrivacyShield — Experiment Results

## Dataset
- IBM Telco Customer Churn
- Total records: 7043
- Features: 19
- Churn rate: 26.5%

## Model Performance (Privacy-Utility Tradeoff)

| Model | Epsilon (ε) | AUC-ROC | F1 Score | Accuracy |
|-------|-------------|---------|----------|----------|
| Baseline (No Privacy) | ∞ | 0.8330 | 0.6111 | 0.7814 |
| DP RandomForest | 0.1 | 0.7307 | 0.2222 | 0.7466 |
| DP RandomForest | 0.5 | 0.7861 | 0.2548 | 0.7509 |
| DP RandomForest | 1.0 | 0.7883 | 0.1040 | 0.7431 |
| DP RandomForest | 5.0 | 0.7988 | 0.0418 | 0.7395 |
| DP RandomForest | 10.0 | 0.8005 | 0.0418 | 0.7395 |

## Membership Inference Attack Results

| Model | Attack AUC | Train Conf | Test Conf | Gap |
|-------|-----------|------------|-----------|-----|
| Baseline | 0.5672 | 0.8403 | 0.8053 | 0.035 |
| DP ε=0.1 | 0.5068 | 0.5915 | 0.5905 | 0.0011 |
| DP ε=0.5 | 0.5044 | 0.6914 | 0.6899 | 0.0015 |
| DP ε=1.0 | 0.5086 | 0.7418 | 0.7378 | 0.0039 |
| DP ε=5.0 | 0.505 | 0.7768 | 0.7741 | 0.0027 |
| DP ε=10.0 | 0.5048 | 0.7786 | 0.7761 | 0.0025 |

## Key Findings
- Recommended epsilon: 1.0 (best privacy-utility balance)
- AUC cost of privacy at ε=1.0: -0.0447 (5.4% reduction)
- Attack AUC reduction at ε=1.0: -0.0586
- At ε=1.0, attack AUC = 0.5086 (near random guessing = 0.50)
