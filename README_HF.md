---
title: PrivacyShield
emoji: 🔒
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
---

# PrivacyShield — Differential Privacy for ML

Demonstrates differential privacy applied to customer churn 
prediction with membership inference attack analysis.

**Live Results:**
- Baseline model AUC: 0.833
- DP model AUC (ε=1.0): 0.788  
- Attack AUC baseline: 0.567 (privacy leak)
- Attack AUC with DP: 0.509 (protected)

**Related project:** 
[Churn Prediction MLOps System](https://nityagangver-churnpredictor.hf.space)
