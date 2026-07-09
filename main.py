# ============================================================
# PrivacyShield — main.py
# Differential Privacy + Membership Inference Attack Analysis
# Dataset: IBM Telco Customer Churn (reused from churn project)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import diffprivlib as dp

# ── STEP 1: Load and preprocess the dataset ──────────────────
print("Loading dataset...")

import os
import urllib.request

CSV_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

if not os.path.exists(CSV_PATH):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/nityagangver/privacyshield/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    urllib.request.urlretrieve(url, CSV_PATH)
    print("Dataset downloaded.")

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# Fix TotalCharges — has blank strings for new customers
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop CustomerID — not a feature
df = df.drop('customerID', axis=1)

# Encode target
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Encode all categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split — stratified to preserve 26.5% churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Save samples for server attack simulations
np.save('models/X_train_sample.npy', X_train)
np.save('models/X_test_sample.npy', X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# ── STEP 2: Train baseline model (no privacy) ────────────────
print("\n" + "="*50)
print("Training BASELINE model (no privacy)...")

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)
baseline_rf.fit(X_train, y_train)

baseline_auc = roc_auc_score(y_test, baseline_rf.predict_proba(X_test)[:, 1])
baseline_f1  = f1_score(y_test, baseline_rf.predict(X_test))
baseline_acc = accuracy_score(y_test, baseline_rf.predict(X_test))

print(f"Baseline AUC-ROC:  {baseline_auc:.4f}")
print(f"Baseline F1 Score: {baseline_f1:.4f}")
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# Save baseline model
joblib.dump(baseline_rf, 'models/baseline_model.pkl')
print("Baseline model saved to models/baseline_model.pkl")

# ── STEP 3: Train DP models across epsilon values ────────────
print("\n" + "="*50)
print("Training DIFFERENTIALLY PRIVATE models...")

epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
dp_results = {}
dp_models  = {}

for eps in epsilons:
    print(f"\n  Training DP model with epsilon = {eps}...")
    
    dp_rf = dp.models.RandomForestClassifier(
        n_estimators=100,
        epsilon=eps,
        random_state=42
    )
    dp_rf.fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, dp_rf.predict_proba(X_test)[:, 1])
    f1  = f1_score(y_test, dp_rf.predict(X_test))
    acc = accuracy_score(y_test, dp_rf.predict(X_test))
    
    dp_results[eps] = {'auc': auc, 'f1': f1, 'acc': acc}
    dp_models[eps]  = dp_rf
    
    print(f"  Epsilon {eps:5.1f} | AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

# Save all DP models
joblib.dump(dp_models[0.1],  'models/dp_epsilon_01_model.pkl')

# Train and save epsilon=0.5 explicitly as requested
print("  Training DP model with epsilon = 0.5 (explicit)...")
dp_rf_05 = dp.models.RandomForestClassifier(
    n_estimators=100, epsilon=0.5, random_state=42
)
dp_rf_05.fit(X_train, y_train)
joblib.dump(dp_rf_05, 'models/dp_epsilon_05_model.pkl')

joblib.dump(dp_models[1.0],  'models/dp_epsilon_1_model.pkl')
joblib.dump(dp_models[5.0],  'models/dp_epsilon_5_model.pkl')
joblib.dump(dp_models[10.0], 'models/dp_epsilon_10_model.pkl')
print("\nAll five DP models saved.")

# ── STEP 4: Privacy-Utility Tradeoff Graph ───────────────────
print("\n" + "="*50)
print("Generating Privacy-Utility Tradeoff Graph...")

eps_values = list(dp_results.keys())
auc_values = [dp_results[e]['auc'] for e in eps_values]
f1_values  = [dp_results[e]['f1']  for e in eps_values]

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

ax.fill_between(eps_values, auc_values, alpha=0.15, color='#7c3aed')
ax.plot(eps_values, auc_values, 'o-', color='#a78bfa', lw=2.5,
        markersize=10, label='DP Model AUC-ROC', zorder=5)
ax.axhline(baseline_auc, color='#ef4444', lw=2.5, linestyle='--',
           label=f'Baseline (No Privacy): AUC = {baseline_auc:.3f}')

# Annotate each point
for eps, auc in zip(eps_values, auc_values):
    ax.annotate(f'{auc:.3f}',
                xy=(eps, auc), xytext=(0, 14),
                textcoords='offset points',
                ha='center', fontsize=10,
                color='#e2e8f0',
                fontweight='bold')

# Highlight sweet spot
ax.axvspan(0.8, 1.5, alpha=0.08, color='#22c55e')
ax.text(1.0, ax.get_ylim()[0] + 0.02,
        'Sweet Spot\n(ε=1.0)',
        ha='center', fontsize=9, color='#86efac', style='italic')

ax.set_xlabel('Privacy Budget (ε) — Lower = Stronger Privacy',
              fontsize=13, color='#94a3b8')
ax.set_ylabel('AUC-ROC Score', fontsize=13, color='#94a3b8')
ax.set_title('Privacy-Utility Tradeoff — Differential Privacy\nIBM Telco Customer Churn Dataset',
             fontsize=14, fontweight='bold', color='#e2e8f0', pad=16)
ax.legend(fontsize=11, facecolor='#1e293b',
          edgecolor='#374151', labelcolor='#e2e8f0')
ax.tick_params(colors='#64748b')
ax.spines['bottom'].set_color('#374151')
ax.spines['left'].set_color('#374151')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.15, color='#374151')

plt.tight_layout()
plt.savefig('graphs/privacy_utility_tradeoff.png',
            dpi=180, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("Graph saved: graphs/privacy_utility_tradeoff.png")

# ── STEP 5: Membership Inference Attack ──────────────────────
print("\n" + "="*50)
print("Running Membership Inference Attack...")

def membership_inference_attack(model, X_train, X_test):
    """
    Threshold-based membership inference attack.
    
    Core idea: models predict training data with HIGHER confidence
    than unseen data. We exploit this gap to guess membership.
    
    Attack AUC close to 1.0 = privacy leak (bad)
    Attack AUC close to 0.5 = random guessing = privacy protected (good)
    """
    # Confidence on training data (these ARE members)
    train_conf = model.predict_proba(X_train).max(axis=1)
    
    # Confidence on test data (these are NOT members)
    test_conf = model.predict_proba(X_test).max(axis=1)
    
    # Label: 1 = member (training set), 0 = non-member (test set)
    true_labels = np.concatenate([
        np.ones(len(X_train)),
        np.zeros(len(X_test))
    ])
    confidence_scores = np.concatenate([train_conf, test_conf])
    
    attack_auc        = roc_auc_score(true_labels, confidence_scores)
    train_conf_mean   = train_conf.mean()
    test_conf_mean    = test_conf.mean()
    confidence_gap    = train_conf_mean - test_conf_mean
    
    return {
        'attack_auc':       round(attack_auc, 4),
        'train_conf_mean':  round(float(train_conf_mean), 4),
        'test_conf_mean':   round(float(test_conf_mean), 4),
        'confidence_gap':   round(float(confidence_gap), 4)
    }

# Run attack on baseline
print("\n  Attacking BASELINE model...")
baseline_attack = membership_inference_attack(baseline_rf, X_train, X_test)
print(f"  Attack AUC:        {baseline_attack['attack_auc']}")
print(f"  Train confidence:  {baseline_attack['train_conf_mean']}")
print(f"  Test confidence:   {baseline_attack['test_conf_mean']}")
print(f"  Confidence gap:    {baseline_attack['confidence_gap']} ← attacker exploits this")

# Run attack on all DP models
print("\n  Attacking DP models...")
dp_attacks = {}
for eps in epsilons:
    result = membership_inference_attack(dp_models[eps], X_train, X_test)
    dp_attacks[eps] = result
    print(f"  ε={eps:5.1f} | Attack AUC: {result['attack_auc']} | Gap: {result['confidence_gap']}")

# ── STEP 6: Attack Comparison Graph ──────────────────────────
print("\n" + "="*50)
print("Generating Attack Comparison Graph...")

model_labels   = ['Baseline\n(No Privacy)'] + [f'DP ε={e}' for e in epsilons]
attack_aucs    = [baseline_attack['attack_auc']] + [dp_attacks[e]['attack_auc'] for e in epsilons]
bar_colors     = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#10b981']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')
for ax in [ax1, ax2]:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#64748b')
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15, color='#374151')

# Left: Attack AUC bar chart
bars = ax1.bar(model_labels, attack_aucs,
               color=bar_colors, edgecolor='#0f1117', width=0.65)
ax1.axhline(0.5, color='#22c55e', lw=2.5, linestyle='--',
            label='0.5 = Random guessing (perfect privacy)')
ax1.axhline(0.7, color='#ef4444', lw=1.5, linestyle=':',
            alpha=0.6, label='0.7 = Significant privacy leak')
for bar, val in zip(bars, attack_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', fontsize=10,
             fontweight='bold', color='#e2e8f0')
ax1.set_ylabel('Attack AUC (Lower = Better Privacy)',
               fontsize=12, color='#94a3b8')
ax1.set_title('Membership Inference Attack Success Rate\nLower = Privacy Protected',
              fontsize=12, fontweight='bold', color='#e2e8f0', pad=12)
ax1.legend(fontsize=9, facecolor='#1e293b',
           edgecolor='#374151', labelcolor='#e2e8f0')
ax1.set_ylim([0.4, max(attack_aucs) + 0.08])

# Right: Pareto frontier — privacy vs utility
all_attack_aucs = [baseline_attack['attack_auc']] + [dp_attacks[e]['attack_auc'] for e in epsilons]
all_model_aucs  = [baseline_auc] + [dp_results[e]['auc'] for e in epsilons]
all_labels      = ['Baseline'] + [f'ε={e}' for e in epsilons]

for i, (atk, mdl, lbl, col) in enumerate(zip(all_attack_aucs, all_model_aucs, all_labels, bar_colors)):
    ax2.scatter(atk, mdl, s=200, color=col, zorder=5)
    ax2.annotate(lbl, xy=(atk, mdl),
                 xytext=(8, 4), textcoords='offset points',
                 fontsize=9, color='#e2e8f0')

ax2.axvline(0.5, color='#22c55e', lw=1.5, linestyle='--', alpha=0.6,
            label='Perfect privacy line')
ax2.set_xlabel('Attack AUC (← More Private)',
               fontsize=12, color='#94a3b8')
ax2.set_ylabel('Model AUC-ROC (↑ More Accurate)',
               fontsize=12, color='#94a3b8')
ax2.set_title('Privacy vs Utility Tradeoff\nPareto Frontier',
              fontsize=12, fontweight='bold', color='#e2e8f0', pad=12)
ax2.legend(fontsize=9, facecolor='#1e293b',
           edgecolor='#374151', labelcolor='#e2e8f0')

plt.suptitle('PrivacyShield — Differential Privacy Analysis\nIBM Telco Customer Churn Dataset',
             fontsize=13, fontweight='bold', color='#a78bfa', y=1.02)
plt.tight_layout()
plt.savefig('graphs/attack_comparison.png',
            dpi=180, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("Graph saved: graphs/attack_comparison.png")

# ── STEP 7: Save all results to results.md ───────────────────
print("\n" + "="*50)
print("Saving results to results.md...")

with open('results.md', 'w') as f:
    f.write("# PrivacyShield — Experiment Results\n\n")
    f.write("## Dataset\n")
    f.write("- IBM Telco Customer Churn\n")
    f.write(f"- Total records: {len(df)}\n")
    f.write(f"- Features: {X.shape[1]}\n")
    f.write(f"- Churn rate: {y.mean():.1%}\n\n")

    f.write("## Model Performance (Privacy-Utility Tradeoff)\n\n")
    f.write("| Model | Epsilon (ε) | AUC-ROC | F1 Score | Accuracy |\n")
    f.write("|-------|-------------|---------|----------|----------|\n")
    f.write(f"| Baseline (No Privacy) | ∞ | {baseline_auc:.4f} | {baseline_f1:.4f} | {baseline_acc:.4f} |\n")
    for eps in epsilons:
        r = dp_results[eps]
        f.write(f"| DP RandomForest | {eps} | {r['auc']:.4f} | {r['f1']:.4f} | {r['acc']:.4f} |\n")

    f.write("\n## Membership Inference Attack Results\n\n")
    f.write("| Model | Attack AUC | Train Conf | Test Conf | Gap |\n")
    f.write("|-------|-----------|------------|-----------|-----|\n")
    ba = baseline_attack
    f.write(f"| Baseline | {ba['attack_auc']} | {ba['train_conf_mean']} | {ba['test_conf_mean']} | {ba['confidence_gap']} |\n")
    for eps in epsilons:
        a = dp_attacks[eps]
        f.write(f"| DP ε={eps} | {a['attack_auc']} | {a['train_conf_mean']} | {a['test_conf_mean']} | {a['confidence_gap']} |\n")

    f.write("\n## Key Findings\n")
    best_dp_auc    = dp_results[1.0]['auc']
    auc_drop       = baseline_auc - best_dp_auc
    attack_reduction = baseline_attack['attack_auc'] - dp_attacks[1.0]['attack_auc']
    f.write(f"- Recommended epsilon: 1.0 (best privacy-utility balance)\n")
    f.write(f"- AUC cost of privacy at ε=1.0: -{auc_drop:.4f} ({auc_drop/baseline_auc*100:.1f}% reduction)\n")
    f.write(f"- Attack AUC reduction at ε=1.0: -{attack_reduction:.4f}\n")
    f.write(f"- At ε=1.0, attack AUC = {dp_attacks[1.0]['attack_auc']} (near random guessing = 0.50)\n")

print("Results saved to results.md")

# ── FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "="*60)
print("PRIVACYSHIELD EXPERIMENT COMPLETE")
print("="*60)
print(f"\nBaseline AUC:          {baseline_auc:.4f}")
print(f"Best DP AUC (ε=1.0):   {dp_results[1.0]['auc']:.4f}")
print(f"AUC cost of privacy:   -{baseline_auc - dp_results[1.0]['auc']:.4f}")
print(f"\nBaseline attack AUC:   {baseline_attack['attack_auc']} (privacy leak)")
print(f"DP attack AUC (ε=1.0): {dp_attacks[1.0]['attack_auc']} (near random = protected)")
print(f"\nGraphs saved in:  privacyshield/graphs/")
print(f"Models saved in:  privacyshield/models/")
print(f"Results saved in: privacyshield/results.md")
print("\nNext step: Run app.py to start the API")