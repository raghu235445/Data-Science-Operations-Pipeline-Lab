# =============================================================================
#   Mobile Money Scam Dataset — Full Data Science Analysis
#   Course: Data Science in Python | Continuous Assessment (CA)
# =============================================================================
#
#   HOW TO RUN:
#   1. Install dependencies:   pip install -r requirements.txt
#   2. Put your CSV file in the same folder as this script
#   3. Update DATASET_PATH below to match your filename
#   4. Run:  python mobile_money_analysis.py
#
#   All plots will be shown on screen AND saved to a folder called "plots/"
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, f1_score, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ── CONFIGURATION — update this path to your CSV file ─────────────────────────
DATASET_PATH = "MobileMoney_Scam_.csv"   # <-- change if your filename is different
PLOTS_DIR    = "plots"                    # folder where plots will be saved
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = "#2E86AB"
sns.set_theme(style="whitegrid", font_scale=1.1)

def save_and_show(filename, title=""):
    """Save the current figure to PLOTS_DIR and display it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {path}")
    plt.show()
    plt.close()


# =============================================================================
# STEP 0 — LOAD DATA
# =============================================================================
print("=" * 65)
print("STEP 0: Loading dataset")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn data types:")
print(df.dtypes.to_string())
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nMissing values per column:")
print(df.isnull().sum().to_string())
print(f"\nTarget variable (Victim) distribution:")
print(df['Victim'].value_counts(dropna=False).to_string())


# =============================================================================
# STEP 1 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 1: Exploratory Data Analysis (EDA)")
print("=" * 65)

# ── 1a. Descriptive Statistics ────────────────────────────────────────────────
print("\n[1a] Descriptive Statistics")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"     Numerical columns found: {num_cols}")

if num_cols:
    desc = df[num_cols].describe().T
    desc['median'] = df[num_cols].median()
    for col in num_cols:
        mode_val = df[col].mode()
        desc.loc[col, 'mode'] = mode_val.iloc[0] if len(mode_val) > 0 else np.nan
    print(desc.to_string())

# Also compute for Amount_Lost_FCFA and Nb_Times (stored as strings in dataset)
for col in ['Amount_Lost_FCFA', 'Nb_Times']:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) > 0:
            print(f"\n  {col} (converted to numeric, {series.isna().sum()} NaN skipped):")
            print(f"    Mean:   {series.mean():.2f}")
            print(f"    Median: {series.median():.2f}")
            print(f"    Std:    {series.std():.2f}")
            print(f"    Min:    {series.min():.2f}  |  Max: {series.max():.2f}")

# ── 1b. Distribution Histograms ───────────────────────────────────────────────
print("\n[1b] Distribution Analysis — Histograms")
if num_cols:
    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
    if len(num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col].dropna(), bins=20, color=PALETTE, edgecolor='white', alpha=0.85)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    plt.suptitle('Distribution of Numerical Features', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_show("01_histograms.png")

# ── 1c. Class Imbalance Check ─────────────────────────────────────────────────
print("\n[1c] Class Imbalance Check")
victim_counts = df['Victim'].value_counts()
print(f"     {victim_counts.to_dict()}")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2E86AB', '#E84855']
bars = ax.bar(victim_counts.index, victim_counts.values,
              color=colors[:len(victim_counts)], edgecolor='white', width=0.5)
for bar, val in zip(bars, victim_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f'{val}\n({val / len(df) * 100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
ax.set_title('Class Distribution: Scam Victim', fontsize=13, fontweight='bold')
ax.set_xlabel('Victim Status')
ax.set_ylabel('Count')
plt.tight_layout()
save_and_show("02_class_imbalance.png")

# ── 1d. Gender & Age Breakdown ────────────────────────────────────────────────
print("\n[1d] Gender & Age Breakdown vs Victim Status")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

gender_victim = df.groupby(['Gender', 'Victim']).size().unstack(fill_value=0)
gender_victim.plot(kind='bar', ax=axes[0],
                   color=['#2E86AB', '#E84855'][:len(gender_victim.columns)],
                   edgecolor='white', rot=0)
axes[0].set_title('Gender vs Victim Status', fontweight='bold')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].legend(title='Victim')

df_age = df.dropna(subset=['Age', 'Victim'])
for v, c in zip(df['Victim'].dropna().unique(), ['#E84855', '#2E86AB', '#28A745']):
    subset = df_age[df_age['Victim'] == v]['Age']
    axes[1].hist(subset, bins=15, alpha=0.65, color=c, label=v, edgecolor='white')
axes[1].set_title('Age Distribution by Victim Status', fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
axes[1].legend(title='Victim')
plt.tight_layout()
save_and_show("03_gender_age_analysis.png")

# ── 1e. Scam Methods Breakdown ────────────────────────────────────────────────
print("\n[1e] Scam Methods Breakdown")
all_methods = []
for entry in df['Scam_Method'].dropna():
    for m in str(entry).split(';'):
        m = m.strip()
        if m and m.lower() not in ['', 'nan', 'none']:
            all_methods.append(m)
method_counts = pd.Series(all_methods).value_counts().head(8)
print(method_counts.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(method_counts.index[::-1], method_counts.values[::-1],
        color=PALETTE, edgecolor='white')
for i, val in enumerate(method_counts.values[::-1]):
    ax.text(val + 0.5, i, str(val), va='center', fontweight='bold')
ax.set_title('Most Common Scam Methods Reported', fontsize=13, fontweight='bold')
ax.set_xlabel('Count')
plt.tight_layout()
save_and_show("04_scam_methods.png")

# ── 1f. Occupation vs Victim ──────────────────────────────────────────────────
print("\n[1f] Occupation vs Victim Status")
occ_victim = df.groupby(['Occupation', 'Victim']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4))
occ_victim.plot(kind='bar', ax=ax,
                color=['#2E86AB', '#E84855'][:len(occ_victim.columns)],
                edgecolor='white', rot=30)
ax.set_title('Occupation vs Victim Status', fontsize=13, fontweight='bold')
ax.set_xlabel('Occupation')
ax.set_ylabel('Count')
ax.legend(title='Victim')
plt.tight_layout()
save_and_show("05_occupation_victim.png")


# =============================================================================
# STEP 2 — DATA PREPROCESSING (CLEANING)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2: Data Preprocessing")
print("=" * 65)

df_model = df.copy()

# Drop columns that are too sparse, free-text, or multi-label
drop_cols = ['Timestamp', 'Other_Method', 'Reported_To', 'Suggestions', 'Info_Source']
df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], inplace=True)
print(f"\nDropped columns: {[c for c in drop_cols if c in df.columns]}")

# Keep only rows where Victim is known
df_model = df_model[df_model['Victim'].notna()].copy()

# Encode target variable
df_model['Victim_Binary'] = (df_model['Victim'].str.strip().str.lower() == 'yes').astype(int)
df_model.drop(columns=['Victim'], inplace=True)
print(f"Target encoded — 1=Victim, 0=Not Victim")
print(f"Target distribution:\n{df_model['Victim_Binary'].value_counts().to_string()}")

# Label encode all remaining categorical columns
cat_cols = df_model.select_dtypes(include='object').columns.tolist()
print(f"\nCategorical columns being label-encoded: {cat_cols}")
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = df_model[col].fillna('Unknown')
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Identify feature columns
num_features = [c for c in df_model.columns if c != 'Victim_Binary']
print(f"\nAll model features ({len(num_features)}): {num_features}")

# Impute remaining missing values (median strategy)
imputer = SimpleImputer(strategy='median')
df_model[num_features] = imputer.fit_transform(df_model[num_features])
print(f"\nMissing values after imputation: {df_model.isnull().sum().sum()}")
print(f"Final model dataset shape: {df_model.shape}")

# ── 2a. Boxplots — Outlier Detection ─────────────────────────────────────────
print("\n[2a] Outlier Detection via Boxplots")
box_cols = [c for c in ['Age', 'Amount_Lost_FCFA', 'Nb_Times'] if c in df_model.columns]
if box_cols:
    fig, axes = plt.subplots(1, len(box_cols), figsize=(5 * len(box_cols), 4))
    if len(box_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, box_cols):
        ax.boxplot(df_model[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='#AED9E0', color='#2E86AB'),
                   medianprops=dict(color='#E84855', linewidth=2),
                   flierprops=dict(marker='o', color='#E84855', alpha=0.5))
        ax.set_title(f'Boxplot: {col}', fontweight='bold')
        ax.set_ylabel('Value')
    plt.suptitle('Outlier Detection via Boxplots', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_and_show("06_boxplots.png")

# ── 2b. Correlation Heatmap ───────────────────────────────────────────────────
print("\n[2b] Correlation Heatmap")
corr_cols = num_features + ['Victim_Binary']
corr_df = df_model[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Correlation Heatmap of Features', fontsize=13, fontweight='bold')
plt.tight_layout()
save_and_show("07_correlation_heatmap.png")


# =============================================================================
# STEP 3 — FEATURE ENGINEERING & SCALING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3: Feature Engineering & Scaling")
print("=" * 65)

X = df_model[num_features].values
y = df_model['Victim_Binary'].values

# ── 3a. StandardScaler ────────────────────────────────────────────────────────
print("\n[3a] Applying StandardScaler (zero mean, unit variance)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"     Scaled feature matrix shape: {X_scaled.shape}")
print(f"     Mean of scaled features (should be ~0): {X_scaled.mean(axis=0).round(3)}")

# ── 3b. PCA ───────────────────────────────────────────────────────────────────
print("\n[3b] PCA — Dimensionality Reduction")
n_components = min(5, X_scaled.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

print(f"     Explained variance per component:")
for i, v in enumerate(explained):
    print(f"       PC{i+1}: {v*100:.1f}%  (cumulative: {np.cumsum(explained)[i]*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, n_components + 1), explained * 100, color=PALETTE, edgecolor='white')
axes[0].set_title('PCA: Variance per Component', fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
for i, v in enumerate(explained * 100):
    axes[0].text(i + 1, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

axes[1].plot(range(1, n_components + 1), np.cumsum(explained) * 100,
             marker='o', color='#E84855', linewidth=2, markersize=8)
axes[1].axhline(80, color='gray', linestyle='--', label='80% threshold')
axes[1].set_title('PCA: Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].legend()
plt.suptitle('Principal Component Analysis (PCA)', fontsize=13, fontweight='bold')
plt.tight_layout()
save_and_show("08_pca.png")


# =============================================================================
# STEP 4 — PREDICTIVE MODELLING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4: Predictive Modelling")
print("=" * 65)

# ── 4a. Train-Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")
print(f"Stratified split — victim ratio preserved in both sets")

# ── 4b. Train 3 Models ────────────────────────────────────────────────────────
print("\n[4b] Training models...")
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    'SVM': SVC(
        probability=True, random_state=42, class_weight='balanced'
    )
}

results = {}
for name, model in models.items():
    print(f"\n  Training: {name} ...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1_val  = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)
    report      = classification_report(y_test, y_pred, zero_division=0,
                                        target_names=['Not Victim', 'Victim'])
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'f1': f1_val, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'report': report
    }
    print(f"  F1-Score = {f1_val:.4f}  |  ROC-AUC = {roc_auc:.4f}")
    print(f"\n  Classification Report — {name}:")
    print(report)


# =============================================================================
# STEP 5 — EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5: Model Evaluation")
print("=" * 65)

# ── 5a. Confusion Matrices ────────────────────────────────────────────────────
print("\n[5a] Confusion Matrices")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Victim', 'Victim'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontweight='bold')
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
plt.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold')
plt.tight_layout()
save_and_show("09_confusion_matrices.png")

# ── 5b. ROC Curves ────────────────────────────────────────────────────────────
print("\n[5b] ROC Curves")
colors_roc = ['#2E86AB', '#E84855', '#28A745']
fig, ax = plt.subplots(figsize=(7, 5))
for (name, res), c in zip(results.items(), colors_roc):
    ax.plot(res['fpr'], res['tpr'], color=c, linewidth=2,
            label=f"{name} (AUC = {res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save_and_show("10_roc_curves.png")

# ── 5c. F1 & AUC Comparison Bar Chart ────────────────────────────────────────
print("\n[5c] F1-Score & AUC Comparison")
names = list(results.keys())
f1s  = [results[n]['f1']  for n in names]
aucs = [results[n]['auc'] for n in names]
x    = np.arange(len(names))

fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - 0.2, f1s,  0.35, label='F1-Score', color='#2E86AB', edgecolor='white')
bars2 = ax.bar(x + 0.2, aucs, 0.35, label='ROC-AUC',  color='#E84855', edgecolor='white')
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Model Comparison: F1-Score & ROC-AUC', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save_and_show("11_model_comparison.png")

# ── 5d. Feature Importance (Random Forest) ───────────────────────────────────
print("\n[5d] Feature Importance — Random Forest")
rf_model    = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_,
                        index=num_features).sort_values(ascending=True)
print(importances.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color=PALETTE, edgecolor='white')
ax.set_title('Random Forest — Feature Importance', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
save_and_show("12_feature_importance.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\nDataset:       {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Victims:       {int(victim_counts.get('Yes', 0))} "
      f"({int(victim_counts.get('Yes', 0))/len(df)*100:.1f}%)")
print(f"Non-victims:   {int(victim_counts.get('No', 0))} "
      f"({int(victim_counts.get('No', 0))/len(df)*100:.1f}%)")
print(f"Age (mean):    {df['Age'].mean():.1f} years")
print(f"Age (median):  {df['Age'].median():.0f} years")
print(f"Avg loss:      {pd.to_numeric(df['Amount_Lost_FCFA'], errors='coerce').mean():.0f} FCFA")
print(f"\nTop scam method: {method_counts.index[0]} ({method_counts.iloc[0]} cases)")
print(f"\nPCA — Variance captured by 5 components: "
      f"{np.cumsum(explained)[-1]*100:.1f}%")
print("\nModel Performance:")
print(f"  {'Model':<25} {'F1-Score':>10} {'ROC-AUC':>10}")
print(f"  {'-'*45}")
for name in names:
    print(f"  {name:<25} {results[name]['f1']:>10.4f} {results[name]['auc']:>10.4f}")

print(f"\nAll plots saved to: ./{PLOTS_DIR}/")
print("\n=== ANALYSIS COMPLETE ===\n")