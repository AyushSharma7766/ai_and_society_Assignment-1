#!/usr/bin/env python3
"""
Adult Census Income Dataset — Full Predictive Modeling & Fairness Analysis
==========================================================================
Dataset: Adult Census Income (UCI ML Repository)
Random Seed: 758   |   Train/Test Split: 70/30

This script covers:
  - Data Preprocessing (missing values, encoding, scaling, outliers)
  - Part 1: Build (OLS Linear Regression + Logistic Regression)
  - Part 2.1: Simpson's Paradox Hunt
  - Part 2.2: Omitted-Variable Bias
  - Part 2.3: Adversarial Subset Construction
  - Part 2.4: Fairness Trade-Off Analysis
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)
import statsmodels.api as sm

SEED = 758
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
#  0. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 0 — DATA LOADING")
print("=" * 80)

col_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Load training & test files from UCI and combine
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(url_train, header=None, names=col_names,
                        na_values=" ?", skipinitialspace=True)
df_test  = pd.read_csv(url_test, header=None, names=col_names,
                        na_values=" ?", skipinitialspace=True, skiprows=1)

# Combine into one dataset (we do our own 70/30 split)
df = pd.concat([df_train, df_test], ignore_index=True)

# Clean the income column (test file has trailing period)
df["income"] = df["income"].str.strip().str.rstrip(".")

print(f"Combined dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nIncome distribution:\n{df['income'].value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 1 — DATA PREPROCESSING")
print("=" * 80)

# --- 1a. Missing Values ---
print("\n--- 1a. Missing Values ---")
missing = df.isnull().sum()
print(f"Missing values per column:\n{missing[missing > 0]}")
print(f"\nTotal rows with any missing: {df.isnull().any(axis=1).sum()}")
print(f"Percentage missing: {df.isnull().any(axis=1).sum() / len(df) * 100:.2f}%")

# Strategy: drop rows with missing values (they are marked ' ?' → NaN)
df_clean = df.dropna().reset_index(drop=True)
print(f"Shape after dropping missing: {df_clean.shape}")

# --- 1b. Target Encoding ---
print("\n--- 1b. Target Encoding ---")
df_clean["income_binary"] = (df_clean["income"] == ">50K").astype(int)
print(f"Target distribution:\n{df_clean['income_binary'].value_counts()}")
print(f"Positive class rate: {df_clean['income_binary'].mean():.4f}")

# --- 1c. Categorical Encoding ---
print("\n--- 1c. Categorical Encoding ---")
categorical_cols = ["workclass", "education", "marital_status", "occupation",
                    "relationship", "race", "sex", "native_country"]
print(f"Categorical columns: {categorical_cols}")

# One-hot encode
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"Shape after one-hot encoding: {df_encoded.shape}")

# Drop the original income string column
df_encoded.drop(columns=["income"], inplace=True)

# --- 1d. Outlier Analysis ---
print("\n--- 1d. Outlier Analysis ---")
numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain",
                  "capital_loss", "hours_per_week"]
for col in numerical_cols:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df_encoded[col] < lower) | (df_encoded[col] > upper)).sum()
    pct = n_outliers / len(df_encoded) * 100
    print(f"  {col:20s}: {n_outliers:6d} outliers ({pct:.2f}%)"
          f"  [Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}]")

# Decision: keep outliers — capital_gain/capital_loss are legitimately skewed
# and removing them would distort the income signal.
print("\n  Decision: Outliers retained. capital_gain and capital_loss are")
print("  legitimately right-skewed; removing them would lose high-income signal.")

# --- 1e. Feature Scaling ---
print("\n--- 1e. Feature Scaling ---")
feature_cols = [c for c in df_encoded.columns if c != "income_binary"]
target_col = "income_binary"

X = df_encoded[feature_cols].copy()
y = df_encoded[target_col].copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

print(f"Features scaled with StandardScaler (mean=0, std=1)")
print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Target shape: {y.shape}")

# --- 1f. Train-Test Split ---
print("\n--- 1f. Train-Test Split (70/30, random_state=758) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=SEED
)
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Test : {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate : {y_test.mean():.4f}")

# Keep a copy of the original (unscaled, un-encoded) test set for demographic analysis
# We need to map test indices back
test_indices = X_test.index
df_test_original = df_clean.loc[test_indices].reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════════
#  PART 1 — BUILD
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 1 — BUILD")
print("=" * 80)

# --- Model A: Linear Regression (OLS) ---
print("\n--- Model A: Linear Regression (OLS) ---")
model_a = LinearRegression()
model_a.fit(X_train, y_train)
y_pred_a_raw = model_a.predict(X_test)

print(f"OLS predictions — min: {y_pred_a_raw.min():.4f}, "
      f"max: {y_pred_a_raw.max():.4f}, "
      f"mean: {y_pred_a_raw.mean():.4f}")

# Threshold at median predicted value
median_pred = np.median(y_pred_a_raw)
y_pred_a_class = (y_pred_a_raw >= median_pred).astype(int)
print(f"Median predicted value (threshold): {median_pred:.4f}")
print(f"Model A classifications: "
      f"High-risk={y_pred_a_class.sum()}, Low-risk={(1-y_pred_a_class).sum()}")

# OLS via statsmodels for full summary
X_train_sm = sm.add_constant(X_train)
X_test_sm  = sm.add_constant(X_test)
ols_model  = sm.OLS(y_train, X_train_sm).fit()
print("\n--- OLS Summary (top coefficients by magnitude) ---")
ols_coefs = ols_model.params.drop("const").abs().sort_values(ascending=False)
print(ols_coefs.head(10))

# --- Model B: Logistic Regression ---
print("\n--- Model B: Logistic Regression ---")
model_b = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
model_b.fit(X_train, y_train)
y_pred_b_proba = model_b.predict_proba(X_test)[:, 1]
y_pred_b_class = (y_pred_b_proba >= 0.5).astype(int)

print(f"Logistic predictions — min prob: {y_pred_b_proba.min():.4f}, "
      f"max prob: {y_pred_b_proba.max():.4f}, "
      f"mean prob: {y_pred_b_proba.mean():.4f}")
print(f"Model B classifications: "
      f"High-risk={y_pred_b_class.sum()}, Low-risk={(1-y_pred_b_class).sum()}")

# --- Comparison ---
print("\n--- Model Comparison: Agreement / Disagreement ---")

# Confusion matrix: rows = Model A classification, cols = Model B classification
cm_ab = confusion_matrix(y_pred_a_class, y_pred_b_class)
print("\nConfusion Matrix (Model A vs Model B):")
print(f"{'':>25s} Model B: Low-risk  Model B: High-risk")
print(f"{'Model A: Low-risk':>25s}    {cm_ab[0,0]:>6d}          {cm_ab[0,1]:>6d}")
print(f"{'Model A: High-risk':>25s}    {cm_ab[1,0]:>6d}          {cm_ab[1,1]:>6d}")

agree = cm_ab[0, 0] + cm_ab[1, 1]
disagree = cm_ab[0, 1] + cm_ab[1, 0]
total = agree + disagree
print(f"\nAgreement:    {agree:>6d} ({agree/total*100:.2f}%)")
print(f"Disagreement: {disagree:>6d} ({disagree/total*100:.2f}%)")

# Accuracy of both models against actual labels
acc_a = accuracy_score(y_test, y_pred_a_class)
acc_b = accuracy_score(y_test, y_pred_b_class)
print(f"\nModel A accuracy (vs actual): {acc_a:.4f}")
print(f"Model B accuracy (vs actual): {acc_b:.4f}")

# Classification reports
print("\n--- Model A Classification Report ---")
print(classification_report(y_test, y_pred_a_class, target_names=["<=50K", ">50K"]))
print("--- Model B Classification Report ---")
print(classification_report(y_test, y_pred_b_class, target_names=["<=50K", ">50K"]))

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Model A vs Actual
cm_a_actual = confusion_matrix(y_test, y_pred_a_class)
ConfusionMatrixDisplay(cm_a_actual, display_labels=["<=50K", ">50K"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Model A (OLS, median threshold) vs Actual")

# Model B vs Actual
cm_b_actual = confusion_matrix(y_test, y_pred_b_class)
ConfusionMatrixDisplay(cm_b_actual, display_labels=["<=50K", ">50K"]).plot(ax=axes[1], cmap="Greens")
axes[1].set_title("Model B (Logistic, 0.5 threshold) vs Actual")

# Model A vs Model B
ConfusionMatrixDisplay(cm_ab, display_labels=["Low-risk", "High-risk"]).plot(ax=axes[2], cmap="Oranges")
axes[2].set_title("Model A vs Model B Agreement")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figures", "part1_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n[Figure saved: figures/part1_confusion_matrices.png]")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2.1 — SIMPSON'S PARADOX HUNT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2.1 — SIMPSON'S PARADOX HUNT")
print("=" * 80)

# We'll split by sex: Male vs Female
# Fit logistic regression on: (1) full dataset, (2) Males only, (3) Females only
# Look at how coefficients for key predictors change

# Prepare unscaled data with binary target for re-fitting
df_analysis = df_clean.copy()
cat_cols_analysis = ["workclass", "education", "marital_status", "occupation",
                     "relationship", "race", "native_country"]

# Keep sex as a column but one-hot encode the rest
df_simpson = pd.get_dummies(df_analysis, columns=cat_cols_analysis, drop_first=True)
df_simpson["sex_Male"] = (df_simpson["sex"] == "Male").astype(int)
df_simpson.drop(columns=["income", "sex"], inplace=True)

feature_cols_simpson = [c for c in df_simpson.columns if c != "income_binary"]

# Standardize
X_full_s = df_simpson[feature_cols_simpson].copy()
y_full_s = df_simpson["income_binary"].copy()

scaler_s = StandardScaler()
X_full_scaled = pd.DataFrame(scaler_s.fit_transform(X_full_s), columns=feature_cols_simpson)

# Split into Male/Female using the original df
male_mask = df_analysis["sex"] == "Male"
female_mask = df_analysis["sex"] == "Female"

# Filter the scaled dataset
# Need indices aligned
X_male = X_full_scaled.loc[male_mask.values].copy()
y_male = y_full_s.loc[male_mask.values].copy()
X_female = X_full_scaled.loc[female_mask.values].copy()
y_female = y_full_s.loc[female_mask.values].copy()

print(f"Full dataset: {len(X_full_scaled)} samples, positive rate: {y_full_s.mean():.4f}")
print(f"Males:        {len(X_male)} samples, positive rate: {y_male.mean():.4f}")
print(f"Females:      {len(X_female)} samples, positive rate: {y_female.mean():.4f}")

# Fit three logistic regressions
lr_full = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_full.fit(X_full_scaled, y_full_s)

lr_male = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_male.fit(X_male, y_male)

# For female, drop sex_Male column (constant=0 for all females)
female_features = [c for c in feature_cols_simpson if c != "sex_Male"]
X_female_nosex = X_female[female_features]
lr_female = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_female.fit(X_female_nosex, y_female)

# For male subgroup, drop sex_Male (constant=1 for all males)
X_male_nosex = X_male[female_features]
lr_male_nosex = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_male_nosex.fit(X_male_nosex, y_male)

# Compare coefficients for key predictors
key_predictors = ["age", "education_num", "hours_per_week", "capital_gain",
                  "capital_loss", "fnlwgt"]

# Also look at marital status (likely driver of Simpson's paradox with sex)
marital_cols = [c for c in feature_cols_simpson if c.startswith("marital_status_")]
key_predictors_extended = key_predictors + marital_cols[:3]

print("\n--- Coefficient Comparison (Full vs Male vs Female) ---")
print(f"{'Predictor':<45s} {'Full':>10s} {'Male':>10s} {'Female':>10s}")
print("-" * 80)

coef_full_dict = dict(zip(feature_cols_simpson, lr_full.coef_[0]))
coef_male_dict = dict(zip(female_features, lr_male_nosex.coef_[0]))
coef_female_dict = dict(zip(female_features, lr_female.coef_[0]))

simpson_results = []
for pred in key_predictors_extended:
    if pred in coef_full_dict and pred in coef_male_dict and pred in coef_female_dict:
        c_full = coef_full_dict[pred]
        c_male = coef_male_dict[pred]
        c_female = coef_female_dict[pred]
        print(f"{pred:<45s} {c_full:>10.4f} {c_male:>10.4f} {c_female:>10.4f}")
        simpson_results.append({
            "predictor": pred, "full": c_full, "male": c_male, "female": c_female
        })

# Look for sign reversals or large differences
print("\n--- Sign Reversals / Large Differences ---")
for r in simpson_results:
    if (r["male"] > 0 and r["female"] < 0) or (r["male"] < 0 and r["female"] > 0):
        print(f"  SIGN REVERSAL: {r['predictor']}")
        print(f"    Full={r['full']:.4f}, Male={r['male']:.4f}, Female={r['female']:.4f}")
    ratio = abs(r["male"] / r["female"]) if abs(r["female"]) > 0.001 else float("inf")
    if ratio > 3 or ratio < 0.33:
        print(f"  LARGE DIFFERENCE: {r['predictor']}")
        print(f"    Full={r['full']:.4f}, Male={r['male']:.4f}, Female={r['female']:.4f}")
        print(f"    Male/Female ratio: {ratio:.2f}")

# Also try splitting by race for additional paradox hunting
print("\n--- Alternative split: by relationship status (Married vs Not-Married) ---")
married_mask = df_analysis["marital_status"].isin(["Married-civ-spouse", "Married-AF-spouse"])
not_married_mask = ~married_mask

# For married/not-married, we fit on features excluding marital status dummies
non_marital_features = [c for c in feature_cols_simpson
                        if not c.startswith("marital_status_")]

X_full_nm = X_full_scaled[non_marital_features]
X_married = X_full_scaled.loc[married_mask.values][non_marital_features]
y_married = y_full_s.loc[married_mask.values]
X_notmarried = X_full_scaled.loc[not_married_mask.values][non_marital_features]
y_notmarried = y_full_s.loc[not_married_mask.values]

lr_married = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_married.fit(X_married, y_married)
lr_notmarried = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_notmarried.fit(X_notmarried, y_notmarried)
lr_full_nm = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
lr_full_nm.fit(X_full_nm, y_full_s)

print(f"\nMarried:     {X_married.shape[0]} samples, positive rate: {y_married.mean():.4f}")
print(f"Not Married: {X_notmarried.shape[0]} samples, positive rate: {y_notmarried.mean():.4f}")

coef_full_nm = dict(zip(non_marital_features, lr_full_nm.coef_[0]))
coef_married = dict(zip(non_marital_features, lr_married.coef_[0]))
coef_notmarried = dict(zip(non_marital_features, lr_notmarried.coef_[0]))

print(f"\n{'Predictor':<45s} {'Full':>10s} {'Married':>10s} {'Not-Married':>12s}")
print("-" * 80)
for pred in ["age", "education_num", "hours_per_week", "capital_gain",
             "sex_Male", "capital_loss", "fnlwgt"]:
    if pred in coef_full_nm:
        c_f = coef_full_nm[pred]
        c_m = coef_married[pred]
        c_nm = coef_notmarried[pred]
        print(f"{pred:<45s} {c_f:>10.4f} {c_m:>10.4f} {c_nm:>12.4f}")
        if (c_m > 0 and c_nm < 0) or (c_m < 0 and c_nm > 0):
            print(f"  *** SIGN REVERSAL DETECTED ***")

# Statsmodels for CIs
print("\n--- Statsmodels Logistic Regression for Confidence Intervals ---")
# Fit on full, male, female for the best Simpson candidate

# Use a subset of features for clean CI display
key_feats_ci = ["age", "education_num", "hours_per_week", "capital_gain",
                "capital_loss", "sex_Male"]
key_feats_ci_nosex = ["age", "education_num", "hours_per_week", "capital_gain",
                      "capital_loss"]

# Full model
X_ci_full = sm.add_constant(X_full_scaled[key_feats_ci])
logit_full = sm.Logit(y_full_s, X_ci_full).fit(disp=0)
print("\n--- Full Model ---")
print(logit_full.summary2().tables[1][["Coef.", "[0.025", "0.975]"]])

# Male model
X_ci_male = sm.add_constant(X_male[key_feats_ci_nosex])
logit_male = sm.Logit(y_male, X_ci_male).fit(disp=0)
print("\n--- Male Subgroup ---")
print(logit_male.summary2().tables[1][["Coef.", "[0.025", "0.975]"]])

# Female model
X_ci_female = sm.add_constant(X_female[key_feats_ci_nosex])
logit_female = sm.Logit(y_female, X_ci_female).fit(disp=0)
print("\n--- Female Subgroup ---")
print(logit_female.summary2().tables[1][["Coef.", "[0.025", "0.975]"]])

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2.2 — OMITTED-VARIABLE BIAS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2.2 — OMITTED-VARIABLE BIAS")
print("=" * 80)

# Strategy: Drop "education_num" — a strong confound correlated with
# occupation, hours worked, and income.
# We could also try dropping "marital_status" related features.

# We'll drop all marital_status dummies — marital status is a known confound
# that correlates with sex, age, and income.
dropped_var = "marital_status"
print(f"\nDropped variable: {dropped_var} (all related dummies)")

marital_dummies = [c for c in feature_cols if c.startswith("marital_status_")]
print(f"Dummies removed: {marital_dummies}")

# Full model (already trained as model_b)
print("\n--- Full Model (all features) ---")
coef_full_all = dict(zip(feature_cols, model_b.coef_[0]))

# Reduced model (without marital_status dummies)
reduced_features = [c for c in feature_cols if c not in marital_dummies]
X_train_reduced = X_train[reduced_features]
X_test_reduced = X_test[reduced_features]

model_b_reduced = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
model_b_reduced.fit(X_train_reduced, y_train)
coef_reduced = dict(zip(reduced_features, model_b_reduced.coef_[0]))

# Compare coefficients
print(f"\n{'Predictor':<45s} {'Full':>10s} {'Reduced':>10s} {'% Change':>10s}")
print("-" * 80)

# Pick key predictors to compare
compare_predictors = ["age", "education_num", "hours_per_week", "capital_gain",
                      "capital_loss", "fnlwgt"]
# Also add sex and relationship dummies
sex_dummies = [c for c in feature_cols if c.startswith("sex_")]
relationship_dummies = [c for c in feature_cols if c.startswith("relationship_")]
compare_predictors += sex_dummies + relationship_dummies[:3]

ovb_results = []
for pred in compare_predictors:
    if pred in coef_full_all and pred in coef_reduced:
        c_full = coef_full_all[pred]
        c_red = coef_reduced[pred]
        if abs(c_full) > 1e-6:
            pct_change = ((c_red - c_full) / abs(c_full)) * 100
        else:
            pct_change = float("inf")
        print(f"{pred:<45s} {c_full:>10.4f} {c_red:>10.4f} {pct_change:>+10.1f}%")
        ovb_results.append({
            "predictor": pred, "full": c_full, "reduced": c_red,
            "pct_change": pct_change
        })

# Find the most shifted coefficient
ovb_df = pd.DataFrame(ovb_results)
ovb_df["abs_pct_change"] = ovb_df["pct_change"].abs()
most_shifted = ovb_df.sort_values("abs_pct_change", ascending=False).iloc[0]
print(f"\nMost shifted coefficient: {most_shifted['predictor']}")
print(f"  Full model: {most_shifted['full']:.4f}")
print(f"  Reduced model: {most_shifted['reduced']:.4f}")
print(f"  Percentage change: {most_shifted['pct_change']:+.1f}%")

# Model performance comparison
acc_full = accuracy_score(y_test, model_b.predict(X_test))
acc_reduced = accuracy_score(y_test, model_b_reduced.predict(X_test_reduced))
print(f"\nFull model accuracy:    {acc_full:.4f}")
print(f"Reduced model accuracy: {acc_reduced:.4f}")
print(f"Accuracy drop:          {acc_full - acc_reduced:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2.3 — ADVERSARIAL SUBSET CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2.3 — ADVERSARIAL SUBSET CONSTRUCTION")
print("=" * 80)

# Find high-confidence wrong predictions
y_test_arr = y_test.values
proba_test = y_pred_b_proba

# False positives: P >= 0.75 but actual = 0
fp_mask = (proba_test >= 0.75) & (y_test_arr == 0)
# False negatives: P <= 0.25 but actual = 1
fn_mask = (proba_test <= 0.25) & (y_test_arr == 1)

adversarial_mask = fp_mask | fn_mask
n_adversarial = adversarial_mask.sum()
n_test = len(y_test_arr)

print(f"Total test samples: {n_test}")
print(f"High-confidence wrong predictions: {n_adversarial} ({n_adversarial/n_test*100:.2f}%)")
print(f"  False positives (P>=0.75, actual=0): {fp_mask.sum()}")
print(f"  False negatives (P<=0.25, actual=1): {fn_mask.sum()}")

# Extract adversarial subset (sample 50 if more exist)
adversarial_indices = np.where(adversarial_mask)[0]
if len(adversarial_indices) > 50:
    rng = np.random.RandomState(SEED)
    sampled_indices = rng.choice(adversarial_indices, size=50, replace=False)
else:
    sampled_indices = adversarial_indices

df_adversarial = df_test_original.iloc[sampled_indices].copy()
df_adversarial["predicted_prob"] = proba_test[sampled_indices]
df_adversarial["actual"] = y_test_arr[sampled_indices]
df_adversarial["error_type"] = np.where(
    proba_test[sampled_indices] >= 0.75, "False Positive", "False Negative"
)

print(f"\nSampled adversarial subset: {len(df_adversarial)} rows")
print(f"  False Positives: {(df_adversarial['error_type'] == 'False Positive').sum()}")
print(f"  False Negatives: {(df_adversarial['error_type'] == 'False Negative').sum()}")

# Demographics of adversarial subset vs full test set
print("\n--- Demographics: Adversarial Subset vs Full Test Set ---")

print("\nAge distribution:")
print(f"  Adversarial — mean: {df_adversarial['age'].mean():.1f}, "
      f"median: {df_adversarial['age'].median():.1f}, "
      f"std: {df_adversarial['age'].std():.1f}")
print(f"  Full test   — mean: {df_test_original['age'].mean():.1f}, "
      f"median: {df_test_original['age'].median():.1f}, "
      f"std: {df_test_original['age'].std():.1f}")

print("\nSex distribution:")
adv_sex = df_adversarial["sex"].value_counts(normalize=True)
full_sex = df_test_original["sex"].value_counts(normalize=True)
for s in ["Male", "Female"]:
    a_pct = adv_sex.get(s, 0) * 100
    f_pct = full_sex.get(s, 0) * 100
    print(f"  {s}: Adversarial={a_pct:.1f}%, Full test={f_pct:.1f}%")

print("\nRace distribution:")
adv_race = df_adversarial["race"].value_counts(normalize=True)
full_race = df_test_original["race"].value_counts(normalize=True)
for r in df_test_original["race"].unique():
    a_pct = adv_race.get(r, 0) * 100
    f_pct = full_race.get(r, 0) * 100
    if f_pct > 1:
        print(f"  {r}: Adversarial={a_pct:.1f}%, Full test={f_pct:.1f}%")

print("\nMarital status distribution:")
adv_mar = df_adversarial["marital_status"].value_counts(normalize=True)
full_mar = df_test_original["marital_status"].value_counts(normalize=True)
for m in df_test_original["marital_status"].unique():
    a_pct = adv_mar.get(m, 0) * 100
    f_pct = full_mar.get(m, 0) * 100
    if f_pct > 3:
        print(f"  {m}: Adversarial={a_pct:.1f}%, Full test={f_pct:.1f}%")

print("\nEducation distribution:")
print(f"  Adversarial — mean education_num: {df_adversarial['education_num'].mean():.1f}")
print(f"  Full test   — mean education_num: {df_test_original['education_num'].mean():.1f}")

print("\nHours per week:")
print(f"  Adversarial — mean: {df_adversarial['hours_per_week'].mean():.1f}, "
      f"median: {df_adversarial['hours_per_week'].median():.1f}")
print(f"  Full test   — mean: {df_test_original['hours_per_week'].mean():.1f}, "
      f"median: {df_test_original['hours_per_week'].median():.1f}")

print("\nCapital gain:")
print(f"  Adversarial — mean: {df_adversarial['capital_gain'].mean():.1f}, "
      f"% with gain>0: {(df_adversarial['capital_gain']>0).mean()*100:.1f}%")
print(f"  Full test   — mean: {df_test_original['capital_gain'].mean():.1f}, "
      f"% with gain>0: {(df_test_original['capital_gain']>0).mean()*100:.1f}%")

# Find common characteristics
print("\n--- Common Characteristics of Adversarial Subset ---")

# Check for FP subset specifically
df_fp = df_adversarial[df_adversarial["error_type"] == "False Positive"]
df_fn = df_adversarial[df_adversarial["error_type"] == "False Negative"]

if len(df_fp) > 0:
    print(f"\nFalse Positives ({len(df_fp)} rows):")
    print(f"  % Married-civ-spouse: {(df_fp['marital_status']=='Married-civ-spouse').mean()*100:.1f}%")
    print(f"  % Male: {(df_fp['sex']=='Male').mean()*100:.1f}%")
    print(f"  Mean age: {df_fp['age'].mean():.1f}")
    print(f"  Mean education_num: {df_fp['education_num'].mean():.1f}")
    print(f"  Mean hours_per_week: {df_fp['hours_per_week'].mean():.1f}")
    print(f"  Mean capital_gain: {df_fp['capital_gain'].mean():.1f}")

if len(df_fn) > 0:
    print(f"\nFalse Negatives ({len(df_fn)} rows):")
    print(f"  % Never-married: {(df_fn['marital_status']=='Never-married').mean()*100:.1f}%")
    print(f"  % Female: {(df_fn['sex']=='Female').mean()*100:.1f}%")
    print(f"  Mean age: {df_fn['age'].mean():.1f}")
    print(f"  Mean education_num: {df_fn['education_num'].mean():.1f}")
    print(f"  Mean hours_per_week: {df_fn['hours_per_week'].mean():.1f}")
    print(f"  Mean capital_gain: {df_fn['capital_gain'].mean():.1f}")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2.4 — FAIRNESS TRADE-OFF ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2.4 — FAIRNESS TRADE-OFF ANALYSIS")
print("=" * 80)

# Sensitive variable: sex (Male=Group A, Female=Group B)
male_test_mask = df_test_original["sex"].values == "Male"
female_test_mask = df_test_original["sex"].values == "Female"

print(f"Group A (Male): {male_test_mask.sum()} test samples")
print(f"Group B (Female): {female_test_mask.sum()} test samples")

thresholds = np.arange(0.05, 1.0, 0.10)
fairness_results = []

for t in thresholds:
    y_pred_t = (proba_test >= t).astype(int)
    
    # Group A (Male)
    y_pred_a_grp = y_pred_t[male_test_mask]
    y_true_a_grp = y_test_arr[male_test_mask]
    
    # Group B (Female)
    y_pred_b_grp = y_pred_t[female_test_mask]
    y_true_b_grp = y_test_arr[female_test_mask]
    
    # Demographic Parity: P(Ŷ=1|A) / P(Ŷ=1|B)
    p_pos_a = y_pred_a_grp.mean() if len(y_pred_a_grp) > 0 else 0
    p_pos_b = y_pred_b_grp.mean() if len(y_pred_b_grp) > 0 else 0
    dp = p_pos_a / p_pos_b if p_pos_b > 0 else float("inf")
    
    # TPR and FPR for each group
    tp_a = ((y_pred_a_grp == 1) & (y_true_a_grp == 1)).sum()
    fn_a = ((y_pred_a_grp == 0) & (y_true_a_grp == 1)).sum()
    fp_a = ((y_pred_a_grp == 1) & (y_true_a_grp == 0)).sum()
    tn_a = ((y_pred_a_grp == 0) & (y_true_a_grp == 0)).sum()
    
    tp_b = ((y_pred_b_grp == 1) & (y_true_b_grp == 1)).sum()
    fn_b = ((y_pred_b_grp == 0) & (y_true_b_grp == 1)).sum()
    fp_b = ((y_pred_b_grp == 1) & (y_true_b_grp == 0)).sum()
    tn_b = ((y_pred_b_grp == 0) & (y_true_b_grp == 0)).sum()
    
    tpr_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
    tpr_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0
    fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
    
    # Equalized Odds: min(TPR_A/TPR_B, FPR_A/FPR_B)
    tpr_ratio = tpr_a / tpr_b if tpr_b > 0 else float("inf")
    fpr_ratio = fpr_a / fpr_b if fpr_b > 0 else float("inf")
    eo = min(tpr_ratio, fpr_ratio) if (tpr_ratio != float("inf") and fpr_ratio != float("inf")) else float("inf")
    
    # Predictive Parity: P(Y=1|Ŷ=1,A) / P(Y=1|Ŷ=1,B)
    ppv_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
    ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    pp = ppv_a / ppv_b if ppv_b > 0 else float("inf")
    
    # Overall accuracy at this threshold
    acc_t = accuracy_score(y_test_arr, y_pred_t)
    
    fairness_results.append({
        "threshold": round(t, 2),
        "DP": round(dp, 4),
        "EO": round(eo, 4),
        "PP": round(pp, 4),
        "accuracy": round(acc_t, 4),
        "TPR_A": round(tpr_a, 4), "TPR_B": round(tpr_b, 4),
        "FPR_A": round(fpr_a, 4), "FPR_B": round(fpr_b, 4),
        "P_pos_A": round(p_pos_a, 4), "P_pos_B": round(p_pos_b, 4),
        "PPV_A": round(ppv_a, 4), "PPV_B": round(ppv_b, 4),
    })

fairness_df = pd.DataFrame(fairness_results)
print("\n--- Fairness Metrics Across Thresholds ---")
print(fairness_df[["threshold", "DP", "EO", "PP", "accuracy",
                    "TPR_A", "TPR_B", "FPR_A", "FPR_B"]].to_string(index=False))

# --- Trade-off Chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Chart 1: DP vs EO scatter with threshold labels
ax1 = axes[0]
dp_vals = fairness_df["DP"].values
eo_vals = fairness_df["EO"].values
pp_vals = fairness_df["PP"].values
t_vals = fairness_df["threshold"].values

scatter = ax1.scatter(dp_vals, eo_vals, c=t_vals, cmap="viridis", s=120,
                       edgecolors="black", linewidths=1.2, zorder=3)
for i, t in enumerate(t_vals):
    ax1.annotate(f"t={t:.2f}", (dp_vals[i], eo_vals[i]),
                 textcoords="offset points", xytext=(8, 8), fontsize=8,
                 fontweight="bold")
ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Ideal EO=1.0")
ax1.axvline(x=1.0, color="blue", linestyle="--", alpha=0.5, label="Ideal DP=1.0")
ax1.set_xlabel("Demographic Parity (DP)", fontsize=12)
ax1.set_ylabel("Equalized Odds (EO)", fontsize=12)
ax1.set_title("Fairness Trade-Off: DP vs EO", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim(0, max(dp_vals) * 1.3)
ax1.set_ylim(0, max(eo_vals) * 1.3)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label="Threshold")

# Chart 2: All three metrics across thresholds
ax2 = axes[1]
ax2.plot(t_vals, dp_vals, "o-", label="Demographic Parity", linewidth=2, markersize=7)
ax2.plot(t_vals, eo_vals, "s-", label="Equalized Odds", linewidth=2, markersize=7)
ax2.plot(t_vals, pp_vals, "^-", label="Predictive Parity", linewidth=2, markersize=7)
ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal (1.0)")
ax2.set_xlabel("Decision Threshold", fontsize=12)
ax2.set_ylabel("Fairness Metric Value", fontsize=12)
ax2.set_title("Fairness Metrics Across Thresholds", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figures", "part2_4_fairness_tradeoff.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n[Figure saved: figures/part2_4_fairness_tradeoff.png]")

# --- Quantify trade-offs ---
print("\n--- Trade-Off Quantification ---")
# Find adjacent thresholds where DP improves but EO drops (or vice versa)
for i in range(1, len(fairness_df)):
    dp_change = fairness_df.iloc[i]["DP"] - fairness_df.iloc[i-1]["DP"]
    eo_change = fairness_df.iloc[i]["EO"] - fairness_df.iloc[i-1]["EO"]
    
    # We want to find where improving DP (moving closer to 1) degrades EO
    dp_improvement = abs(1.0 - fairness_df.iloc[i]["DP"]) < abs(1.0 - fairness_df.iloc[i-1]["DP"])
    eo_degradation = abs(1.0 - fairness_df.iloc[i]["EO"]) > abs(1.0 - fairness_df.iloc[i-1]["EO"])
    
    if dp_improvement and eo_degradation:
        dp_gain_pct = abs(dp_change / fairness_df.iloc[i-1]["DP"]) * 100
        eo_loss_pct = abs(eo_change / fairness_df.iloc[i-1]["EO"]) * 100
        tradeoff_cost = eo_loss_pct / dp_gain_pct if dp_gain_pct > 0 else float("inf")
        
        print(f"  Threshold {fairness_df.iloc[i-1]['threshold']:.2f} -> {fairness_df.iloc[i]['threshold']:.2f}:")
        print(f"    DP: {fairness_df.iloc[i-1]['DP']:.4f} -> {fairness_df.iloc[i]['DP']:.4f} "
              f"({dp_gain_pct:.1f}% closer to ideal)")
        print(f"    EO: {fairness_df.iloc[i-1]['EO']:.4f} -> {fairness_df.iloc[i]['EO']:.4f} "
              f"({eo_loss_pct:.1f}% farther from ideal)")
        print(f"    Trade-off cost: {tradeoff_cost:.2f}")

# ═════════════════════════════════════════════════════════════════════════════
#  SAVE ALL RESULTS TO CSV FOR REPORT
# ═════════════════════════════════════════════════════════════════════════════
fairness_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_metrics.csv"), index=False)
ovb_df.to_csv(os.path.join(OUTPUT_DIR, "omitted_variable_bias.csv"), index=False)
df_adversarial.to_csv(os.path.join(OUTPUT_DIR, "adversarial_subset.csv"), index=False)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE — All figures saved to figures/ directory")
print("=" * 80)
