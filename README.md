# 🛡️ Credit Card Fraud Detection Using Machine Learning

A machine learning project focused on detecting fraudulent credit card transactions. The dataset is highly imbalanced, so this project compares two approaches to handling that imbalance: **SMOTE oversampling** (baseline) and **cost-sensitive learning via class weighting** (improved), evaluated with Stratified K-Fold Cross-Validation throughout.

---

## 📌 Introduction

Credit card fraud detection is a major challenge due to the highly imbalanced nature of real-world transaction data, where fraudulent activities are rare but extremely costly. This project builds a baseline fraud detection model using Logistic Regression with SMOTE, then improves on it using XGBoost with class weighting — prioritizing high recall throughout, since missing actual fraud is far costlier than a false alarm.

---

## 🎯 Objectives

- Build a predictive model to classify fraudulent transactions.
- Handle extreme class imbalance using appropriate resampling and cost-sensitive methods.
- Evaluate models using recall, precision, F1-score, ROC-AUC, **and PR-AUC** — the metric best suited to severe class imbalance.
- Ensure robust, leakage-free model validation using Stratified K-Fold Cross-Validation.
- Tune the decision threshold using out-of-fold predictions, never the test set, to avoid leakage.

---

## 🗂️ Dataset Information

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Frauds:** 492 (approx. 0.17%)
- **Features:**
  - `V1` to `V28`: PCA-transformed features
  - `Amount` and `Time`: Raw features
  - `Class`: Target variable (1 = Fraud, 0 = Normal)

---

## ⚙️ What Was Done

### ✔️ Data Preprocessing
- Checked for missing values (none present).
- Scaled `Amount` and `Time` using `StandardScaler` — required for Logistic Regression, since it's a linear, distance-sensitive model.
- Applied `train_test_split` with stratification to preserve class balance.

### ✔️ Approach 1 — Logistic Regression + SMOTE (Baseline)
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) only on the training data to synthetically generate minority class samples.
- Trained **Logistic Regression** with default parameters on the SMOTE-balanced training set.
- Validated with **Stratified 5-Fold Cross-Validation**.

### ✔️ Approach 2 — XGBoost + Cost-Sensitive Learning (Improved)
- Trained **XGBoost** directly on the original, unscaled, imbalanced training data — no SMOTE.
- Handled imbalance via `scale_pos_weight` (cost-sensitive learning), which penalizes misclassifying the minority (fraud) class without altering the training data.
- Tree-based models are scale-invariant, so no `StandardScaler` was needed for this approach.
- Tuned the decision threshold using **leakage-safe pooled out-of-fold (OOF) predictions**: each training row received a prediction from a model that never saw it during training (5-fold rotation), and all predictions were pooled before selecting a single threshold — avoiding both test-set leakage and the instability of tuning on small individual folds (the dataset has only ~492 total fraud cases).
- The tuned threshold was applied exactly once on the untouched test set.

### ✔️ Evaluation
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC-AUC and **PR-AUC** (Precision-Recall AUC) — PR-AUC is reported as the primary metric, since ROC-AUC can overstate performance under extreme imbalance like this dataset's 0.17% fraud rate.

---

## ✅ Results

| Metric | LR + SMOTE (Baseline) | XGBoost + Class Weighting (Tuned) |
|---|---|---|
| Recall (Fraud) | 0.9184 | 0.9184 |
| Precision (Fraud) | 0.0580  | 0.0930 |
| F1-score | 0.11 | 0.1689 |
| ROC-AUC | 0.9699 | 0.9682 |
| **PR-AUC** | 0.7249 | **0.8800** |
| False Alarms (FP) | 1,461 | 878 |

**Confusion Matrix — LR + SMOTE:**
```
[[55403  1461]
 [    8    90]]
```

**Confusion Matrix — XGBoost + Class Weighting (tuned threshold):**
```
[[55986   878]
 [    8    90]]
```

At matched recall (both models catch the same 90 of 98 frauds), switching from SMOTE to cost-sensitive XGBoost nearly **doubles precision** and cuts false alarms by **40%** (1,461 → 878), while substantially improving PR-AUC (0.7249 → 0.8800) — the metric most appropriate for this level of class imbalance.

---

## 🔍 Key Insights

- Both models achieve the same high recall, effectively catching the vast majority of fraudulent transactions.
- **PR-AUC, not ROC-AUC, is the appropriate headline metric** for this dataset: with frauds at just 0.17% of transactions, ROC-AUC can look strong even when a model isn't precisely separating the two classes, while PR-AUC reflects that separation more honestly.
- Cost-sensitive learning on a tree-based model (XGBoost) substantially reduces false alarms compared to SMOTE-based Logistic Regression, at the same recall level — a more deployable trade-off for a real fraud review workflow.
- Threshold tuning on a dataset with very few positive (fraud) cases is unstable if done naively per cross-validation fold; pooling out-of-fold predictions before tuning gives a far more stable, trustworthy threshold.

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```

---

## 🔧 What Can Be Improved

- Explore additional models (Random Forest, LightGBM, Isolation Forest) for further comparison.
- Add model explainability tools like SHAP or LIME to understand which features drive fraud predictions.
- Explore time-based features (e.g., transaction hour, day, time gap between transactions).
- Investigate ensemble approaches combining multiple cost-sensitive models.

---

## 🧠 Final Thoughts

This project demonstrates an iterative approach to handling extreme class imbalance in fraud detection: starting from a standard SMOTE + Logistic Regression baseline, then improving on it with cost-sensitive learning on a tree-based model and leakage-free threshold tuning. At matched recall, the improved model nearly doubles precision and substantially reduces false alarms, while using PR-AUC — a more honest metric for this level of imbalance — to guide the comparison.

---

## 📜 License

This project is under the MIT License.

---

## 👤 Author

**Vivek Negi**
_Data Analyst | Machine Learning Enthusiast_
