# 🛡️ Credit Card Fraud Detection Using Logistic Regression

A machine learning project focused on detecting fraudulent credit card transactions using Logistic Regression. The dataset is highly imbalanced, so techniques like SMOTE and Stratified K-Fold Cross-Validation are used to enhance model performance and generalization.

---

## 📌 Introduction

Credit card fraud detection is a major challenge due to the highly imbalanced nature of real-world transaction data, where fraudulent activities are rare but extremely costly. This project uses Logistic Regression to build a baseline fraud detection model that prioritizes high recall to reduce undetected fraud.

---

## 🎯 Objectives

- Build a predictive model to classify fraudulent transactions.
- Handle extreme class imbalance using appropriate resampling methods.
- Evaluate the model using recall, precision, F1-score, and ROC-AUC instead of just accuracy.
- Ensure robust model validation using Stratified K-Fold Cross-Validation.

---

## 🗂️ Dataset Information

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Frauds:** 492 (approx. 0.17%)
- **Features:**
  - `V1` to `V28`: PCA-transformed features
  - `Amount` and `Time`: Raw features (scaled)
  - `Class`: Target variable (1 = Fraud, 0 = Normal)

---

## ⚙️ What Was Done

### ✔️ Data Preprocessing
- Checked for missing values (none present).
- Scaled `Amount` and `Time` using `StandardScaler`.
- Applied `train_test_split` with stratification to preserve class balance.

### ✔️ Handling Imbalanced Data
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) only on the training data to synthetically generate minority class samples.

### ✔️ Model Building
- Used **Logistic Regression** with default parameters.
- Evaluated with:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - ROC-AUC Score
  - Stratified K-Fold Cross-Validation

### ✔️ Performance
- **Test ROC-AUC:** `0.9699`
- **Fraud Recall:** `0.9184` ✅ (very strong)
- **Fraud Precision:** `0.0580` ⚠️ (low, common in fraud detection)
- **Confusion Matrix:**
  ```
  [[55403 1461]
   [    8   90]]
  ```

---

## 📊 Visualizations

- Class imbalance visualization
- Distribution of `Amount` and `Time` by class
- ROC Curve
- Confusion Matrix heatmap

---

## 🔍 Key Insights

- The model effectively **detects most fraudulent transactions** (high recall).
- However, it suffers from **low precision**, meaning it produces many false alarms — a trade-off often accepted in fraud detection.
- High ROC-AUC and CV scores indicate **good generalization** and model stability.

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-logistic.git
   cd credit-card-fraud-logistic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```

---

## 🔧 What Can Be Improved

- Try **threshold tuning** to improve fraud precision without losing too much recall.
- Experiment with **other algorithms** like XGBoost or Isolation Forest.
- Add **model explainability** tools like SHAP or LIME.
- Use **SMOTE with TomekLinks** or **SMOTE-ENN** for better sampling.
- Explore **time-based features** (e.g., transaction hour, day, time gap between transactions).

---

## 🧠 Final Thoughts

This project builds a solid foundation for fraud detection using logistic regression. With high recall and robust evaluation, it’s an excellent start toward real-world deployment — while highlighting areas for further optimization.

---

## 📜 License

This project is under the MIT License.

---

## 👤 Author

**Vivek Negi**  
_Data Analyst | Machine Learning Enthusiast_
