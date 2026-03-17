# Financial Crime Detection — Machine Learning Credit Card Fraud Detection Pipeline

## Project Overview
This project develops a machine learning pipeline to detect fraudulent credit card transactions in a highly imbalanced dataset where fraudulent activity represents approximately **0.17% of all transactions**.

The primary objective is to design a model that prioritises **high recall**, ensuring that as many fraudulent transactions as possible are detected while maintaining a manageable false-positive rate.

Fraud detection represents a common **financial risk and anomaly detection problem**, requiring specialised modelling strategies due to extreme class imbalance.

---

# Dataset

The dataset contains anonymised credit card transaction records with engineered numerical features.

Key characteristics:

- **Total transactions:** ~56,000
- **Fraud cases:** 98 (~0.17%)
- **Features:** anonymised numerical variables (V1–V28) and transaction amount
- **Target variable:** Fraud (1) or Legitimate (0)

The extreme imbalance between fraud and legitimate transactions requires dedicated techniques to ensure the model effectively learns minority-class patterns.

---

# Methodology

## 1. Exploratory Data Analysis

Initial analysis focused on understanding feature relationships and distributions.

Techniques used:

- **Correlation matrix** to identify strongly related features
- **Boxplots** to inspect distributions and detect potential outliers

These steps helped identify variables with stronger predictive signals.

---

## 2. Data Preprocessing

### Outlier Handling
Outliers were addressed using the **Interquartile Range (IQR) method**, reducing noise while preserving anomaly signals.

### Feature Scaling
Transaction features were scaled using **RobustScaler**, which is resistant to extreme values and suitable for financial transaction data.

---

## 3. Handling Class Imbalance

The dataset contains only **0.17% fraudulent transactions**, creating a significant class imbalance.

To address this:

- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied
- Synthetic fraud samples were generated to improve model learning on the minority class

---

## 4. Baseline Model

A **Decision Tree classifier** was first trained to establish baseline performance and understand feature behaviour.

Evaluation metrics included:

- Confusion Matrix
- Precision
- Recall
- F1-score

---

## 5. Model Development

Multiple models were trained and compared.

### Random Forest

A **Random Forest classifier** was trained to improve robustness and reduce overfitting relative to the baseline model.

Feature importance scores were extracted and visualised to identify the most influential predictors.

This step helped guide feature engineering and improve model interpretability.

---

### XGBoost Model

The final model used **XGBoost**, a gradient boosting algorithm known for strong performance on structured datasets.

Hyperparameters were tuned using **RandomizedSearchCV** to efficiently search the parameter space.

Additional improvements included:

- Feature engineering (ratios and interaction features)
- Creation of a custom **risk flag feature**
- Model evaluation across multiple metrics

---

## 6. Feature Importance Analysis

Feature importance scores from the Random Forest and XGBoost models were analysed to identify the most predictive variables.

The **top features were visualised**, providing insight into which transaction characteristics contribute most to fraud detection.

This analysis also guided additional feature engineering.

---

## 7. Decision Threshold Optimization

Rather than using the default classification threshold (0.5), the model threshold was adjusted to **0.1**.

This prioritises fraud detection (recall) while controlling the number of false alerts.

This approach better aligns with real-world fraud detection priorities where **missing fraud cases is more costly than reviewing legitimate transactions**.

---

# Results

| Metric | Value |
|------|------|
| Recall | **87.7%** |
| Fraud Cases Detected | **86 / 98** |
| False Positives | **65 / 56,000+ transactions** |

The model demonstrates strong ability to detect fraudulent transactions while maintaining a low false-positive rate.

---

# Inference Pipeline

A reproducible inference pipeline was implemented to ensure consistent preprocessing and scoring of incoming transaction data.

Pipeline components include:

- Data validation and preprocessing
- Feature scaling using **RobustScaler**
- Model loading using **Joblib**
- Automated fraud prediction

This structure enables easy integration into downstream systems or reporting tools.

---

# Project Structure


---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- SMOTE (imbalanced-learn)
- Matplotlib / Seaborn

---

# How to Run

Install dependencies:

```bash
pip install -r requirements.txt
notebooks/fraud_detection_analysis.ipynb

---

# Future Improvements

Potential enhancements include:

- Real-time fraud detection pipeline
- Deployment as an API using FastAPI


---


JY
