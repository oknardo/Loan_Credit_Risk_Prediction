# 💳 Loan Credit Risk Prediction

**Author:** Oknardo Tulung  
**Role:** Data Scientist Intern — ID/X Partners  
**LinkedIn:** https://www.linkedin.com/in/oknardo-tulung/  
**GitHub:** https://github.com/oknardo/Loan_Credit_Risk_Prediction  

---

## 📌 Project Overview

This project was developed as a final task during an internship at **ID/X Partners**, conducted in collaboration with a lending company (multifinance). The goal is to build a machine learning model that predicts credit risk based on historical Lending Club loan data (2007–2014), enabling the lending company to make more accurate and data-driven credit approval decisions.

The project covers a complete end-to-end data science workflow, from exploratory data analysis and data preparation to model training, evaluation, and borrower risk scoring.

---

## 🎯 Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) on the loan dataset
- Build and evaluate 6 machine learning models for credit default prediction
- Apply hyperparameter tuning to identify the best-performing model
- Generate default probability scores and risk categories for borrowers
- Deliver actionable insights and business impact simulations for the lending company

---

## 📂 Dataset Overview

The project uses a single dataset from Lending Club's historical loan records:

| Dataset | Rows | Features | Description |
|---|---|---|---|
| `loan_data_2007_2014.csv` | 466,285 | 75 | Historical loan records with borrower profiles, loan characteristics, and repayment outcomes |

**Target Variable:** `loan_group` (Good Loan / Bad Loan)  
**Resolved Loans (used for modeling):** 230,795 rows after removing Ongoing loans  
**Default Rate:** 19.09% (Bad Loan)

---

## 📓 Notebook Structure

| # | Notebook | Description |
|---|---|---|
| 1 | `EDA_loan_data.ipynb` | Comprehensive EDA on the loan dataset |
| 2 | `Data_Cleaning_Handling.ipynb` | Data cleaning, feature engineering, encoding, scaling, and train/test split |
| 3 | `Train_Model.ipynb` | Model training, evaluation, hyperparameter tuning, SHAP analysis, and business simulation |
| 4 | `Prediction_Model.ipynb` | Credit risk scoring and borrower risk segmentation |

---

## 🔍 EDA Key Findings

### Target Distribution
- Dataset contains 3 loan groups: Good Loan (40.0%), Bad Loan (9.5%), and Ongoing (50.5%)
- Among resolved loans, default rate is **19.09%**, indicating moderate class imbalance

### Numerical Features
- `int_rate` is the strongest predictor with correlation of 0.24 against default
- `dti` shows consistent increasing trend where higher debt burden leads to higher default risk
- `annual_inc` shows inverse relationship — higher income associated with lower default risk
- `revol_util` shows steady increasing trend where higher utilization is associated with higher risk

### Categorical Features
- Loan grade shows perfect monotonic relationship with default risk (Grade A ~6% to Grade G ~42%)
- 60-month loans have significantly higher default rate (~31%) vs 36-month loans (~16%)
- `small_business` purpose has highest default rate (~30%) among all loan purposes
- `Verified` borrowers show counterintuitively higher default rate (~22%) than unverified (~15%)

### Correlation Analysis
- `int_rate` is the strongest legitimate predictor (0.24), post-origination features excluded
- `loan_amnt`, `funded_amnt`, and `installment` are almost perfectly correlated (0.96–1.00)
- No single feature is strongly predictive on its own, confirming the need for multi-feature modeling

---

## ⚙️ Feature Engineering

Key transformations and derived features created during data preparation:

| Feature | Source | Description |
|---|---|---|
| `loan_group` | `loan_status` | Binary target: Good Loan / Bad Loan |
| `issue_year` | `issue_d` | Loan vintage year extracted from issue date |
| `credit_age_years` | `earliest_cr_line`, `issue_d` | Years since first credit line opened |
| `emp_length` | `emp_length` | Employment length converted to ordinal numeric (0–10) |
| `emp_length_MISSING` | `emp_length` | Missing indicator for employment length |

---

## 🤖 Modeling Results

### Model Comparison

| Model | ROC-AUC | Gini | KS Stat | Recall (Bad Loan) | Precision (Bad Loan) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7073 | 0.4145 | 0.3070 | **0.6450** | 0.3071 |
| Decision Tree | 0.5476 | 0.0952 | 0.0952 | 0.2774 | 0.2644 |
| Random Forest | 0.7039 | 0.4078 | 0.2965 | 0.0382 | 0.5177 |
| KNN | 0.6197 | 0.2394 | 0.1811 | 0.1472 | 0.3604 |
| XGBoost | 0.7120 | 0.4241 | 0.3171 | 0.6291 | 0.3208 |
| LightGBM | 0.7180 | 0.4360 | 0.3229 | 0.6668 | 0.3125 |
| **LightGBM Tuned** | **0.7209** | **0.4418** | **0.3290** | 0.6600 | 0.3200 |

**LightGBM Tuned** achieves the best overall performance with ROC-AUC of **0.7209**, Gini of **0.4418**, and KS Statistic of **0.3290**.

### Top Features (SHAP Analysis — LightGBM Tuned)

| Rank | Feature | Direction |
|---|---|---|
| 1 | `annual_inc` | Higher = Lower Risk |
| 2 | `int_rate` | Higher = Higher Risk |
| 3 | `grade` | Higher Grade = Higher Risk |
| 4 | `dti` | Higher = Higher Risk |
| 5 | `term_60 months` | 60 months = Higher Risk |
| 6 | `sub_grade` | Higher = Higher Risk |
| 7 | `loan_amnt` | Higher = Higher Risk |
| 8 | `issue_year` | Earlier Year = Higher Risk |
| 9 | `revol_util` | Higher = Higher Risk |
| 10 | `open_acc` | Higher = Lower Risk |

---

## 📊 Credit Risk Scoring Results

Applied to **46,159 borrowers** from the test set:

| Risk Category | Count | Percentage | Avg Probability | Min | Max |
|---|---|---|---|---|---|
| Very Low (0.00–0.20) | 5,796 | 12.56% | 0.1312 | 0.0050 | 0.2000 |
| Low (0.20–0.40) | 15,208 | 32.95% | 0.3049 | 0.2001 | 0.4000 |
| Medium (0.40–0.60) | 14,493 | 31.40% | 0.4964 | 0.4001 | 0.6000 |
| High (0.60–0.80) | 9,531 | 20.65% | 0.6885 | 0.6001 | 0.8000 |
| Very High (0.80–1.00) | 1,131 | 2.45% | 0.8340 | 0.8002 | 0.9361 |

---

## 💰 Business Simulation

**Baseline (No Model):**
- Total defaulters in test set: 8,814 borrowers
- Estimated total loss: **USD 76.42 Million** (assuming 60% Loss Given Default)

**Model Impact at Threshold 0.5 (Recommended):**
- Defaults prevented: **5,653 borrowers** (64.1% recall)
- Loss prevented: **USD 49.01 Million**
- Opportunity cost (good borrowers rejected): USD 155.68 Million

**Key Insight:**
At all tested thresholds (0.20–0.50), opportunity cost exceeds loss prevented due to low precision. Threshold 0.5 remains the most cost-efficient option with the highest precision (32.31%). Achieving positive net benefit requires precision improvement above 60% through further feature engineering or additional data sources.

---

## 🛠 Tech Stack

- **Language**: Python 3
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Hyperparameter Tuning**: Optuna
- **Interpretability**: SHAP
- **Model Persistence**: Joblib

---

## 📁 Project Structure
Loan_Credit_Risk_Prediction  
├── EDA_loan_data.ipynb  
├── Data_Cleaning_Handling.ipynb  
├── Train_Model.ipynb  
├── Prediction_Model.ipynb  
├── train_set.csv  
├── test_set.csv  
├── prediction_results.csv  
├── lgbm_tuned.pkl  
└── README.md  

---
---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/oknardo/Loan_Credit_Risk_Prediction.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run notebooks in order:
`EDA_loan_data` → `Data_Cleaning_Handling` → `Train_Model` → `Prediction_Model`

---

## 📄 License  

This project is developed for internship and portfolio purposes at ID/X Partners.