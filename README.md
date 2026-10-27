# Credit Risk Scoring Engine (Loan Default Prediction)

## 📌 Business Context
Financial institutions lose millions annually to loan defaults. The goal of this project is to build a machine learning model that predicts the probability of a borrower defaulting on a loan. This enables underwriters to make data-driven decisions, reduce risk, and automate the approval process for low-risk applicants.

## 📊 Dataset
* **Source:** Kaggle (Loan Default Dataset)
* **Size:** ~255,000 rows
* **Target Variable:** `Default` (1 = Defaulted, 0 = Paid back)
* **Base Default Rate:** 11.61% (Imbalanced dataset)

## 🛠️ Approach & Technical Steps

### 1. Data Preparation & Cleaning
Real-world data is messy. Before modeling, we performed the following:
* **Missing Value Imputation:** Filled numeric missing values with the median and categorical missing values with the mode to prevent data loss.
* **Feature Engineering:** Created a custom feature `Income_to_Loan_Ratio` to identify over-leveraged applicants. 
* **Encoding:** Dropped the irrelevant `LoanID` column and used `LabelEncoder` to convert text categories (e.g., Education, EmploymentType) into numeric formats.
* **Scaling:** Applied `StandardScaler` to normalize numerical features, ensuring the Logistic Regression baseline was not biased by large numbers (like Income).

### 2. Exploratory Data Analysis (EDA)
* Analyzed the distribution of defaults across different demographics.
* Identified that the dataset is highly imbalanced (only ~11.6% defaults). This guided our choice to use `class_weight='balanced'` and `scale_pos_weight` to prevent the model from simply predicting "No Default" every time.

### 3. Modeling
Trained and compared two models using an 80/20 stratified train-test split:
1. **Baseline:** Logistic Regression.
2. **Advanced:** XGBoost Classifier.

### 4. Explainability (SHAP)
A black-box model is unacceptable in finance. We used **SHAP (SHapley Additive exPlanations)** to explain the model:
* **Global Summary Plot:** Revealed that `Age`, `InterestRate`, and our custom `Income_to_Loan_Ratio` were the top drivers of default risk.
* **Waterfall Plot:** Generated per-applicant reason codes. This provides regulators with an exact breakdown of *why* a specific applicant was approved or declined.

### 5. Output for Power BI
Exported the test set predictions (`PowerBI_Credit_Risk_Data.csv`) containing the actual default, predicted probability, and final decision (Approve/Decline) for dashboard reporting.

## 💻 Tech Stack
* **Data Processing:** Python, Pandas, NumPy, Scikit-Learn (Preprocessing)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainability:** SHAP
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or install pandas, scikit-learn, xgboost, shap, matplotlib, seaborn).
3. Place your dataset in the `data/` folder.
4. Run the script: `python data/src/loan_default_prediction.py`