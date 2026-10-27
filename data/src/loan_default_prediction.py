import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# 1. LOAD DATA
print("Loading data...")
file_path = r'C:\Users\tejas\PycharmProjects\PythonProject\PythonProject\data\Loan_default.csv'
df = pd.read_csv(file_path)
print(f"✅ Dataset shape: {df.shape}")

# 2. DATA PREPARATION & CLEANING
print("\n--- Data Preparation ---")
# A. Handle Missing Values
print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
# Fill numeric missing values with the median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
# Fill categorical missing values with the mode (most frequent)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# B. Drop Unnecessary Columns
if 'LoanID' in df.columns:
    df = df.drop(columns=['LoanID'])

# C. Feature Engineering
df['Income_to_Loan_Ratio'] = df['Income'] / (df['LoanAmount'] + 1)

# D. Encode Categorical Variables
print("Encoding categorical variables...")
for col in cat_cols:
    if col in df.columns: # Check if not dropped
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- EDA ---")
default_rate = df['Default'].mean()
print(f"Default Rate: {default_rate:.2%}")

plt.figure(figsize=(8, 4))
sns.barplot(x='Education', y='Default', data=df, errorbar=None, palette='viridis')
plt.title('Default Rate by Education Level')
print("Showing EDA chart... (Close the chart window to continue)")
plt.show()

# 4. TRAIN/TEST SPLIT & SCALING
print("\n--- Splitting and Scaling Data ---")
X = df.drop(columns=['Default'])
y = df['Default']

# Stratified split to maintain the 11.6% default rate in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for Logistic Regression (XGBoost doesn't strictly need this, but it's best practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODEL TRAINING
print("\n--- Training Models ---")
# Baseline: Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")

# Advanced: XGBoost
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=scale_weight, eval_metric='auc', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_preds)
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")

# 6. EVALUATION
print("\n--- Classification Report (XGBoost) ---")
print(classification_report(y_test, (xgb_preds > 0.5).astype(int)))

# 7. EXPLAINABILITY (SHAP)
print("\n--- Generating SHAP Explainability Plots ---")
print("(Close each plot window to let the code proceed)")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

applicant_index = 0
shap.plots.waterfall(shap.Explanation(values=shap_values[applicant_index],
                                      base_values=explainer.expected_value,
                                      data=X_test.iloc[applicant_index],
                                      feature_names=X_test.columns.tolist()), max_display=10)

# 8. EXPORT FOR POWER BI
print("\n--- Exporting Data for Power BI ---")
powerbi_df = X_test.copy()
powerbi_df['Actual_Default'] = y_test
powerbi_df['Predicted_Probability'] = xgb_preds
powerbi_df['Decision'] = np.where(xgb_preds > 0.5, 'Decline', 'Approve')

output_path = r'C:\Users\tejas\PycharmProjects\PythonProject\PythonProject\PowerBI_Credit_Risk_Data.csv'
powerbi_df.to_csv(output_path, index=False)
print(f"✅ Successfully saved 'PowerBI_Credit_Risk_Data.csv' to your project folder!")