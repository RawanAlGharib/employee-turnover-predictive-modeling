# ==============================================================================
# PREDICTING EMPLOYEE TURNOVER: HR ANALYTICS & MACHINE LEARNING
# Objective: Identify the primary drivers of employee attrition and build a 
# predictive classification model to flag employees at risk of leaving.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Notebook Formatting
pd.set_option('display.max_columns', None)

# ------------------------------------------------------------------------------
# 2. DATA IMPORT AND CLEANING
# ------------------------------------------------------------------------------
print("--- Loading and Cleaning Data ---")
# Load dataset
df0 = pd.read_csv("HR_dataset.csv")

# Standardize column names
df0 = df0.rename(columns={
    'Work_accident': 'work_accident',
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'tenure',
    'Department': 'department'
})

# Drop duplicates (approx 20% of the dataset)
df1 = df0.drop_duplicates(keep='first')

# Identify outliers in tenure using IQR
percentile25 = df1['tenure'].quantile(0.25)
percentile75 = df1['tenure'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print(f"Tenure lower limit: {lower_limit}")
print(f"Tenure upper limit: {upper_limit}")

# ------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
# Insight: A massive cluster of employees who left the company were working 
# between 240 and 315 hours per month, correlating heavily with satisfaction 
# levels approaching zero.

# Visualize monthly hours vs. satisfaction levels
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='Standard 166.67 hrs/mo', ls='--')
plt.legend()
plt.title('Monthly Hours by Satisfaction Level', fontsize=14)
plt.show()

# Visualize department attrition 
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1, hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='45')
plt.title('Attrition Counts by Department', fontsize=14)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(df1.corr(numeric_only=True), vmin=-1, vmax=1, annot=True, cmap="vlag")
plt.title('Correlation Heatmap', fontsize=14)
plt.show()

# ------------------------------------------------------------------------------
# 4. DATA PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
# To prevent data leakage and improve model robustness, we drop the subjective 
# `satisfaction_level` metric. We also engineer a new binary feature, 
# `overworked`, to flag employees logging more than 175 hours per month.

# Copy dataframe and encode categorical variables
df_enc = df1.copy()
df_enc['salary'] = df_enc['salary'].astype('category').cat.set_categories(['low', 'medium', 'high']).cat.codes
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Feature Engineering: Create 'overworked' and drop variables prone to leakage
df2 = df_enc.drop(['satisfaction_level', 'average_monthly_hours'], axis=1)
df2['overworked'] = (df_enc['average_monthly_hours'] > 175).astype(int)

# Isolate target and features
y = df2['left']
X = df2.drop('left', axis=1)

# Train-Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# ------------------------------------------------------------------------------
# 5. MACHINE LEARNING: RANDOM FOREST MODEL
# ------------------------------------------------------------------------------
print("\n--- Training Random Forest Model (This may take a few minutes) ---")
# Instantiate model and define hyperparameters
rf = RandomForestClassifier(random_state=0)
cv_params = {
    'max_depth': [3, 5, None], 
    'max_features': [1.0],
    'max_samples': [0.7, 1.0],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [300, 500]
}  

# GridSearch Cross Validation optimized for AUC
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# Fit the model
rf2.fit(X_train, y_train)

# Output best AUC score
print(f"Best cross-validation AUC score: {rf2.best_score_:.4f}")

# ------------------------------------------------------------------------------
# 6. MODEL EVALUATION & BUSINESS INSIGHTS
# ------------------------------------------------------------------------------

# Generate predictions on the unseen test set
preds = rf2.best_estimator_.predict(X_test)

# Print comprehensive classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, preds))

# Plot confusion matrix
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Departed'])
disp.plot(values_format='')
plt.title('Random Forest Confusion Matrix')
plt.show()

# ==============================================================================
# STRATEGIC RECOMMENDATIONS:
# 1. Implement Workload Caps: Hard caps must be implemented to prevent top-performer burnout (>175 hrs).
# 2. Review the Four-Year Mark: Retention drops significantly around year four.
# 3. Restructure Incentives: Reward output quality rather than sheer hours logged.
# ==============================================================================