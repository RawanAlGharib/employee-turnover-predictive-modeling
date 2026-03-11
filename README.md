# Employee Turnover Prediction: HR Analytics & Machine Learning

## 📌 Project Overview
High employee turnover represents a significant cost to organizations in terms of recruitment, training, and lost productivity. This project leverages human resources data to identify the key drivers of employee attrition and builds a predictive machine learning model to flag employees who are at high risk of leaving the company. 

## 🎯 Business Objectives
* Conduct Exploratory Data Analysis (EDA) to uncover trends and correlations driving employee dissatisfaction and turnover.
* Engineer features from existing HR metrics (such as evaluation scores, working hours, and project loads).
* Develop and evaluate machine learning classification models to accurately predict the binary target variable (`left`).
* Provide actionable, data-driven recommendations to HR stakeholders to improve retention strategies.

## 📊 The Dataset
The dataset contains employee performance and demographic metrics. During the data cleaning phase, variables were standardized for consistency:
* **Target Variable:** `left` (0 = Stayed, 1 = Left)
* **Key Features:** `satisfaction_level`, `last_evaluation`, `number_project`, `average_monthly_hours`, `tenure`, `work_accident`, `promotion_last_5years`, `department`, `salary`

## 🛠️ Tools & Methodology
* **Language:** Python
* **Libraries:** `pandas`, `numpy` (Data Manipulation) | `matplotlib`, `seaborn` (Data Visualization) | `scikit-learn` (Machine Learning)
* **Models Evaluated:** Logistic Regression, Decision Trees, and Random Forest.
* **Evaluation Metrics:** Precision, Recall, F1-Score, and AUC-ROC, with a specific focus on Recall to ensure at-risk employees are accurately captured.

## 📈 Visualizing the Problem & The Solution

### 1. Distinct Attrition Clusters
*The scatterplot below maps monthly working hours against satisfaction levels, revealing three very distinct clusters of departing employees (orange):*
1. **The Burned-Out:** Working massively over the 166.67 standard hours (240–310+ hours) with satisfaction nearing zero (~0.1).
2. **The Under-Utilized/Unsatisfied:** Working below standard hours (130–160) with moderately low satisfaction (~0.4).
3. **The Overworked but Satisfied:** Working heavy hours (220–280) with very high satisfaction (~0.8+), indicating high-performers who might be poached by competitors.

![Satisfaction vs Hours](images/burnout_scatter.png)

### 2. Model Accuracy (Random Forest Confusion Matrix)
*The Confusion Matrix demonstrates the model's high reliability in identifying flight-risk employees.*

![Random Forest Confusion Matrix](images/confusion_matrix.png)

## 💡 Key Insights & Recommendations
1. **Address Extreme Overwork:** The most severe flight risk comes from employees logging 240+ hours per month. HR must monitor the 166.67-hour baseline and intervene when hours begin to spike to prevent satisfaction levels from collapsing.
2. **Investigate the "Satisfied Leavers":** A notable segment of employees leave despite reporting high satisfaction and working long hours. Management should investigate if these individuals are leaving for better compensation elsewhere after building their skills.
3. **Deploy the Predictive Model:** The Random Forest model correctly identified 456 out of 498 departing employees (a ~91.5% recall rate), while only generating 69 false positives. HR can confidently deploy this model to generate proactive monthly "retention risk" alerts.
