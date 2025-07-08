# 🔄 Telco Customer Churn Prediction | End-to-End ML Pipeline 🚀

Built a production-ready machine learning pipeline using Scikit-learn to predict customer churn. Includes full preprocessing, model tuning with GridSearchCV, evaluation, and pipeline export with joblib.

A complete machine learning pipeline built using **Scikit-learn** to predict customer churn based on subscription and service usage data from a telecom company. This project demonstrates how to construct a reusable, scalable, and production-ready ML system using real-world practices.


## 🎯 Objective

The objective of this task is to:

- Predict whether a customer will **churn (leave the service)** or not.
- Build a **clean and modular ML pipeline** using `Pipeline` and `ColumnTransformer`.
- Perform **data preprocessing**, **model training**, **hyperparameter tuning**, and **model export**.
- Make the model **reusable and ready for deployment** using `joblib`.


## 🧠 Methodology / Approach

### 1. **Data Loading & Cleaning**
 - Used the **Telco Customer Churn Dataset**.
 - Cleaned missing values and dropped non-informative columns.
 - Converted `Churn` target to binary labels (`Yes` → `1`, `No` → `0`).

### 2. **Train-Test Split**
 - Split data into **training (80%)** and **testing (20%)** sets using stratification to preserve class balance.

### 3. **Preprocessing with Pipelines**
  - Used `ColumnTransformer` to apply:
  - `StandardScaler` to numerical features.
  - `OneHotEncoder` to categorical features.
  - Created clean pipelines for preprocessing using `Pipeline`.

### 4. **Modeling**
 - Built two machine learning models:
 - **Logistic Regression**
 - **Random Forest Classifier**
 - Combined models with preprocessing pipelines for modularity.

### 5. **Hyperparameter Tuning**
  - Used `GridSearchCV` to find optimal hyperparameters:
  - `C` for Logistic Regression
  - `n_estimators`, `max_depth` for Random Forest

### 6. **Model Evaluation**
- Evaluated on accuracy, precision, recall, and F1-score using `classification_report`.
- Compared performance of both models on the test set.

### 7. **Model Export**
- Exported the **best model pipeline (Random Forest)** using `joblib` for reuse and deployment.



## 📊 Key Results / Observations

| Model                | Accuracy (Test Set) | Observations |
|---------------------|---------------------|--------------|
| Logistic Regression | ~80%                | Simpler, linear model |
| Random Forest       | ~83%                | Better performance due to non-linearity |

- **Top churn indicators**: month-to-month contract, high monthly charges, low tenure, lack of tech support.
- **Random Forest** model selected as the final model due to superior performance.

**💾 Reusability:**

To load and use the saved pipeline:

import joblib

Load saved model

model = joblib.load('telco_churn_pipeline.pkl')

Predict on new data

predictions = model.predict(X_new)


**🚀 Future Improvements:**

 Add feature importance and visual explanations

 Deploy the model using Flask or Streamlit

 Build a REST API for real-time predictions

 Monitor model performance over time (production monitoring)


**📚 Tech Stack:**

Python 3

pandas, numpy, matplotlib, seaborn

scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, joblib)

Google Colab

**📥 Dataset:**

Telco Customer Churn Dataset

Source: Kaggle

Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn






