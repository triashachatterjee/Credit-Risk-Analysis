Loan Default Prediction and Risk Modeling

Project Overview

This project focuses on building and evaluating predictive models to assess the probability of loan default (a binary classification problem). The goal is to leverage advanced data analysis, feature engineering, and machine learning techniques (Logistic Regression, Random Forest, XGBoost) to provide actionable insights for credit risk management.

The project is structured into two main Jupyter notebooks covering the end-to-end data science lifecycle: Data Inspection & Preprocessing, and Model Training & Evaluation.

Repository Contents

File

Description

Key Activities

Data_Inspection.ipynb

Data Preparation and Feature Engineering. This notebook handles the initial data loading, inspection, cleaning, transformation, and encoding of the raw dataset.

Missing Value Imputation, Exploratory Data Analysis (EDA), Feature Selection, Categorical Encoding (One-Hot), Data Saving (df_encoded.csv), and Target Correlation analysis.

Model_Training.ipynb

Predictive Modeling and Evaluation. This notebook loads the prepared data, splits it for training and testing, scales numerical features, trains multiple classification models, and evaluates their performance.

Data Scaling (StandardScaler), Model Training (LogisticRegression, RandomForestClassifier, XGBClassifier), Performance Metrics (AUC-ROC, Classification Report, Confusion Matrix), and Feature Importance Analysis (XGBoost).

Raw_Data/df_encoded.csv

(Output Data) The fully processed and encoded dataset, ready for immediate model training.



Raw_Data/target_correlations.csv

(Output Data) Series containing the correlation of all features with the target variable (is_default).



Data Requirements

This project expects a specific file structure for data loading.

The following file is required for the notebooks to run:

Raw_Data/accepted_Q4_list.csv (The raw input data file, referenced in Data_Inspection.ipynb).

Please ensure the Raw_Data directory exists and contains the necessary files.

Setup and Installation

1. Environment Setup

It is highly recommended to create a virtual environment for this project.

# Create a new environment (e.g., using conda)
conda create -n loan_risk python=3.9
conda activate loan_risk


2. Install Dependencies

The following key libraries are required. They can be installed using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime


3. Running the Notebooks

Place your raw dataset (accepted_Q4_list.csv) inside the Raw_Data directory.

Run Data_Inspection.ipynb first. This will process the data and save the intermediate files (df_encoded.csv, target_correlations.csv) back into the Raw_Data directory.

Once the output files are generated, run Model_Training.ipynb to load the clean data and proceed with model development and evaluation.

Key Features

Comprehensive Preprocessing: Robust handling of missing data and categorical features.

Multiple Model Comparison: Benchmarking performance across linear (Logistic Regression) and ensemble methods (Random Forest, XGBoost).

Model Explainability: Includes preparation for advanced interpretability techniques (SHAP, LIME) to understand model decisions (though the main usage is concentrated on feature importance).

Performance Focus: Primary evaluation metric is AUC-ROC, essential for binary classification in imbalanced financial datasets.
