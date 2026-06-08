# 🏠 NYC Housing Intelligence Platform

## Overview

NYC Housing Intelligence Platform is a professional end-to-end Machine Learning application designed to predict real estate property prices using advanced regression techniques and interactive business intelligence dashboards.

The platform combines data preprocessing, feature engineering, predictive modeling, and visual analytics into a deployment-ready Streamlit application.

---

# Project Objectives

* Predict housing prices with high accuracy
* Analyze real estate market trends
* Generate property valuation insights
* Provide interactive visual analytics
* Demonstrate production-level machine learning workflow
* Showcase deployment-ready AI application development

---

# Features

### Machine Learning

* CatBoost Regressor
* Automated Feature Engineering
* Data Cleaning Pipeline
* Missing Value Handling
* Feature Scaling
* Model Serialization with Joblib
* Real-Time Prediction Engine

### Data Analytics

* Housing Price Distribution
* Correlation Matrix
* Property Statistics
* Building Area Analysis
* Real Estate Trend Insights

### Dashboard Features

* Interactive Streamlit Interface
* KPI Metrics
* Real-Time Property Valuation
* Plotly Visualizations
* Responsive Dashboard Layout
* Cached Data Loading
* Model Performance Monitoring

---

# Dataset Features

The model utilizes the following features:

* zip_code
* yearbuilt
* lotarea
* bldgarea
* resarea
* comarea
* unitsres
* unitstotal
* numfloors
* latitude
* longitude
* building_age

Target Variable:

* sale_price

---

# Machine Learning Workflow

Dataset
↓
Data Cleaning
↓
Missing Value Treatment
↓
Feature Engineering
↓
Feature Scaling
↓
Train-Test Split
↓
CatBoost Regression
↓
Model Evaluation
↓
Model Persistence
↓
Streamlit Deployment

---

# Evaluation Metrics

The model is evaluated using:

### Mean Absolute Error (MAE)

MAE = (1/n) Σ |yi - ŷi|

### Root Mean Squared Error (RMSE)

RMSE = √[(1/n) Σ(yi - ŷi)²]

### R² Score

R² = 1 - (SSres / SStot)

---

# Technologies Used

## Programming Language

* Python

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-Learn
* CatBoost

## Visualization

* Plotly

## Deployment

* Streamlit

## Model Persistence

* Joblib

---

# Installation

Clone Repository

git clone 

Move to Project Directory

cd NYC-Housing-Intelligence-Platform

Install Dependencies

pip install -r requirements.txt

---

# Requirements

streamlit
pandas
numpy
plotly
scikit-learn
catboost
joblib

---

# Train Model

python train_model.py

---

# Launch Application

streamlit run app.py

---

# Project Structure

NYC-Housing-Intelligence-Platform/

├── app.py

├── train_model.py

├── nyc_housing_base.csv

├── nyc_housing_ai.pkl

├── requirements.txt

├── README.md

└── assets/

---

# Business Impact

This platform can assist:

* Real Estate Agencies
* Property Investors
* Housing Market Analysts
* Real Estate Consultants
* Financial Institutions
* Mortgage Service Providers

by providing rapid and data-driven property valuation insights.

---

# Future Enhancements

* XGBoost Integration
* LightGBM Integration
* Ensemble Learning
* SHAP Explainability
* Property Investment Score
* Risk Assessment Engine
* Automated PDF Reports
* Cloud Deployment
* User Authentication
* AI-Powered Property Recommendations

---

# Author

M Viswa Teja reddy

Machine Learning | Data Science | Artificial Intelligence | Analytics Engineering

---

⭐ If you found this project useful, consider giving it a star and sharing it with the community.

