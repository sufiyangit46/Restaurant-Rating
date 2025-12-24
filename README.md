🍽️ Restaurant Rating Prediction – End-to-End Machine Learning Project
🔍 Problem Statement

Online food platforms and restaurant aggregators rely heavily on ratings to influence customer decisions.
However, new or less-reviewed restaurants often suffer from rating cold-start problems.

This project predicts a restaurant’s aggregate rating based on location, cuisine, cost, services, and operational attributes—helping platforms make data-driven recommendations.

🎯 Business Objective

Predict restaurant ratings before sufficient user feedback exists

Help food platforms improve ranking and discovery

Enable restaurant owners to understand factors impacting ratings

Reduce bias caused by low review volume

🧠 Solution Overview

An end-to-end machine learning pipeline was developed to transform raw restaurant data into accurate rating predictions.
The project follows production-grade ML engineering practices, including modular design, artifact management, and deployment readiness.

⚙️ Tech Stack

Language: Python

Data Handling: Pandas, NumPy

ML Framework: Scikit-learn

Feature Engineering: ColumnTransformer, Pipelines

Model Deployment: Flask

Version Control: Git & GitHub

🔄 Machine Learning Workflow

Data Ingestion – Load and split raw dataset

Data Transformation

Missing value imputation

One-Hot Encoding for low-cardinality features

Ordinal Encoding for high-cardinality features

Feature scaling

Model Training – Train multiple regression models

Model Evaluation – Compare models using R² Score

Artifact Creation – Save trained model & preprocessor

Prediction Pipeline – Serve predictions via Flask app

🤖 Models Used

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Best model selected based on cross-validated performance.

📊 Evaluation Metric

R² Score – Measures predictive power of the model

🌐 Web Application

Interactive UI for rating prediction

Takes restaurant attributes as input

Outputs predicted aggregate rating instantly

💼 Real-World Impact

Helps food platforms rank restaurants fairly

Supports restaurant owners in decision-making

Improves customer experience

Reduces cold-start problem in recommendation systems

🧑‍💻 Skills Demonstrated

End-to-end ML pipeline development

Feature engineering at scale

Model selection & evaluation

Clean code & modular architecture

Model serialization & deployment

Production-ready ML systems
