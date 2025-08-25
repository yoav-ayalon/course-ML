# course-ML

This repository contains a machine learning project for a university course, focused on predicting car insurance claims using a real-world dataset.

## Project Overview

The main script, [part_B_group_7.py](part_B_group_7.py), demonstrates the full workflow of a supervised classification task, including:

- **Data Preprocessing:** Handling missing values, feature engineering, and categorical encoding for variables such as driving experience, credit score, annual mileage, and more.
- **Exploratory Analysis & Visualization:** Includes 3D and 2D plots to analyze decision tree parameters and visualize clustering results.
- **Modeling:** Implements and compares several machine learning models:
  - Decision Tree Classifier (with hyperparameter tuning and visualization)
  - Neural Network (MLPClassifier, with grid search for optimal parameters)
  - XGBoost Classifier (with manual and grid search hyperparameter tuning)
  - KMeans clustering and PCA for unsupervised analysis
- **Evaluation:** Uses metrics like F1 score, accuracy, and confusion matrices to assess model performance.
- **Prediction:** Generates final predictions for test data after preprocessing and model selection.

## Structure

- [part_B_group_7.py](part_B_group_7.py): Main code for data processing, modeling, and evaluation.
- `data/`: Contains training and test datasets.
- `grid search results & final predictions/`: Stores results from hyperparameter tuning and final model predictions.
- `part B group 7.pdf`: Project report.

This project showcases practical machine learning techniques for tabular data, including feature engineering, model selection, and performance analysis.