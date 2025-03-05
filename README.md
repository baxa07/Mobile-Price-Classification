# Mobile-Price-Classification
This repository contains the code for a mobile price classification project. The objective is to predict the price range of mobile phones using various features such as battery power, RAM, screen dimensions, and more. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, and model training with hyperparameter tuning.

Project Overview

The project pipeline includes the following key steps:

## Data Preprocessing:
Cleaning and preparing the dataset (train.csv) for analysis and modeling.
## Exploratory Data Analysis (EDA):
Analyzing the dataset to understand its distribution, summary statistics, and feature relationships. The EDA is performed in the EDA.py script.
## Feature Engineering:
Identifying important features using techniques such as feature importance ranking and Principal Component Analysis (PCA). The most significant features selected include ram, battery_power, px_height, and px_width. This is implemented in feature_engineering.py.
## Model Training and Hyperparameter Tuning:
Training a classification model using a combination of clustering (GMM) and decision tree-based methods, along with an extensive grid search for hyperparameter tuning. The best model parameters found were:
max_depth: 15
max_features: log2
min_samples_leaf: 2
min_samples_split: 5
The training and testing accuracies achieved are approximately 99.25% and 91.50%, respectively. This step is handled in model_training.py.
Pipeline Integration:
The main.py script ties all the steps together, providing a smooth workflow from data preprocessing to final evaluation.
File Structure

#### main.py:
Main script to run the entire pipeline.
#### preprocessing.py:
Script for data cleaning and preprocessing.
#### EDA.py:
Contains code for performing exploratory data analysis and generating plots.
#### feature_engineering.py:
Script for feature selection, importance calculation, and dimensionality reduction using PCA.
#### model_training.py:
Code for training the classification model, including hyperparameter tuning with GridSearchCV.
#### train.csv:
The dataset file containing mobile phone specifications and their corresponding price ranges.

# Installation

Clone the repository:
git clone https://github.com/baxa07/Mobile-Price-Classification

cd mobile-price-classification

Results

Training Accuracy (GMM): 99.25%
Test Accuracy (GMM): 91.50%
A detailed classification report on the test data is generated, showing precision, recall, and F1-scores for each price range category.

Methodology

Data Preprocessing:
Basic cleaning and preparation of the dataset.
Exploratory Data Analysis:
Visualizations and statistical summaries to understand the distribution of features and target variable.
Feature Engineering:
Ranking feature importance.
Reducing dimensionality with PCA.
Selecting top features based on importance (i.e., ram, battery_power, px_height, px_width).
Model Training:
Training a model using Gaussian Mixture Model (GMM) clustering combined with a classification approach.
Using GridSearchCV for hyperparameter tuning.
Evaluation:
Assessing model performance using training and test accuracies along with a detailed classification report.



