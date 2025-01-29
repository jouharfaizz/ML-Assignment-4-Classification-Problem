# ML-Assignment-4-Classification-Problem
Breast Cancer Classification Project

Objective

This project aims to evaluate the understanding and ability to apply supervised learning techniques to a real-world dataset. The task involves using the Breast Cancer dataset from the sklearn library to implement, evaluate, and compare five classification algorithms.

Dataset

The dataset used in this project is the Breast Cancer dataset, available in the sklearn library. It includes data on 569 instances of breast cancer cases, with 30 features that describe the cases and a target variable indicating whether the cancer is malignant or benign.

Components

1. Loading and Preprocessing 

Steps:

Loaded the Breast Cancer dataset.

Checked for missing values (none were found).

Scaled the feature values using StandardScaler to standardize the data.

Split the dataset into training and testing sets (80% train, 20% test).

Purpose:

Preprocessing ensures that all features are on the same scale, which is essential for optimal performance of many machine learning algorithms.

2. Classification Algorithm Implementation 

The following classification algorithms were implemented:

Logistic Regression:

A statistical model that uses a logistic function to model the probability of a binary outcome.

Suitable for this dataset due to its simplicity and effectiveness in binary classification tasks.

Decision Tree Classifier:

A tree-based model that splits the data into subsets based on feature values.

Handles non-linear relationships well.

Random Forest Classifier:

An ensemble of decision trees that improves accuracy by reducing overfitting.

Suitable for this dataset due to its robustness.

Support Vector Machine (SVM):

A model that finds the hyperplane that best separates the classes in the feature space.

Effective for datasets with clear class separations.

k-Nearest Neighbors (k-NN):

A non-parametric method that classifies data points based on the majority class of their nearest neighbors.

Simple and intuitive for classification tasks.

3. Model Comparison 

Evaluation Metrics:

Accuracy score was used to compare the models.

Classification reports and confusion matrices were generated for the best and worst-performing models.

Results:

Models were ranked based on their accuracy.

A bar plot visualized the comparison of accuracies across models.


Results

Best Model: The model with the highest accuracy and detailed classification report.

Worst Model: The model with the lowest accuracy and its classification report.

Files Included

breast_cancer_classification.ipynb: The Jupyter Notebook containing the code implementation.

README.md: This file, providing an overview of the project.

Usage

Clone the repository to your local machine.

Install the necessary Python libraries (numpy, pandas, scikit-learn, matplotlib, seaborn).

Open the Jupyter Notebook file and run the cells sequentially to reproduce the results.

Dependencies

Python 3.x

NumPy

Pandas

scikit-learn

Matplotlib

Seaborn

Author

This project was completed as part of an assessment to demonstrate supervised learning techniques.

Feel free to reach out if you have any questions or need further assistance!
