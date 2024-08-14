
# Heart Disease Prediction using Logistic Regression

## Project Overview

This project aims to predict the presence of heart disease in patients using a logistic regression model. The model is trained on a dataset containing various medical attributes and has been evaluated for accuracy and performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Development](#model-development)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease is one of the leading causes of death globally. Early prediction and diagnosis can save lives and improve the quality of life for many individuals. This project utilizes machine learning techniques, specifically logistic regression, to predict the likelihood of heart disease based on patient data.

## Dataset

The dataset used in this project is the [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from the UCI Machine Learning Repository. It contains 303 records with 14 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, the slope of the peak exercise ST segment, number of major vessels, and thalassemia.

## Preprocessing

Data preprocessing steps include:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Splitting the dataset into training and testing sets

## Model Development

The logistic regression model was developed using Python and the scikit-learn library. Key steps include:

1. **Feature Selection:** Identified the most significant predictors.
2. **Model Training:** Trained the logistic regression model using the training dataset.
3. **Hyperparameter Tuning:** Optimized the model parameters for better performance.

## Evaluation

The model was evaluated using the following metrics:

- Accuracy: 81%
- Precision: 82%
- Recall: 88%
- F1-score: 85%
- ROC-AUC score: 0.92

## Usage

To use the model, follow these steps:

1. Clone the repository:
    git clone https://github.com/deependraxx/heartdiseaseprediction.git
2. Install the required packages:
    pip install -r requirements.txt
3. Run the prediction script:
    python predict.py

## Results

The logistic regression model effectively predicts the presence of heart disease with high accuracy. The ROC-AUC score of 0.92 indicates a strong ability to distinguish between patients with and without heart disease.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.
