
---

# Heart Disease Prediction Model

## Overview

This project involves creating a predictive model to determine the presence of heart disease in patients based on various medical features. The model is developed using Logistic Regression and trained on a dataset containing information about patients' health indicators.

## Table of Contents

1. [Project Description](#project-description)
2. [Dependencies](#dependencies)
3. [Data Collection and Processing](#data-collection-and-processing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [License](#license)

## Project Description

The goal of this project is to predict the likelihood of heart disease in patients using a logistic regression model. The dataset used contains features such as age, sex, blood pressure, cholesterol levels, and more, which are used to classify patients as either having heart disease or being healthy.

## Dependencies

To run this project, you will need to have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Data Collection and Processing

The dataset used for this project is stored in `heart.csv`. It includes the following columns:

- `age`: Age of the patient
- `sex`: Gender of the patient (1 = male, 0 = female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol level (in mg/dl)
- `fbs`: Fasting blood sugar (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: Depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- `target`: Heart disease (1 = disease, 0 = no disease)

The data is processed by handling missing values, and splitting into features (`X`) and target (`Y`).

## Model Training

The model is trained using Logistic Regression with the following code:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

## Evaluation

The model is evaluated using accuracy score, confusion matrix, and classification report. These metrics help in understanding the performance of the model and its effectiveness in predicting heart disease.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd heart-disease-prediction
    ```

3. Run the script:
    ```bash
    python heart_disease_prediction.py
    ```

Ensure that the `heart.csv` file is in the same directory as the script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
