# Breast Cancer Classification — Logistic Regression

End-to-end Machine Learning pipeline for breast cancer tumor classification using Logistic Regression and Scikit-Learn.

## Project Overview

This project implements a complete machine learning workflow to classify breast tumors as:

- Malignant (1)

- Benign (0)

- The pipeline includes:

- Data preprocessing

- Feature engineering

- Model training

- Model evaluation

- Model serialization

The trained model achieves an AUC-ROC score of 0.8866, demonstrating strong classification performance.

## Problem Statement

Given clinical diagnostic features of breast tumors, build a binary classification model that can accurately predict whether a tumor is malignant or benign.

Reliable classification is critical in medical diagnosis scenarios.

## Tech Stack

- Python 3.x

- NumPy

- Pandas

- Scikit-Learn

- Pickle

## Machine Learning Pipeline
### Data Cleaning

- Replace '?' with NaN

- Mean imputation for numerical features

- Mode imputation for categorical features

### Feature Engineering

- Label Encoding for non-numeric features

- Dropped features (columns 11 and 13)

- Feature scaling using MinMaxScaler

### Train-Test Split

- 80% Training data

- 20% Testing data

- random_state = 42 for reproducibility

## Model Training
```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
```
## Model Evaluation

### Metrics used:

- AUC-ROC

- Accuracy

- Precision

- Recall

- F1 Score

## Model Performance
| Metric |	Score |
| AUC-ROC |	0.8866 |
| Accuracy	| Strong |
| Precision |	Strong |
| Recall |	Strong |
| F1 Score |	Balanced |

An AUC score close to 1 indicates strong separability between malignant and benign classes.

## Project Structure
├── Breast Cancer Logistic Regression Model.ipynb

├── Breast_cancer.pkl

├── README.md

## How to Run
** Install Dependencies **
```
pip install pandas numpy scikit-learn
```
** Run the Notebook **

Open:

- Breast Cancer Logistic Regression Model.ipynb

Run all cells to:

- Clean the data

- Train the model

- Evaluate performance

## Load the Saved Model
```
import pickle

with open("Breast_cancer.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(X_new)
# Example: Compute AUC Score
from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print(auc_score)
```
## Key Learnings

- Proper preprocessing significantly impacts model performance.

- Logistic Regression performs well on structured medical datasets.

- AUC-ROC is more reliable than accuracy for classification tasks.

- Feature scaling improves convergence in gradient-based models.

## Future Improvements

- Cross-validation

- Hyperparameter tuning

- Feature selection

- Compare with other models (Random Forest, SVM, XGBoost)

- Deploy as a REST API

- Add CI/CD pipeline
