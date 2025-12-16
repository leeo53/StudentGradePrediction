# Student Grades Linear Regression

## Overview

This project predicts student academic performance using **linear regression models implemented entirely from scratch** (no sklearn regressors).

The goal is both **predictive** and **educational**: to understand how linear regression, loss functions, and gradient descent behave under different data representations.

The project demonstrates:

* Linear regression implemented from first principles
* Training with **Gradient Descent (MSE)** and **Gradient Descent (MAE)**
* Closed-form solution using the **Normal Equation**
* Careful data preprocessing and categorical encoding
* Proper **train / validation / test** evaluation
* Baseline comparison to verify real learning

---

## Datasets

### 1. UCI Student Performance Dataset

* **Source:** UCI Machine Learning Repository
* **Target:** Final grade (0–20 scale)
* **Features:** prior grades, study time, attendance, family and social factors

This dataset is used to compare different optimization methods (MSE vs MAE vs Normal Equation).

---

### 2. Mendeley Student Performance Metrics

* **Source:** Mendeley Data
* **Target:** Overall GPA (≈ 0–4 scale)
* **Features:**

  * Academic history (HSC, SSC)
  * Attendance (ordinal)
  * Preparation time
  * Gaming hours
  * English proficiency
  * Semester progression
  * Department (one-hot encoded)

This dataset focuses on **feature engineering**, ordinal encoding, and robustness to noisy categorical data.

---

## Model Implementation

All models are implemented manually using NumPy:

* **Prediction:**
  [ \hat{y} = Xw + b ]

* **Loss Functions:**

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)

* **Optimization:**

  * Batch Gradient Descent (MSE)
  * Subgradient Descent (MAE)
  * Normal Equation (MSE)

No automatic differentiation or sklearn regressors are used.

---

## Data Preprocessing

Key preprocessing steps include:

* Cleaning string columns (whitespace, invisible characters, casing)
* Encoding categorical features:

  * Binary encoding for yes/no features
  * Ordinal encoding for ordered categories (attendance, income, semester)
  * One-hot encoding for nominal features (department)
* Ensuring all inputs are numeric and free of missing values

Special care is taken to avoid data leakage and category mismatches.

---

## Train / Validation / Test Split

The data is split as follows:

* **60% Training**
* **20% Validation** (hyperparameter tuning)
* **20% Test** (final evaluation)

Validation data is used **only** for model selection.

---

## Results

### UCI Dataset (with and without school feature)

Typical performance:

* MAE ≈ 2.1
* RMSE ≈ 2.9

Gradient Descent (MAE) performs slightly better than MSE and the Normal Equation.

---

### Mendeley Dataset

Best model: **Gradient Descent trained with MAE**

| Metric | Value |
| ------ | ----- |
| MAE    | 0.376 |
| MSE    | 0.246 |
| RMSE   | 0.496 |

**Baseline MAE (mean predictor):** 0.501
**Improvement:** ~25% reduction in error

An MAE of 0.376 corresponds to an average prediction error of **less than half a GPA point**.

---

## Baseline Comparison

A naive baseline predicting the training mean is used to verify learning:

```python
baseline_mae = abs(y_test - y_train.mean()).mean()
```

The trained model consistently outperforms this baseline, confirming meaningful feature learning.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/leeo53/StudentGradePrediction.git
cd StudentGradePrediction
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

---

## Usage

Run experiments from the `src/` directory. Example:

```bash
python mendeley_run.py
```

This will:

1. Load and preprocess the dataset
2. Tune hyperparameters using the validation set
3. Train the final model
4. Report MAE, MSE, and RMSE on the test set
5. Output 4 training loss vs validation loss over epochs graphs

---

## Model Behavior & Generalization

An important observation from this project concerns **generalization and dataset bias**.

When training on the **UCI Student Performance dataset**, most target grades cluster around a narrow range (roughly the middle of the scale). As a result, the learned linear model tends to predict values close to this average. While this yields reasonable aggregate metrics (MAE/MSE), it limits the model’s usefulness for **out-of-distribution inputs**.

To explore this, the learned weights from the UCI-trained model were applied to manually constructed feature vectors representing real student data. In these cases, predictions were often far from actual outcomes, indicating that the model was relying heavily on the dataset’s grade distribution rather than learning broadly transferable relationships.

This motivated the inclusion of the **Mendeley dataset**, which:

* Uses a GPA-scale target variable
* Has more evenly distributed outcomes
* Includes stronger academic and behavioral predictors

Future work will involve testing real-world inputs against models trained on the Mendeley dataset to evaluate whether improved feature quality leads to better personalization and generalization.

---

## Key Takeaways

* Dataset choice strongly influences model behavior
* Models trained on narrowly distributed targets may appear accurate but generalize poorly
* Feature quality matters more than model complexity
* Building models from scratch exposes these limitations clearly

---

## Future Work

* Add L2 regularization (ridge regression)
* Compare against sklearn's LinearRegression
* Feature importance visualization
* Residual and prediction plots

---

## Author

**Liam L.**
BS Computer Science
