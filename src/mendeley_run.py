import numpy as np
import regression_model as model
import tune as tune
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load data
# -------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mendeley.csv")
datafile = pd.read_csv(data_path)

# -------------------------------------------------
# Clean string columns (remove spaces, invisibles, lowercase)
# -------------------------------------------------
datafile.columns = datafile.columns.str.strip().str.lower()
def clean_strings(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'[\x00-\x1F\x7F\u00A0]', '', regex=True)
        df[col] = df[col].str.lower()
    return df

datafile = clean_strings(datafile)

# -------------------------------------------------
# Separate features and target
# -------------------------------------------------
X = datafile.drop(columns=["last", "overall"])
y = datafile["overall"].values

# -------------------------------------------------
# One-hot encode Department (nominal)
# -------------------------------------------------
X = pd.get_dummies(X, columns=["department"], drop_first=True)

# -------------------------------------------------
# Encode binary and ordinal features
# -------------------------------------------------
X["gender"] = X["gender"].map({"female": 0, "male": 1})

X["income"] = X["income"].map({
    "low (below 15,000)": 0,
    "lower middle (15,000-30,000)": 1,
    "upper middle (30,000-50,000)": 2,
    "high (above 50,000)": 3
})

X["hometown"] = X["hometown"].map({
    "village": 0,
    "city": 1
})

X["preparation"] = X["preparation"].map({
    "0-1 hour": 0,
    "2-3 hours": 1,
    "more than 3 hours": 2
})

X["gaming"] = X["gaming"].map({
    "0-1 hour": 0,
    "2-3 hours": 1,
    "more than 3 hours": 2
})

X["attendance"] = X["attendance"].map({
    "below 40%": 0,
    "40%-59%": 1,
    "60%-79%": 2,
    "80%-100%": 3
})

X["job"] = X["job"].map({"no": 0, "yes": 1})
X["extra"] = X["extra"].map({"no": 0, "yes": 1})

X["semester"] = X["semester"].map({
    "2nd": 0,
    "3rd": 1,
    "4th": 2,
    "5th": 3,
    "6th": 4,
    "7th": 5,
    "8th": 6,
    "9th": 7,
    "10th": 8,
    "11th": 9,
    "12th": 10
})
X = X.values.astype(float)

learning_rates = [0.0001,0.001,0.01]
epochs_list = [50,100,200,400]

#split up data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

def testMSE(X,y,w,b):
    y_pred = model.predict(X, w, b)
    mse = model.lossMSE(y, y_pred)
    rmse = np.sqrt(mse)
    return mse,rmse

def testMAE(X,y,w,b):
    y_pred = model.predict(X, w, b)
    mae = model.lossMAE(y, y_pred)
    return mae

best_params = tune.run(model.fit,X_train,y_train,X_val,y_val,learning_rates=learning_rates, epochs_list=epochs_list,
                       showEpochs=True, method="MAE")

w = best_params['w']
b = best_params['b']
mae = testMAE(X_test,y_test,w,b)
mse,rmse = testMSE(X_test,y_test,w,b)

print("\n" + "="*40)
print(" Student Grade Prediction")
print("="*40)

print("\n"+"Dataset: Mendeley Student Performance Metrics")
print("Train/Validation/Test Split: 60% / 20% / 20%")

print(f"{'Method':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
print("-"*60)
print(f"{'Gradient Descent (MAE)':<25} {mae:<10.3f} {mse:<10.3f} {rmse:<10.3f}")
print("-"*60)
print(f"{'Hyperparameters':<25} {'LR':<10} {'Epochs':<10}")
print("-"*60)
print(f"{'Gradient Descent (MAE)':<25} {best_params['learning_rate']:<10} {best_params['epochs']:<10}")

baseline = abs(y_test - y_train.mean()).mean()
print("Baseline MAE:", baseline)
