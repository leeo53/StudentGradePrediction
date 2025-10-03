from ucimlrepo import fetch_ucirepo
import numpy as np
import RegressionModel as model
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Student Grade Prediction")
parser.add_argument("--show-epochs", action="store_true",
                    help="Print loss at each epoch during training")
args = parser.parse_args()

learning_rates = [0.0001,0.001,0.01]
epochs_list = [50,100,200,400]

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets.iloc[:, 2]


# data to numeric
yes_no = {'yes':0, 'no':1}
X.loc[:,'sex'] = X['sex'].map({'F':0, 'M':1})
X.loc[:,'Pstatus'] = X['Pstatus'].map({'T':0, 'A':1})
X.loc[:,'address'] = X['address'].map({'U':0, 'R':1})
X.loc[:,'famsize'] = X['famsize'].map({'LE3':0, 'GT3':1})
X.loc[:,'school'] = X['school'].map({'GP':0, 'MS':1})
X.loc[:,'schoolsup'] = X['schoolsup'].map(yes_no)
X.loc[:,'famsup'] = X['famsup'].map(yes_no)
X.loc[:,'paid'] = X['paid'].map(yes_no)
X.loc[:,'activities'] = X['activities'].map(yes_no)
X.loc[:,'nursery'] = X['nursery'].map(yes_no)
X.loc[:,'higher'] = X['higher'].map(yes_no)
X.loc[:,'internet'] = X['internet'].map(yes_no)
X.loc[:,'romantic'] = X['romantic'].map(yes_no)
# add colums for the different options (one-hot encoding)
X = pd.get_dummies(X, columns=['Mjob','Fjob','reason','guardian'])
X_no_school = X.drop(columns="school")

#for predict.py
train_columns = list(X_no_school.columns)

X_no_school = np.array(X_no_school,dtype=float)
X = np.array(X, dtype=float)
y = y.to_numpy(dtype=float)

#split up data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# metadata
# print(student_performance.metadata)

# variable information
# print(student_performance.variables)

def trainMSE(X,y,X_val,y_val,lr,epochs,showEpochs):
    w=np.random.randn(X.shape[1]) * 0.01
    b=0
    t_losses = []
    val_losses = []
    for epoch in range(epochs):
        y_pred = model.predict(X, w, b)
        loss = model.lossMSE(y, y_pred)
        t_losses.append(loss)

        y_val_pred = model.predict(X_val, w, b)
        val_loss = model.lossMSE(y_val, y_val_pred)
        val_losses.append(val_loss)
        if showEpochs and epoch % 10 == 0:
                print(f"lr: {lr},Epoch {epoch}: Loss: {loss:.3f}")
        w, b = model.gradientMSE(X, y, w, b, lr)
    return w,b,t_losses,val_losses

def trainMAE(X,y,X_val,y_val,lr,epochs,showEpochs):
    w=np.random.randn(X.shape[1]) * 0.01
    b=0
    t_losses = []
    val_losses = []
    for epoch in range(epochs):
        y_pred = model.predict(X, w, b)
        loss = model.lossMAE(y, y_pred)
        t_losses.append(loss)

        y_val_pred = model.predict(X_val, w, b)
        val_loss = model.lossMAE(y_val, y_val_pred)
        val_losses.append(val_loss)
        if showEpochs and epoch % 10 == 0:
            print(f"lr: {lr}, Epoch {epoch}: Loss: {loss:.3f}")
        w, b = model.gradientMAE(X, y, w, b, lr)
    return w,b,t_losses, val_losses

def hyperparameter_tuning(trainfunction,X,y,X_val,y_val, learning_rates, epochs_list,showEpochs):
    best_mae = float('inf')
    best_params = {}
    for lr in learning_rates:
        for ep in epochs_list:
            w, b, t_losses, val_losses = trainfunction(X, y, X_val, y_val, lr, epochs=ep, showEpochs=showEpochs)
            y_val_pred = model.predict(X_val, w, b)
            val_mae = model.lossMAE(y_val_pred, y_val)
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = {'learning_rate': lr, 'epochs': ep, 'w': w, 'b': b,
                               't_losses': t_losses, 'val_losses': val_losses}
    return best_params

def testMSE(X,y,w,b):
    y_pred = model.predict(X, w, b)
    mse = model.lossMSE(y, y_pred)
    rmse = np.sqrt(mse)
    return mse,rmse

def testMAE(X,y,w,b):
    y_pred = model.predict(X, w, b)
    mae = model.lossMAE(y, y_pred)
    return mae

# testing and output
best_paramsGD1= hyperparameter_tuning(trainMSE,X_train,y_train,X_val,y_val,learning_rates,
                                       epochs_list,showEpochs=False)
w_GD1 = best_paramsGD1['w']
b_GD1 = best_paramsGD1['b']
GD1_mse,GD1_rmse = testMSE(X_test, y_test, w_GD1, b_GD1)
GD1_mae = testMAE(X_test, y_test, w_GD1, b_GD1)

best_paramsGD2= hyperparameter_tuning(trainMAE,X_train,y_train,X_val,y_val,learning_rates,
                                       epochs_list,showEpochs=False)
w_GD2 = best_paramsGD2['w']
b_GD2 = best_paramsGD2['b']
GD2_mse,GD2_rmse = testMSE(X_test,y_test,w_GD2,b_GD2)
GD2_mae = testMAE(X_test,y_test,w_GD2,b_GD2)

w_NE, b_NE = model.normalequationMSE(X_train, y_train)
NE_mse,NE_rmse = testMSE(X_test,y_test,w_NE,b_NE)
NE_mae = testMAE(X_test,y_test,w_NE,b_NE)

print("\n" + "="*40)
print(" Student Grade Prediction")
print("="*40)

print("\n"+"Dataset: UCI Student Performance")
print("Train/Validation/Test Split: 60% / 20% / 20%")

print("\n"+"="*40)
print("WITH 'school' feature")
print("="*40)

print(f"{'Method':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
print("-"*60)
print(f"{'Gradient Descent (MSE)':<25} {GD1_mae:<10.3f} {GD1_mse:<10.3f} {GD1_rmse:<10.3f}")
print(f"{'Gradient Descent (MAE)':<25} {GD2_mae:<10.3f} {GD2_mse:<10.3f} {GD2_rmse:<10.3f}")
print(f"{'Normal Equation':<25} {NE_mae:<10.3f} {NE_mse:<10.3f} {NE_rmse:<10.3f}")
print("-"*60)
print(f"{'Hyperparameters':<25} {'LR':<10} {'Epochs':<10}")
print("-"*60)
print(f"{'Gradient Descent (MSE)':<25} {best_paramsGD1['learning_rate']:<10} {best_paramsGD1['epochs']:<10}")
print(f"{'Gradient Descent (MAE)':<25} {best_paramsGD2['learning_rate']:<10} {best_paramsGD1['epochs']:<10}")

#loss plot
plt.figure(figsize=(8,5))
plt.plot(best_paramsGD1['t_losses'], label="Training Loss")
plt.plot(best_paramsGD1['val_losses'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("GD1MSE_school.png",dpi=300)
plt.figure(figsize=(8,5))
plt.plot(best_paramsGD2['t_losses'], label="Training Loss")
plt.plot(best_paramsGD2['val_losses'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MAE Loss")
plt.title("Training vs Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("GD2MAE_school.png",dpi=300)

#--------------------------------------
# 'school' feature removed test
#--------------------------------------

#split up data for without school column
X_train, X_test, y_train, y_test = train_test_split(
    X_no_school, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

best_paramsGD1= hyperparameter_tuning(trainMSE,X_train,y_train,X_val,y_val,learning_rates,
                                       epochs_list,showEpochs=False)
w_GD1 = best_paramsGD1['w']
b_GD1 = best_paramsGD1['b']
GD1_mse,GD1_rmse = testMSE(X_test, y_test, w_GD1, b_GD1)
GD1_mae = testMAE(X_test, y_test, w_GD1, b_GD1)

best_paramsGD2= hyperparameter_tuning(trainMAE,X_train,y_train,X_val,y_val,learning_rates,
                                       epochs_list,showEpochs=False)
w_GD2 = best_paramsGD2['w']
b_GD2 = best_paramsGD2['b']
GD2_mse,GD2_rmse = testMSE(X_test,y_test,w_GD2,b_GD2)
GD2_mae = testMAE(X_test,y_test,w_GD2,b_GD2)

w_NE, b_NE = model.normalequationMSE(X_train, y_train)
NE_mse,NE_rmse = testMSE(X_test,y_test,w_NE,b_NE)
NE_mae = testMAE(X_test,y_test,w_NE,b_NE)

#for predict.py
np.savez("weights.npz", w=w_GD2, b=b_GD2, train_columns=np.array(train_columns, dtype=object))

print("\n"+"="*40)
print("WITHOUT 'school' feature")
print("="*40)

print(f"{'Method':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
print("-"*60)
print(f"{'Gradient Descent (MSE)':<25} {GD1_mae:<10.3f} {GD1_mse:<10.3f} {GD1_rmse:<10.3f}")
print(f"{'Gradient Descent (MAE)':<25} {GD2_mae:<10.3f} {GD2_mse:<10.3f} {GD2_rmse:<10.3f}")
print(f"{'Normal Equation':<25} {NE_mae:<10.3f} {NE_mse:<10.3f} {NE_rmse:<10.3f}")
print("-"*60)
print(f"{'Hyperparameters':<25} {'LR':<10} {'Epochs':<10}")
print("-"*60)
print(f"{'Gradient Descent (MSE)':<25} {best_paramsGD1['learning_rate']:<10} {best_paramsGD1['epochs']:<10}")
print(f"{'Gradient Descent (MAE)':<25} {best_paramsGD2['learning_rate']:<10} {best_paramsGD1['epochs']:<10}")

#loss plot
plt.figure(figsize=(8,5))
plt.plot(best_paramsGD1['t_losses'], label="Training Loss")
plt.plot(best_paramsGD1['val_losses'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("GD1MSE_noschool.png",dpi=300)
plt.figure(figsize=(8,5))
plt.plot(best_paramsGD2['t_losses'], label="Training Loss")
plt.plot(best_paramsGD2['val_losses'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MAE Loss")
plt.title("Training vs Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("GD2MAE_noschool.png",dpi=300)