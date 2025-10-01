from ucimlrepo import fetch_ucirepo
import numpy as np
import RegressionModel as rm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets.iloc[:, 0]

# data to numeric
X.loc[:,'sex'] = X['sex'].map({'F':0, 'M':1})
X.loc[:,'Pstatus'] = X['Pstatus'].map({'T':0, 'A':1})
X.loc[:,'address'] = X['address'].map({'U':0, 'R':1})
X.loc[:,'famsize'] = X['famsize'].map({'LE3':0, 'GT3':1})
X.loc[:,'school'] = X['school'].map({'GP':0, 'MS':1})
X.loc[:,'schoolsup'] = X['schoolsup'].map({'yes':0, 'no':1})
X.loc[:,'famsup'] = X['famsup'].map({'yes':0, 'no':1})
X.loc[:,'paid'] = X['paid'].map({'yes':0, 'no':1})
X.loc[:,'activities'] = X['activities'].map({'yes':0, 'no':1})
X.loc[:,'nursery'] = X['nursery'].map({'yes':0, 'no':1})
X.loc[:,'higher'] = X['higher'].map({'yes':0, 'no':1})
X.loc[:,'internet'] = X['internet'].map({'yes':0, 'no':1})
X.loc[:,'romantic'] = X['romantic'].map({'yes':0, 'no':1})
# add colums for the different options (one-hot encoding)
X = pd.get_dummies(X, columns=['Mjob','Fjob','reason','guardian'])
X = np.array(X, dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
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

def trainMSE(X,y):
    w=np.random.randn(X.shape[1]) * 0.01
    b=0
    epochs = 100
    losses = []
    for epoch in range(epochs):
        y_pred = rm.predict(X,w,b)
        loss = rm.lossMSE(y,y_pred)
        losses.append(loss)
        w,b = rm.gradientMSE(X,y,w,b,.001)
        print("Epoch:",epoch,"Loss:",loss)
    return w,b,losses

def trainMAE(X,y):
    w=np.random.randn(X.shape[1]) * 0.01
    b=0
    epochs = 100
    losses = []
    for epoch in range(epochs):
        y_pred = rm.predict(X,w,b)
        loss = rm.lossMAE(y,y_pred)
        losses.append(loss)
        w,b = rm.gradientMAE(X,y,w,b,.001)
        print("Epoch:",epoch,"Loss:",loss)
    return w,b,losses
def testMSE(X,y,w,b):
    y_pred = rm.predict(X,w,b)
    mse = rm.lossMSE(y,y_pred)
    rmse = np.sqrt(mse)
    return mse,rmse
def testMAE(X,y,w,b):
    y_pred = rm.predict(X,w,b)
    mae = rm.lossMAE(y,y_pred)
    return mae

#w,b,losses = trainMSE(X_train,y_train)
#mse,rmse = testMSE(X_test,y_test,w,b)
#print("Average Loss for linear regression with gradient descent and MSE:",np.mean(losses))
#print("Mean Squared Error on testing data:",mse,"\nand Root Mean Squared error on testing data:",rmse)
#w, b, losses = trainMAE(X_train,y_train)
#mae = testMAE(X_train,y_train,w,b)
#print("Average Loss for linear regression with gradient descent and MAE:", np.mean(losses))
#print("Mean Absolute Error on testing data:",mae)

w, b = rm.normalequationMSE(X_train,y_train)
mse,rmse = testMSE(X_test,y_test,w,b)
print("Mean Squared Error on testing data:",mse,"\nand Root Mean Squared error on testing data:",rmse)