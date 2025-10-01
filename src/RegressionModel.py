import numpy as np


def predict(X,w,b):
    y_pred = np.dot(X,w) + b
    return y_pred

def lossMSE(y,y_pred):
    return float((1/len(y))*np.dot((y_pred-y).T,(y_pred-y)))

def lossMAE(y,y_pred):
    return (1/len(y))*np.sum(np.abs(y_pred-y))

def gradientMSE(X,y,w,b,learning_rate):
    y_pred = predict(X,w,b)
    grad_w = (2/len(y))*np.dot(X.T,(y_pred - y))
    grad_b = (2/len(y))*np.sum(y_pred - y)
    w = w - learning_rate*grad_w
    b = b - learning_rate*grad_b
    return w,b

def gradientMAE(X,y,w,b,learning_rate):
    y_pred = predict(X,w,b)
    grad_w = (1/len(y))*np.dot(X.T,np.sign(y_pred - y))
    grad_b = (1/len(y))*np.sum(np.sign(y_pred - y))
    w = w - learning_rate*grad_w
    b = b - learning_rate*grad_b
    return w,b

#using pseudoinverse to handle dummy columns better
def normalequationMSE(X,y):
    N = X.shape[0]
    X_aug = np.hstack((X,np.ones((N,1))))
    w_aug = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
    w=w_aug[:-1]
    b=w_aug[-1]
    return w,b

