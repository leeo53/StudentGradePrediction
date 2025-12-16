import numpy as np

# -----------------------------
# Prediction function
# -----------------------------

def predict(X, w, b):
    """
    Computes the linear model prediction.

    Parameters:
    X : (N, d) numpy array
        Input feature matrix with N samples and d features
    w : (d,) numpy array
        Weight vector
    b : float
        Bias (intercept) term

    Returns:
    y_pred : (N,) numpy array
        Predicted values
    """
    # Linear model: y = Xw + b
    y_pred = np.dot(X, w) + b
    return y_pred


# -----------------------------
# Loss functions
# -----------------------------

# Mean Squared Error (MSE)
def lossMSE(y, y_pred):
    """
    Computes Mean Squared Error loss.

    MSE = (1/N) * sum((y_pred - y)^2)
    """
    # Using dot product for efficient squared error computation
    return float((1 / len(y)) * np.dot((y_pred - y).T, (y_pred - y)))


# Mean Absolute Error (MAE)
def lossMAE(y, y_pred):
    """
    Computes Mean Absolute Error loss.

    MAE = (1/N) * sum(|y_pred - y|)
    """
    return (1 / len(y)) * np.sum(np.abs(y_pred - y))


# -----------------------------
# Gradient descent update rules
# -----------------------------

# Gradient descent step for MSE loss
def gradientMSE(X, y, w, b, learning_rate):
    """
    Performs one gradient descent update using MSE loss.
    """
    # Forward pass: predictions
    y_pred = predict(X, w, b)

    # Gradients derived from MSE loss
    # dL/dw = (2/N) * X^T (y_pred - y)
    grad_w = (2 / len(y)) * np.dot(X.T, (y_pred - y))

    # dL/db = (2/N) * sum(y_pred - y)
    grad_b = (2 / len(y)) * np.sum(y_pred - y)

    # Parameter update
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    return w, b


# Gradient descent step for MAE loss
def gradientMAE(X, y, w, b, learning_rate):
    """
    Performs one gradient descent update using MAE loss.

    Note:
    MAE is not differentiable at 0, so we use the subgradient
    given by sign(y_pred - y).
    """
    # Forward pass
    y_pred = predict(X, w, b)

    # Subgradient of MAE
    grad_w = (1 / len(y)) * np.dot(X.T, np.sign(y_pred - y))
    grad_b = (1 / len(y)) * np.sum(np.sign(y_pred - y))

    # Parameter update
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    return w, b


# -----------------------------
# Closed-form solution (Normal Equation)
# -----------------------------

def normalequationMSE(X, y):
    """
    Solves linear regression using the normal equation:

    w_aug = (X_aug^T X_aug)^(-1) X_aug^T y

    where X_aug = [X | 1] includes the bias term.
    """
    N = X.shape[0]

    # Augment X with a column of ones to absorb the bias term
    X_aug = np.hstack((X, np.ones((N, 1))))

    # Compute weights using the pseudoinverse (numerically stable)
    w_aug = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y

    # Separate weights and bias
    w = w_aug[:-1]
    b = w_aug[-1]

    return w, b


# -----------------------------
# Training dispatcher
# -----------------------------

def fit(X, y, X_val, y_val, learning_rate, epochs, showEpochs, method):
    """
    Selects and runs the desired training method.

    method:
        "MSE" -> Gradient descent with MSE loss
        "MAE" -> Gradient descent with MAE loss
        "NE"  -> Normal equation (closed-form)
    """
    if method == "MSE":
        return fit_gradientMSE(X, y, X_val, y_val, learning_rate, epochs, showEpochs)
    elif method == "MAE":
        return fit_gradientMAE(X, y, X_val, y_val, learning_rate, epochs, showEpochs)
    elif method == "NE":
        return normalequationMSE(X, y)


# -----------------------------
# Training loops
# -----------------------------

# Gradient descent training with MSE
def fit_gradientMSE(X, y, X_val, y_val, learning_rate, epochs, showEpochs):
    # Small random initialization of weights
    w = np.random.randn(X.shape[1]) * 0.01
    b = 0

    # Track training and validation losses
    t_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training loss
        y_pred = predict(X, w, b)
        loss = lossMSE(y, y_pred)
        t_losses.append(loss)

        # Validation loss
        y_val_pred = predict(X_val, w, b)
        val_loss = lossMSE(y_val, y_val_pred)
        val_losses.append(val_loss)

        # Optional logging
        if showEpochs and epoch % 10 == 0:
            print(f"learning_rate: {learning_rate}, Epoch {epoch}: Loss: {loss:.3f}")

        # Gradient descent update
        w, b = gradientMSE(X, y, w, b, learning_rate)

    return w, b, t_losses, val_losses


# Gradient descent training with MAE
def fit_gradientMAE(X, y, X_val, y_val, learning_rate, epochs, showEpochs):
    # Small random initialization of weights
    w = np.random.randn(X.shape[1]) * 0.01
    b = 0

    # Track training and validation losses
    t_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training loss
        y_pred = predict(X, w, b)
        loss = lossMAE(y, y_pred)
        t_losses.append(loss)

        # Validation loss
        y_val_pred = predict(X_val, w, b)
        val_loss = lossMAE(y_val, y_val_pred)
        val_losses.append(val_loss)

        # Optional logging
        if showEpochs and epoch % 10 == 0:
            print(f"learning_rate: {learning_rate}, Epoch {epoch}: Loss: {loss:.3f}")

        # Gradient descent update
        w, b = gradientMAE(X, y, w, b, learning_rate)

    return w, b, t_losses, val_losses
