import regression_model as model

# -------------------------------------------------
# Hyperparameter search / model selection helper
# -------------------------------------------------

def run(trainfunction, X, y, X_val, y_val,
        learning_rates, epochs_list, showEpochs, method):
    """
    Performs a simple grid search over learning rates and epochs
    to find the model with the lowest validation MAE.

    Parameters:
    trainfunction : callable
        Training function (e.g. fit from regression_model)
    X, y : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    learning_rates : iterable
        List of learning rates to try
    epochs_list : iterable
        List of epoch counts to try
    showEpochs : bool
        Whether to print training progress
    method : str
        Training method ("MSE", "MAE", or "NE")

    Returns:
    best_params : dict
        Dictionary containing the best hyperparameters,
        trained weights, and loss histories
    """

    # Track the best (lowest) validation MAE seen so far
    best_mae = float('inf')

    # Store parameters and results of the best model
    best_params = {}

    # Grid search over learning rates and epoch counts
    for lr in learning_rates:
        for ep in epochs_list:
            # Train model with current hyperparameters
            w, b, t_losses, val_losses = trainfunction(
                X, y,
                X_val, y_val,
                lr,
                epochs=ep,
                showEpochs=showEpochs,
                method=method
            )

            # Predict on validation set using trained model
            y_val_pred = model.predict(X_val, w, b)

            # Compute validation MAE
            # Note: MAE is symmetric, so argument order does not matter
            val_mae = model.lossMAE(y_val_pred, y_val)

            # Update best model if validation MAE improves
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = {
                    'learning_rate': lr,
                    'epochs': ep,
                    'w': w,
                    'b': b,
                    't_losses': t_losses,
                    'val_losses': val_losses
                }

    return best_params
