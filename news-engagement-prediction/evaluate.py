def mse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())