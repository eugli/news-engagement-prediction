def mse(outputs, labels):
    return ((outputs - labels) ** 2).mean()