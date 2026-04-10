import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.atleast_1d(x).astype(float)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    pass