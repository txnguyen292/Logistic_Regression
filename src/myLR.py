import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import TypeVar

X = TypeVar("X", int, float, str)
# y = TypeVar("y", int)

#===================================================================
#======== HELPER FUNCTIONS =========================================
#===================================================================

def sigmoid(h: "np.floating[X]") -> "np.ndarray[float]":
    """Implementing sigmoid activation

    Args:
        h (ArrayLike): [description]

    Returns:
        ArrayLike: [description]
    """
    return 1 / (1 + np.exp(-h))

def softmax(h: ArrayLike) -> ArrayLike:
    """Implementing softmax function for logistic regression
    Reference: https://en.wikipedia.org/wiki/Softmax_function
    Args:
        y (ArrayLike): data after matrix transformation with weights

    Returns:
        ArrayLike: data activated with softmax
    """
    return (np.exp(h.T) / np.sum(np.exp(h), axis=1)).T

def binary_cross_entropy(y: ArrayLike, p: ArrayLike) -> ArrayLike:
    """Implement binary cross entropy

    Args:
        y (ArrayLike): targets
        p (ArrayLike): predictions

    Returns:
        ArrayLike: loss
    """
    y = np.array(y)
    p = np.array(p)
    return 1 / len(y) * (-y * np.log(p) - (1 - y) * np.log(1 - p))


def cross_entropy(Y: ArrayLike, P_hat: ArrayLike) -> ArrayLike:
    """Compute the loss
    reference: https://en.wikipedia.org/wiki/Cross_entropy
    Args:
        Y (ArrayLike): Vector of ground truths
        P_hat (ArrayLike): Vector of predictions

    Returns:
        ArrayLike: [description]
    """
    return -(1 / len(Y)) * np.sum(np.sum(Y * np.log(P_hat), axis=1), axis=0)

def indices_to_one_hot(data: ArrayLike, nb_classes: int) -> ArrayLike:
    """One-hot encoding

    Args:
        data (ArrayLike): [description]
        nb_classes (int): [description]

    Returns:
        ArrayLike: [description]
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

#-------------------------------------------------------------------

class MVLogisiticRegression:
    def __init__(self, thresh=0.5, binary: bool=False):
        self.thresh = thresh
        self.binary = binary
    def fit(self, X, y, eta=1e-3, epochs=1e-3, show_curve=False) -> None:
        epochs = int(epochs)
        N, D = X.shape
        K = len(np.unique(y))
        Y = indices_to_one_hot(y, K).astype(int)
        self.W = np.random.randn(D, K)
        self.B = np.random.randn(1, K)

        J = np.zeros(int(epochs)) #losses

        for epoch in range(epochs):
            P_hat = self.__forward__(X, binary=self.binary)
            J[epoch] = cross_entropy(Y, P_hat)
            self.W -= eta*(1 / N) * X.T@(P_hat - Y)
            self.B -= eta*(1 / N) * np.sum(P_hat, axis=0)

        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("Training Curve")
    def predict(self, X):
        return np.argmax(self.__forward__(X, binary=self.binary), axis=1)
        

            
    def __forward__(self, X: ArrayLike, binary: bool=False) -> np.ndarray:
        if binary:
            return sigmoid(X@self.W + self.B)
        else:
            return softmax(X@self.W + self.B)

class LogisticRegression:

    def __init__(self, thresh=0.5):
        self.thresh = thresh
        self.w = None
        self.b = None

    def fit(self, X, y, eta=1e-3, epochs=1e3):
        epochs = int(epochs)
        N, D = X.shape
        self.w = np.random.randn(D)
        self.b = np.random.randn(1)

        J = np.zeros(epochs)

        for epoch in range(epochs):
            p_hat = self.__forward(X)
            J[epoch] = binary_cross_entropy(y, p_hat)
            self.w -= eta * (1 / N) * X.T@(p_hat - y)
            self.b -= eta * (1 / N) * np.sum(p_hat - y)

    def __forward(self, X):
        return sigmoid(X@self.w + self.b)

    def predict(self, X):
        return (self.__forward(X) >= self.thresh).astype(np.int32)
        
if __name__ == "__main__":
    def my_sigmoid(h):
        return 1 / (1 + np.exp(-h))

    def my_softmax(h):
        return np.exp(-h.T) / np.sum(1 + np.exp(-h), axis=1)