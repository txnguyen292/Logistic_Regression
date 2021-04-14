import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, TypeVar, Union

from matplotlib.colors import ListedColormap

from config import CONFIG

import logging
from logzero import setup_logger

logger = setup_logger(__file__, level=logging.INFO, logfile=str(CONFIG.report / "logisticReg.log"))

cmap_bold = ListedColormap(["#FF0000", "#0000FF"])
cmap_light = ListedColormap(["#FFBBBB", "#BBBBFF"])

Vector = np.array
Matrix = np.array
T = TypeVar("T")

def main():
    N = 1000
    D = 2
    X0 = np.random.randn((N//2),D) + np.array([1, 1])
    X1 = np.random.randn((N//2),D) + np.array([-1, -1])
    X = np.vstack((X0, X1))
    
    y = np.array([0]*(N//2) + [1]*(N//2))
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c = y, alpha = 0.5)
    plt.show()
    
    log_reg = LogisticRegression()
    log_reg.fit(X, y, eta = 1e-1, show_curve = True)
    y_hat = log_reg.predict(X)
    
    print(f"Training Accuracy: {accuracy(y, y_hat):0.4f}")
    
    x1 = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 1000)
    x2 = -(log_reg.b/log_reg.w[1]) - (log_reg.w[0]/log_reg.w[1])*x1
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c = y, alpha = 0.5)
    plt.plot(x1, x2, color = "#000000", linewidth = 2)
    plt.show()
    
    xx1, xx2 = np.meshgrid(x1, x1)
    Z = log_reg.predict(np.c_[xx1.ravel(),xx2.ravel()]).reshape(*xx1.shape)
    
    plt.figure()
    plt.pcolormesh(xx1, xx2, Z, cmap = cmap_light)
    plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap_bold)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()

def sigmoid(h: Vector):
    return 1/(1+np.exp(-h))

def binary_cross_entropy(y: Vector, p_hat: Vector):
    return -(1 / len(y)) * np.sum(y*np.log(p_hat)+(1-y)*np.log(1-p_hat)) # Log loss

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

class LogisticRegression:
    """Implement logistic regression
    """
    def __init__(self, thresh=0.5):
        self.thresh = thresh
        self.w = None
        self.b = None

    def fit(self, X: List[List[float]], y: Vector, eta=1e-3, epochs=1e3, show_curve=False):
        epochs = int(epochs)
        N, D = X.shape
        self.w = np.random.rand(D)
        self.b = np.random.randn(1)

        J = np.zeros(epochs)

        for epoch in range(epochs):
            p_hat = self.__forward(X)
            J[epoch] = binary_cross_entropy(y, p_hat)
            self.w -= eta*(1 / N)*X.T@(p_hat-y)
            self.b -= eta*(1 / N)*np.sum(p_hat - y)

        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")

    def __forward(self, X: Matrix):
        return sigmoid(X@self.w+self.b)
    
    def predict(self, X: Matrix):
        return (self.__forward(X) >= self.thresh).astype(np.int32)

# multiple multicariate logistic regression

def softmax(h: Matrix) -> Vector:
    """

    Args:
        h (Matrix): [description]

    Returns:
        Vector: [description]
    """
    return (np.exp(h.T) / np.sum(np.exp(h), axis=1)).T

def cross_entropy(Y: Vector, P_hat: Vector) -> Vector:
    """compute cross entropy

    Args:
        Y (Vector): ground truth
        P_hat (Vector): predictions

    Returns:
        Vector: [description]
    """
    return -(1 / len(Y)) * np.sum(np.sum(Y * np.log(P_hat), axis=1), axis=0)

def indices_to_one_hot(data: Vector, nb_classes: int) -> Matrix:
    """one hot encoding

    Args:
        data (Vector): [description]
        nb_classes (int): [description]

    Returns:
        Matrix: [description]
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

class MVLogisticRegression:
    """Multivariate Logistic Regression
    """
    def __init__(self, thresh=0.5, binary: bool=False):
        self.thresh = thresh
        self.binary = binary
    def fit(self, X: Matrix, y: Vector, eta = 2e-1, epochs = 1e3, show_curve=False) -> None:
        epochs = int(epochs)
        N, D = X.shape
        K = len(np.unique(y))
        y_values = np.unique(y, return_index=False)
        Y = indices_to_one_hot(y, K).astype(int)
        self.W = np.random.randn(D, K)
        self.B = np.random.randn(1, K)

        J = np.zeros(int(epochs))

        for epoch in range(epochs):
            P_hat = self.__forward__(X, binary=self.binary)
            J[epoch] = cross_entropy(Y, P_hat)
            # weight and bias update rules
            self.W -= eta*(1 / N) * X.T@(P_hat - Y)
            self.B -= eta*(1 / N) * np.sum(P_hat, axis=0)
    
        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("Training Curve")
    
    def __forward__(self, X: Matrix, binary=False) -> Vector:
        if binary:
            return sigmoid(X@self.W + self.B)
        else:
            return softmax(X@self.W + self.B)

    def predict(self, X: Union[Matrix, Vector]) -> Vector:
        return np.argmax(self.__forward__(X, binary=self.binary), axis=1)

if __name__ == "__main__":
    # assert softmax()
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    # assert indices_to_one_hot(y, len(np.unique(y))) == res, "your onehot is wrong"
    data = pd.read_csv(CONFIG.data / "final" / "TripGaussKNN.csv", header = 0)
    X = data.iloc[:, :2].to_numpy()
    y = data.iloc[:, -1].to_numpy()
