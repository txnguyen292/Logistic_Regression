import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.special import softmax as scsoftmax
from scipy.special import expit
from pathlib import Path
import tensorflow as tf
file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(file_dir.parent))
from LogisticRegression import binary_cross_entropy as temp_binary_cross_entropy
sys.path.insert(0, str(file_dir.parent))

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def softmax(h):
    return (np.exp(h.T) / np.sum(np.exp(h), axis=1)).T

def binary_cross_entropy(y, p):
    return -1 / len(y) * (-y * np.log(p) - (1 - y) * np.log(1 - p))

def cross_entropy(y, p):
    # return -1 / len(y) * 
    return -1/len(y) * (np.sum(np.sum(y * np.log(p), axis=1), axis=0))

def test_binary_cross_entropy():
    x = np.array([[1, 2, 3], [2, 3, 5]])
    p = np.array([0.2, 0.3, 0.5])
    t1 = temp_binary_cross_entropy(x, p)
    t2 = binary_cross_entropy(x, p)
    return np.allclose(t1, t2), f"Compare {t1} vs. {t2}"

def test_sigmoid():
    x = np.array([[1, 2, 3], [2, 3, 5]])
    assert np.allclose(sigmoid(x), expit(x)), f"Compare: {sigmoid(x)} vs. {expit(x)}"

def test_softmax():
    h = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    t1 = scsoftmax(h, axis=1)
    t2 = softmax(h)
    assert np.allclose(t1, t2), f"Compare: {t1} vs {t2}"

if __name__ == "__main__":
    cc = tf.keras.losses.CategoricalCrossentropy()
    y_true = np.array([[0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    print(cc(y_true, y_pred).numpy())
    print(cross_entropy(y_true, y_pred))