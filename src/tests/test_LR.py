from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import sys
from pathlib import Path
file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(file_dir.parent))


from myLR import MVLogisiticRegression, LogisticRegression
from LogisticRegression import sigmoid, softmax
from scipy.special import softmax as scsoftmax
X, y = load_iris(return_X_y=True)
X_binary = X[(y == 1) | (y == 0)]
y_binary = y[(y == 1) | (y == 0)]
# def test_MyLR():
#     lr = MVLogisiticRegression()
#     lr.fit(X, y)
#     y_hat = lr.predict(X)
#     acc = np.mean(y == y_hat)
#     assert isinstance(acc, float), "Check your MVLogisticRegression Implementation!"
#     assert acc > 0.6, "Something's wrong with your model!"

def test_MyLR():
    lr = LogisticRegression()
    lr.fit(X_binary, y_binary)
    y_hat = lr.predict(X_binary)
    acc = np.mean(y_binary == y_hat)
    assert isinstance(acc, float), "wrong type of results"
    assert acc > 0.5, "your model performs poorly"


if __name__ == "__main__":
    pass