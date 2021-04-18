from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import sys
from pathlib import Path
file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(file_dir.parent))


from myLR import MVLogisiticRegression
from LogisticRegression import sigmoid, softmax
from scipy.special import softmax as scsoftmax
X, y = load_iris(return_X_y=True)

# def test_MyLR():
#     lr = MVLogisiticRegression()
#     lr.fit(X, y)
#     y_hat = lr.predict(X)
#     acc = np.mean(y == y_hat)
#     assert isinstance(acc, float), "Check your MVLogisticRegression Implementation!"
#     assert acc > 0.6, "Something's wrong with your model!"

def test_sigmoid():
    h = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    t1 = scsoftmax(h, axis=1)
    t2 = softmax(h)
    assert np.allclose(t1, t2), f"Compare: {t1} vs {t2}"

if __name__ == "__main__":
    pass
