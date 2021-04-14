from sklearn.datasets import load_iris
from myLR import MVLogisiticRegression

X, y = load_iris(return_X_y=True)

def test_MyLR():
    lr = MVLogisiticRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X, y)
    acc = np.mean(y == y_hat)
    assert isinstance(acc, float), "Check your MVLogisticRegression Implementation!"

if __name__ == "__main__":
    pass
