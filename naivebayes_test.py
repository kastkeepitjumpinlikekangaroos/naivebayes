import math
import unittest

import numpy as np
from sklearn import datasets

import naivebayes

class TestNaiveBayesClassifier(unittest.TestCase):
    def test_predict(self):
        c = naivebayes.NaiveBayesClassifer()
        print('loading iris')
        X, y = datasets.load_iris(return_X_y=True)
        indices = np.random.permutation(len(X))
        X_train = X[indices[:-10]]
        y_train = y[indices[:-10]]
        X_test = X[indices[-10:]]
        y_test = y[indices[-10:]]
        c.fit(X_train, y_train)
        pred = c.predict(X_test)
        print(pred, y_test)


if __name__ == '__main__':
    unittest.main()
