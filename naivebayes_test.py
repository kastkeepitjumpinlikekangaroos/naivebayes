import math
import unittest

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score


import naivebayes


class TestNaiveBayesClassifier(unittest.TestCase):
    def test_predict(self):
        c = naivebayes.NaiveBayesClassifer()

        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        sc = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='kmeans')
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Accuracy : ", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    unittest.main()
