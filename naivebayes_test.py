import math
import unittest

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score


import naivebayes


class TestNaiveBayesClassifier(unittest.TestCase):
    def test_predict_breast_cancer(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        t = KBinsDiscretizer(n_bins=2, encode='onehot-dense', strategy='kmeans')
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("BC Accuracy : ", accuracy_score(y_test, y_pred))

    def test_iris(self):
        X, y = datasets.load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        t = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans')
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Iris Accuracy : ", accuracy_score(y_test, y_pred))

    def test_wine(self):
        X, y = datasets.load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        t = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans')
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Wine Accuracy : ", accuracy_score(y_test, y_pred))

    def test_digits(self):
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Digits Accuracy : ", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    unittest.main()
