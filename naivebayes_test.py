import math
import unittest

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


import naivebayes


class TestNaiveBayesClassifier(unittest.TestCase):
    def test_wine_dummy(self):
        X, y = datasets.load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = DummyClassifier(strategy='uniform')
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Dummy Wine Accuracy: ", accuracy_score(y_test, y_pred))

    def test_wine_gaussian(self):
        X, y = datasets.load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = GaussianNB()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("GaussianNB Wine Accuracy: ", accuracy_score(y_test, y_pred))

    def test_wine_probabilistic(self):
        X, y = datasets.load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        t = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans')
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Wine Accuracy: ", accuracy_score(y_test, y_pred))

    def test_digits_dummy(self):
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = DummyClassifier(strategy='uniform')
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("Dummy Digits Accuracy: ", accuracy_score(y_test, y_pred))

    def test_digits_gaussian(self):
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = GaussianNB()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        print("GaussianNB Digits Accuracy: ", accuracy_score(y_test, y_pred))

    def test_digits(self):
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        c = naivebayes.NaiveBayesClassifer()
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)

        print("Digits Accuracy: ", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    unittest.main()
