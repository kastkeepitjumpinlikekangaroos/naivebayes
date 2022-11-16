import multiprocessing
from functools import partial

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NaiveBayesClassifer(ClassifierMixin, BaseEstimator):
    """ A classifier which implements a naive bayes algorithm.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def fit(self, X, y):
        """Fits the model
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ Uses naive bayes algorithm to make a prediction
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the class with
            the highest probability according to the algorithm outlined
            here:
            https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Constructing_a_classifier_from_the_probability_model
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        func = partial(self._bayes_classification, X=self.X_, y=self.y_, classes=self.classes_)
        with multiprocessing.Pool() as p:
            pred = p.map(func, X)
        return pred

    @staticmethod
    def _bayes_classification(x, X, y, classes):
        # merge training data with labels
        data = np.column_stack((X, y))
        class_probs = np.array([])
        # get probability for each class
        for cls in classes:
            # bitmask for the rows in this class
            class_eq_bitmask = data[:, -1] == cls
            # P(cls)
            prob = len(data[class_eq_bitmask]) / len(data)
            for feature in np.arange(len(x)):
                # P(x[feature] | cls)
                prob_ = (
                    len(data[
                        (class_eq_bitmask) &
                        (data[:, feature] == x[feature])  # rows where class and feature value match
                    ])
                    /
                    len(data[class_eq_bitmask])
                )
                # P(cls) * (P(x[feature] | cls) for each feature)
                prob = prob * prob_
            class_probs = np.append(class_probs, prob)
        # return class with max probability
        return classes[np.argmax(class_probs)]
