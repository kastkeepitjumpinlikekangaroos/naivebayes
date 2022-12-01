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
    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    data_: ndarray of shape (n_samples, n_features + 1)
        matrix containing input X with y stacked as the last column
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
        self.class_priors_ = np.array([len(y[y == cls]) for cls in self.classes_])
        self.data_ = np.column_stack((X, y))

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
        check_is_fitted(self, ['classes_', 'class_priors_', 'data_'])

        # Input validation
        X = check_array(X)
        func = partial(self._bayes_classification, data=self.data_, classes=self.classes_, class_priors=self.class_priors_)
        with multiprocessing.Pool() as p:
            pred = p.map(func, X)
        return pred

    @staticmethod
    def _bayes_classification(X, data, classes, class_priors):
        class_probs = np.array([])
        for cls, cls_prior in zip(classes, class_priors):
            # bitmask for the rows in this class
            class_eq_bitmask = data[:, -1] == cls
            # P(cls)
            prob = cls_prior
            for feature in np.arange(len(X)):
                # P(x[feature] | cls)
                prob_ = (
                    len(data[
                        (class_eq_bitmask) &
                        (data[:, feature] == X[feature])  # rows where class and feature value match
                    ])
                    /
                    len(data[class_eq_bitmask])
                )
                # P(cls) * (P(x[feature] | cls) for each feature)
                prob = prob * prob_
            class_probs = np.append(class_probs, prob)
        # return class with max probability
        return classes[np.argmax(class_probs)]
