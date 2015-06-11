'''
Classifier API

TODO
'''
from abc import ABCMeta, abstractmethod


class Classifier(object):
    __metaclass__ = ABCMeta


    @abstractmethod
    def train(self, X, y):
        pass


    @abstractmethod
    def test(self, X, y):
        """Test on example matrix X
        
        parameters:
        -----------
        X : {array-like, sparse-matrix}, shape = [n_samples, n_features]
            input examples
        y : array, shape = [n_samples] (optional)
            true labels of each sample if known, None otherwise

        returns:
        --------
        If y is None, returns an array of shape = [n_samples],
           with the predicted class label for each sample.
        Otherwise, returns a tuple
            (predictions, [precision, recall, f1, kappa])
        """
        pass


    @abstractmethod
    def cross_validate(self, X, y):
        """Cross validation

        parameters:
        -----------
        X : {array-like, sparse-matrix}, shape = [n_samples, n_features]
            input examples
        y : array, shape = [n_samples]
            true labels of each sample

        returns:
        --------
        Return [PRECISION, RECALL, F1, KAPPA]_train,
               [PRECISION, RECALL, F1, KAPPA]_test,
        where

        PRECISION : list, shape = [n_folds, n_labels]
            [ [ precision per label ]_fold_1, ..., [ precision per label ]_fold_n ]
       
        RECALL : list, shape = [n_folds, n_labels]
            [ [ recall per label ]_fold_1, ..., [ recall per label ]_fold_n ]
           
        F1 : list, shape = [n_folds, n_labels]
            [ [ f1-score per label ]_fold_1, ..., [ f1-score per label ]_fold_n ]

        KAPPA : list, shape = [n_folds, n_labels]
            [ [ kappa per label ]_fold_1, ..., [ kappa per label ]_fold_n ]
        """
        pass
