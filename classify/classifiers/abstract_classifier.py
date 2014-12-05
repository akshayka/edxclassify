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
        pass


    # Return [PRECISION, RECALL, F1]_train, [PRECISION, RECALL, F1]_test
    #
    # PRECISION (LIST)
    # [ [ precision per label ]_fold_1, ..., [ precision per label ]_fold_n ]
    #
    # RECALL (LIST)
    # [ [ recall per label ]_fold_1, ..., [ recall per label ]_fold_n ]
    #
    # F1-SCORE (LIST)
    # [ [ f1-score per label ]_fold_1, ..., [ f1-score per label ]_fold_n ]
    @abstractmethod
    def cross_validate(self, X, y):
        pass
