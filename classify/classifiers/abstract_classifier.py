'''
Classifier API

TODO
'''
from abc import ABCMeta, abstractmethod


class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, training_examples):
        pass

    @abstractmethod
    def test(self, test_examples):
        pass

    # Return [PRECISION, RECALL, F1]
    #
    # PRECISION (LIST)
    # [ [ precision  per label ]_fold_1, ..., [ precision per label ]_fold_n ]
    #
    # RECALL (LIST)
    # [ [ recall per label ]_fold_1, ..., [ recall per label ]_fold_n ]
    #
    # F1-SCORE (LIST)
    # [ [ f1-score per label ]_fold_1, ..., [ f1-score per label ]_fold_n ]
    @abstractmethod
    def cross_validate(self, examples):
        pass
