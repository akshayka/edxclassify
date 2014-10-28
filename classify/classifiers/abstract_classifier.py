'''
Classifier API

TODO
'''
from abc import ABCMeta, abstractmethod


class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def labels(self):
        pass

    @abstractmethod
    def train(self, training_examples):
        pass

    @abstractmethod
    def test(self, test_examples):
        pass
