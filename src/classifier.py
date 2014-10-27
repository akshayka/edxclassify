'''
Classifier API

TODO
'''
from abc import ABCMeta, abstractmethod

class Classifier(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def train(self, raw_training_examples):
		pass

	@abstractmethod
	def test(self, raw_test_examples):
		pass
