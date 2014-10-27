'''
	Format raw data passed from the driver in a way
	that's compatible with particular classifiers.

	TODO
'''

from abc import ABCMeta, abstractmethod

class DataCleaner(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def process_records(self, records):	
		pass
