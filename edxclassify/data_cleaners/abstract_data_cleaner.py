'''
	Format raw data passed from the driver in a way
	that's compatible with particular classifiers.

	Each classifier implementation should specify the format in which it wants
	its data. If your particular dataset's format doesn't match the classifier's
	format, then you should write a DataCleaner to reformat it.
'''

from abc import ABCMeta, abstractmethod

class DataCleaner(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def process_records(self, records):	
		pass
