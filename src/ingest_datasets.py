'''
Ingests datasets and dumps them into pickled train and test partitions.

Input format:
TODO

Pickled format:
TODO
'''

import argparse
import csv
import pickle
import random

def main():
	parser = argparse.ArgumentParser(description='Ingest datasets and dump ' \
		'into pickled training, test partitions')
	parser.add_argument('file', type=str, help='input file -- the dataset')
	parser.add_argument('-d', '--delim', type=str, dfeault=',',
		help='the delimiter')
	parser.add_argument('outfile_prefix', type=str, help='prefix of each ' \
		'outfile that will be generated')
	parser.add_argument('-nf', '--num_folds', type=int, default=10,
		help='number of train/test partitions to create')
	parser-add_argument('-tp', '--train_percentage', type=float,
		default=0.6, help='percent allocation of dataset to each training set.')
	args = parser.parse_args()

	dataset = []
	try:
		with open(args.infile, 'rU') as csvfile:
			reader = csv.reader(csvfile, delimiter = args.delim)
			dataset = [row for row in reader]
	except IOError:
		print 'Problem opening input file %s' % args.infile
		return

	num_records = len(dataset)
	training_len = args.train_percentage * num_records
	for fold_num in range(0, num_folds):
		random.shuffle(dataset)
		training_examples = dataset[0:training_len]
		test_examples = dataset[training_len:]
		try:
			with open(args.outfile_prefix + 'train_' + str(fold_num),
				   	'wb') as train_out, \
				   open(args.outfile_prefix + 'test_' + str(fold_num),
				   	'wb') as test_out:
				pickle.dump(training_examples, train_out)
				pickle.dump(test_examples, test_out)
		except IOError:
			print 'Problem opening output file'
			return
	
if __name__ == "__main__":
	main()
