import argparse
from classify.feature_spec import FEATURE_COLUMNS
import cPickle
import csv

parser = argparse.ArgumentParser(description='dump sentiment, confusion, etc.')
parser.add_argument('goldset', type=str, help='input file -- the dataset')
parser.add_argument('csv', type=str, help='dump to csv file named as such')

args = parser.parse_args()

goldfile = cPickle.load(open(args.goldset, 'rb'))
variables = ['confusion', 'urgency', 'sentiment', 'opinion',\
                'answer', 'question']
entries = []
for row in goldfile:
    entries.append([row[FEATURE_COLUMNS[v]] for v in variables])

with open(args.csv, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(variables)
    writer.writerows(entries)
