import argparse
from classify.feature_spec import FEATURE_COLUMNS
import cPickle
import csv
import skll
from tabulate import tabulate

parser = argparse.ArgumentParser(description='compute pearson\'s corellation '
                                'coefficient for a particular variable, '
                                'against all other variables')
parser.add_argument('goldset', type=str, help='input file -- the dataset')
parser.add_argument('-csv', type=str, help='optionally dump to a csv file')
args = parser.parse_args()

goldfile = cPickle.load(open(args.goldset, 'rb'))
variables = ['confusion', 'urgency', 'sentiment', 'opinion',\
                'answer', 'question']
header = [''] + variables
entries = []

for target in variables:
    entry = [target]
    target_values = [float(record[FEATURE_COLUMNS[target]]) for\
                        record in goldfile]
    for var in variables:
        var_values = [float(record[FEATURE_COLUMNS[var]]) for\
                        record in goldfile]
        entry.append(skll.metrics.pearson(var_values, target_values))
    entries.append(entry)
print tabulate(entries, header, tablefmt='grid')

if args.csv is not None:
    with open(args.csv, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(variables)
        writer.writerows(entries)
