import argparse
from classify.feature_spec import FEATURE_COLUMNS
import cPickle
import random

parser = argparse.ArgumentParser(description='random subset of confused posts.')
parser.add_argument('course', type=str)
args = parser.parse_args()

records = cPickle.load(open(args.course, 'rb'))
confused = []
for record in records:
    if float(record[FEATURE_COLUMNS['confusion']]) > 4:
        confused.append(record[0] + ' : ' + record[FEATURE_COLUMNS['confusion']])

indices = random.sample(xrange(len(confused)), 10)
for i in indices:
    print confused[i]
