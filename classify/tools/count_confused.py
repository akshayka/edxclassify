import csv
import argparse
import pickle
from classify.feature_spec import FEATURE_COLUMNS

parser = argparse.ArgumentParser(description='parition by course')
parser.add_argument('goldfile', type=str, help='gold file -- the dataset')
args = parser.parse_args()

with open(args.goldfile, 'rU') as goldfile:
    goldlist = pickle.load(goldfile)
    course_names = set([record[10] for record in goldlist if record[10] != ''])
    for cn in course_names:
        confused = [record for record in goldlist if record[10] == cn and\
                        float(record[FEATURE_COLUMNS['confusion']]) > 4]
        neutral = [record for record in goldlist if record[10] == cn and\
                        float(record[FEATURE_COLUMNS['confusion']]) == 4]
        kno = [record for record in goldlist if record[10] == cn and\
                        float(record[FEATURE_COLUMNS['confusion']]) < 4]
        print cn + ' confused %d ' % len(confused)
        print cn + ' neutral %d ' % len(neutral)
        print cn + ' knowledgable %d ' % len(kno)
