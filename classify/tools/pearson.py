import argparse
from classify.feature_spec import FEATURE_COLUMNS
import cPickle
import skll

parser = argparse.ArgumentParser(description='compute pearson\'s corellation '
                                'coefficient between two variables')
parser.add_argument('goldset', type=str, help='input file -- the dataset')
parser.add_argument('var1', type=str, help='first of two variables, '
                    'corresponding to a feature name in feature_spec.py')
parser.add_argument('var2', type=str, help='second of two variables, '
                    'corresponding to a feature name feature_spec.py')
args = parser.parse_args()

goldfile = cPickle.load(open(args.goldset, 'rb'))
var1 = [float(record[FEATURE_COLUMNS[args.var1]]) for record in goldfile]
var2 = [float(record[FEATURE_COLUMNS[args.var2]]) for record in goldfile]

r = skll.metrics.pearson(var1, var2)
print 'The correlation coefficient between ' + args.var1 + ' and '\
        + args.var2 + ' is ' + str(r)
