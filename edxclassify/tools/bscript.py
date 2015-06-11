import edxclassify.harness as harness
import sys
import argparse

parser = argparse.ArgumentParser(description='run logistic regression on some '
    'courses.')
parser.add_argument('suffix', type=str, help='suffix of output file')
parser.add_argument('-c', '--chained', action='store_true',
    help='use chained classifiers')
args = parser.parse_args()

hum = 'data/gold_sets/humanities_gold_v9'
med = 'data/gold_sets/medicine_gold_v9'
math = 'data/gold_sets/math_gold_v9'
stats_2013 =\
    'data/gold_sets/partitions/single_courses/'\
    'medicine_gold_v8_Medicine_HRP258_Statistics_in_Medicine'
stats_2014 =\
    'data/gold_sets/partitions/single_courses/'\
    'medicine_gold_v8_Medicine_MedStats_Summer2014'
files = [(hum, 'hum'), (med, 'med'), (math, 'math'), (stats_2013,\
            'stats_in_med_2013'), (stats_2014, 'stats_in_med_2014')]


cmd = 'logistic -p 0.27 -b'
if args.chained:
    cmd = cmd + ' -c'

for f, name in files:
    harness_args = [f, 'confusion'] + cmd.split(' ')
    output_file = 'data/ablative_analysis/' + name + '_bcfn_' + args.suffix
    print 'executing ' + ' '.join(harness_args) + ' ...'
    print 'output: ' + output_file
    stdout_save = sys.stdout
    sys.stdout = open(output_file, "wb")
    harness.main(harness_args)
    sys.stdout.close()
    sys.stdout = stdout_save

