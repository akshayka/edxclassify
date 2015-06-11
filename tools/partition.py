import csv
import argparse
import pickle
import pprint

from edxclassify.feature_spec import FEATURE_COLUMNS

parser = argparse.ArgumentParser(description='parition by course')
parser.add_argument('goldfile', type=str, help='gold file -- the dataset')
parser.add_argument('course_names', type=str, nargs='+',
                    help='list of courses to include in set')
parser.add_argument('-o', '--output_prefix', type=str, help='prefix')
args = parser.parse_args()

with open(args.goldfile, 'rU') as goldfile:
    goldlist = pickle.load(goldfile)

    course_names = set([record[10] for record in goldlist if record[10] != ''])
    pprint.pprint(course_names)

    # Generate a pair of goldfiles:
    # One with only courses in cn, one without cn
    with open(args.output_prefix + '_train', 'wb') as with_cn,\
         open(args.output_prefix + '_test', 'wb') as without_cn:
        print args.course_names
        with_cn_list = [record for record in goldlist
                                if record[FEATURE_COLUMNS['course_name']]
                                    in args.course_names]
        print len(with_cn_list)
        without_cn_list = [record for record in goldlist
                                    if record[FEATURE_COLUMNS['course_name']]
                                        not in args.course_names]
        print len(without_cn_list)
        pickle.dump(with_cn_list, with_cn)
        pickle.dump(without_cn_list, without_cn)
