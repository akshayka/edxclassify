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


def to_unicode(row, encoding):
    return [entry.decode(encoding).encode('utf-8') for entry in row]


def main():
    parser = argparse.ArgumentParser(description='Ingest datasets and dump '
                                     'into pickled training, test partitions')
    parser.add_argument('infile', type=str, help='input file -- the dataset')
    parser.add_argument('-d', '--delim', type=str, default=',',
                        help='the delimiter; defaults to ","')
    parser.add_argument('outfile', type=str,
                        help='outfile that will be generated')
    parser.add_argument('-c', '--encoding', type=str,
                        default='utf-8')
    args = parser.parse_args()

    try:
        with open(args.infile, 'rU') as csvfile, \
             open(args.outfile, 'wb') as outfile:
            reader = csv.reader(csvfile, delimiter=args.delim)
            dataset = [to_unicode(row, args.encoding) for row in reader]
            print dataset
            pickle.dump(dataset, outfile)
    except IOError:
        print 'Problem opening file.'
        return


if __name__ == "__main__":
    main()
