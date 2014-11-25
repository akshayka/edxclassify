import argparse
import harness
import sys

def main():
    parser = argparse.ArgumentParser(description='applies a classifier to '
                                     'train, test folds generated using '
                                     'ingest_datasets.py')
    parser.add_argument('data_file', type=str, help='ingested data file')
    parser.add_argument('data_cleaner', type=str,
                        help='apply a DataCleaner to the data ingested by '
                        'ingest_datasets.py; see data_cleaner_factory.py for '
                        'a list of supported cleaners')
    parser.add_argument('classifier', type=str,
                        help='apply a particular classifier to the data; see '
                        'classifier_factory.py for a list of supported '
                        'classifiers')
    parser.add_argument('output_prefix', type=str, help='output file prefix')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    prefix =  [args.data_file, args.data_cleaner, args.classifier, \
               '-t', '5', '-c', '-n', '-p', '0.4']
    argsuffixes = [
                [],
                ['-l'],
                ['-fs', '2'],
                ['-tf']]
    for i in range (200, 1200, 200):
        argsuffixes = argsuffixes + [['-kb', str(i), '-l']]
        argsuffixes = argsuffixes + [['-tf', '-kb', str(i), '-l']]

    for argsuffix in argsuffixes:
        harness_args = prefix + argsuffix
        if args.verbose:
            print 'executing ' + ' '.join(harness_args)

        stdout_save = sys.stdout
        sys.stdout = open(args.output_prefix + '-'.join(argsuffix) + '.txt',
                          'wb')
        harness.main(harness_args)
        sys.stdout.close()
        sys.stdout = stdout_save

if __name__ == '__main__':
    main()
