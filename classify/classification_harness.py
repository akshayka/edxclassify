import argparse
from classifier_factory import make_classifier
from data_cleaner_factory import make_data_cleaner


def invoke_classifier(classifier, training_files,
                      test_files, data_cleaner=None):


def main():
    parser = argparse.ArgumentParser(description='applies a classifier to '
                                     'train, test folds generated using '
                                     'ingest_datasets.py')
    parser.addArgument('-d', '--datacleaner', type=str,
                       help='apply a DataCleaner to the data ingested by '
                       'ingest_datasets.py; see data_cleaner_factory.py for '
                       'a list of supported cleaners')
    parser.addArgument('classifier', type=str,
                       help='apply a particular classifier to the folds; see '
                       'classifier_factory.py for a list of supported '
                       'classifiers')
    parser.addArgument('training_files', metavar='training_file', type=str,
                       nargs='+',
                       help='training files produced by ingest_datasets.py')
    parser.addArgument('test_files', metavar='test_file', type=str,
                       nargs='+',
                       help='test files produced by ingest_datasets.py')
    args = parser.parse_args()

    if len(args.training_files) != len(args.test_files):
        print 'Error: number of training files (%d) does not match ' \
              'number of test files (%d).' % \
              (len(args.training_files), len(args.test_files))
        return

    classifier = make_classifier(args.classifier)
    data_cleaner = None
    if args.data_cleaner is not None:
        data_cleaner = make_data_cleaner(args.data_cleaner)
    invoke_classifier(classifier, args.training_files,
                      args.test_files, data_cleaner)


if __name__ == 'main':
    main()
