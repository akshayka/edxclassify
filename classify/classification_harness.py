import argparse
from classifier_factory import make_classifier
from data_cleaner_factory import make_data_cleaner
import pickle
from tabulate import tabulate


def evaluate_predictions(clf_results, possible_labels):
    predictions = clf_results[0]
    correct_map = {label: 0 for label in possible_labels}
    ex_per_label = {label: 1 for label in possible_labels}
    for example, prediction, label in predictions:
        ex_per_label[label] = ex_per_label[label] + 1
        if prediction == label:
            correct_map[label] = \
                correct_map[label] + 1
    accuracies = []
    for label in possible_labels:
        accuracies.append(float(correct_map[label]) /
                          float(ex_per_label[label]))
    accuracies.append(clf_results[1])
    return accuracies


def invoke_classifier(classifier, training_files,
                      test_files, data_cleaner=None):
    results = []
    labels = classifier.labels()
    for fold, (train_file, test_file) in enumerate(
            zip(training_files, test_files)):
        with open(train_file, 'rb') as train, \
                open(test_file, 'rb') as test:
            # Slice off headers
            training_examples = pickle.load(train)[1:]
            test_examples = pickle.load(test)[1:]
            if data_cleaner is not None:
                training_examples = \
                    data_cleaner.process_records(training_examples)
                test_examples = \
                    data_cleaner.process_records(test_examples)
            classifier.train(training_examples)
            results.append([str(fold)] +
                           evaluate_predictions(classifier.test(
                                                test_examples), labels))
    header = ['Fold Number'] + ['Label' + str(label) for label in labels] + \
             ['Overall Accuracy']
    print tabulate(results, header, tablefmt='grid')


def main():
    parser = argparse.ArgumentParser(description='applies a classifier to '
                                     'train, test folds generated using '
                                     'ingest_datasets.py')
    parser.add_argument('-d', '--data_cleaner', type=str,
                        help='apply a DataCleaner to the data ingested by '
                        'ingest_datasets.py; see data_cleaner_factory.py for '
                        'a list of supported cleaners')
    parser.add_argument('classifier', type=str,
                        help='apply a particular classifier to the folds; see '
                        'classifier_factory.py for a list of supported '
                        'classifiers')
    parser.add_argument('-tr', '--training_files', required=True, type=str,
                        nargs='+',
                        help='training files produced by ingest_datasets.py')
    parser.add_argument('-tst', '--test_files', required=True, type=str,
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

if __name__ == '__main__':
    main()
