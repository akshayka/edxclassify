import argparse
from classifier_factory import make_classifier
from data_cleaner_factory import make_data_cleaner
import pickle
from tabulate import tabulate


def invoke_classifier(classifier, data_filename, data_cleaner):
    results = []
    labels = data_cleaner.labels()
    with open(data_filename, 'rb') as infile:
        # Slice off headers
        # TODO: Headers aren't getting used anywhere,
        # perhaps don't take them in ingest_dataset
        dataset = pickle.load(infile)[1:]
        dataset =  data_cleaner.process_records(dataset)
        cv_results = classifier.cross_validate(dataset)

    dcname = ''
    if data_cleaner is not None:
        dcname = data_cleaner.name
    print 'Classification results for file %s ...;\nusing classifier %s and ' \
          'data_cleaner %s' % (data_filename, classifier.name, dcname)
    for i, label in enumerate(labels):
        print str(label) + ' had %d occurrences' % \
              classifier.label_counts[i]

    header = ['fold']
    for label in labels:
        label_str = str(label)
        header.append(label_str + ': precision')
        header.append(label_str + ': recall')
        header.append(label_str + ': f1')
    results = []
    print cv_results
    avgs = [0] * len(labels) * 3
    fold = 1
    for p, r, f in zip(cv_results[0], cv_results[1], cv_results[2]):
        results.append([str(fold)] +
                       [p[0], r[0], f[0]] +
                       [p[1], r[1], f[1]] +
                       [p[2], r[2], f[2]])
        for i in range(len(avgs) / 3):
            a_i = i * 3
            avgs[a_i] = avgs[a_i] + p[i]
            avgs[a_i+1] = avgs[a_i+1] + r[i]
            avgs[a_i+2] = avgs[a_i+2] + f[i]
        fold = fold + 1
    avgs = [avg / (fold - 1) for avg in avgs]
    avgs = ['avg'] + avgs
    results.append(avgs)
    print results
    print tabulate(results, header, tablefmt='grid')


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
    args = parser.parse_args()

    classifier = make_classifier(args.classifier)
    data_cleaner = None
    if args.data_cleaner is not None:
        data_cleaner = make_data_cleaner(args.data_cleaner)
    invoke_classifier(classifier, args.data_file, data_cleaner)


if __name__ == '__main__':
    main()
