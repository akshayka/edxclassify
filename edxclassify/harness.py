import argparse
from edxclassify.classifiers.sklearn_clf import SklearnCLF
from data_cleaner_factory import make_data_cleaner
from sklearn.externals import joblib
import pickle
from tabulate import tabulate

"""
Classification harness, designed for running experiments on the MOOCPosts
dataset.

TODO Finish documentation.
"""

def tabulate_f1_cv_summary(cv_results_tr, cv_results_tst, labels):
    headers = ['Label', 'Train: F1', 'Test: F1']
    avgs = []
    for label in labels:
        avgs = avgs + [[0, 0]]
    fold = 1
    for f_tr, f_tst, in zip(cv_results_tr[2], cv_results_tst[2]):
        for i in range(len(labels)):
            avgs[i][0] = avgs[i][0] + f_tr[i]
            avgs[i][1] = avgs[i][1] + f_tst[i]
        fold = fold + 1
    avgs = [[avg[0] / (fold - 1), avg[1] / (fold - 1)] for avg in avgs]
    for i, label in enumerate(labels):
        avgs[i] = [label] + avgs[i]
    print tabulate(avgs, headers, tablefmt='grid')


def tabulate_full_cv_summary(cv_results, average, labels):
    header = ['fold']
    for label in labels:
        label_str = str(label)
        header.append(label_str + ': precision')
        header.append(label_str + ': recall')
        header.append(label_str + ': f1')
    header.append('kappa')
    results = []

    # Subtract one to account for the 'fold' column
    avgs = [0] * (len(header) - 1)
    fold = 1
    for p, r, f, K in zip(cv_results[0], cv_results[1], cv_results[2],
                        cv_results[3]):
        entry = [str(fold)]
        for i in range(len(labels)):
            entry = entry + [p[i], r[i], f[i]]
        entry.append(K)
        if not average:
            results.append(entry)
        for i in range(len(avgs) / 3):
            a_i = i * 3
            avgs[a_i] = avgs[a_i] + p[i]
            avgs[a_i+1] = avgs[a_i+1] + r[i]
            avgs[a_i+2] = avgs[a_i+2] + f[i]
        avgs[-1] = avgs[-1] + K
        fold = fold + 1
    avgs = [avg / (fold - 1) for avg in avgs]
    avgs = ['avg'] + avgs
    results.append(avgs)
    print tabulate(results, header, tablefmt='grid')


def examples_labels(data_filename, data_cleaner):
    infile = open(data_filename, 'rb')
    dataset = pickle.load(infile)
    infile.close()
    X, y =  zip(*data_cleaner.process_records(dataset))
    return X, y

def cross_validation(classifier, data_filename, X, y,
                     average, train_test_only, data_cleaner, wordlist):
    results = []
    labels = data_cleaner.labels()

    if wordlist is not None:
        relevant_features = classifier.relevant_features(X, y, labels)
        for key in relevant_features:
            with open(wordlist + key + '.txt', 'wb') as outfile:
                features = relevant_features[key]
                outfile.write('\n'.join([f.encode('utf-8') for f in features]))
    else:
        cv_results_train, cv_results_test =\
            classifier.cross_validate(X, y)
        dcname = data_cleaner.name
        print 'Classification results for file %s ...;\nusing classifier %s and ' \
              'data_cleaner %s' % (data_filename, classifier.name, dcname)
        
        if train_test_only:
            tabulate_f1_cv_summary(cv_results_train, cv_results_test, labels)
        else:
            print 'Results: Making predictions on the training set.'
            tabulate_full_cv_summary(cv_results_train, average, labels)
            print 'Results: Making predictions on the test set.'
            tabulate_full_cv_summary(cv_results_test, average, labels)


def test_specified_partition(clf, X, y, test_file, data_cleaner):
    train_metrics = []
    clf.train(X, y)
    # element 0 contains predictions made by the classifier
    # element 1 contains the metrics
    train_metrics.extend(clf.test(X, y)[1])

    test_metrics = []
    with open(test_file, 'rb') as testfile:
        test_data = pickle.load(testfile)
        X, y =  zip(*data_cleaner.process_records(test_data))
        test_metrics.extend(clf.test(X, y)[1])

    header = []
    for label in data_cleaner.labels():
        label_str = str(label)
        header.append(label_str + ': precision')
        header.append(label_str + ': recall')
        header.append(label_str + ': f1')
    header.append('kappa')

    train_record = []
    test_record = []
    for i in range(len(data_cleaner.labels())):
        for j in range(3):
            # The first index is over precision, recall, f1; the second is over labels
            train_record.append(train_metrics[j][i])
            test_record.append(test_metrics[j][i])
    train_record.append(train_metrics[3])
    test_record.append(test_metrics[3])

    print tabulate([train_record], header, tablefmt='grid')
    print tabulate([test_record], header, tablefmt='grid')


def main(args=None):
    parser = argparse.ArgumentParser(description='applies a classifier to '
                                     'train, test folds generated using '
                                     'ingest_datasets.py')
    # Required arguments
    parser.add_argument('data_file', type=str,
                        help='data file that adheres to the feature '
                             'specification laid out in feature_spec.py')
    parser.add_argument('data_cleaner', type=str,
                        help='apply a DataCleaner to the data ingested by '
                             'ingest_datasets.py; see data_cleaner_factory.py '
                             'for a list of supported cleaners.')
    parser.add_argument('classifier', type=str,
                        help='apply a particular classifier to the data; see '
                        'classifier_factory.py for a list of supported '
                        'classifiers')

    # Classification parameters
    parser.add_argument('-b', '--binary', action='store_true',
                        help='formulate the classification task '
                             'as a binay problem.')
    parser.add_argument('-p', '--penalty', type=float, default=1.0,
                        help='regularization constant')

    # Feature generation
    parser.add_argument('-c', '--chained', action='store_true',
                        help='pipe the output of other classifiers into the '
                        'input of this one; train using ground truth')
    parser.add_argument('-cg', '--chained_guess', action='store_true',
                        help='pipe the output of other classifiers into the '
                        'input of this one; train using guesses')
    parser.add_argument('-fs', '--first_sentence', type=int, default=1,
                        help='upweight each post\'s first sentence')
    parser.add_argument('-no_txt', '--no_text', action='store_true',
                        help='disregard body text when generating features')
    parser.add_argument('-np', '--noun_phrases', action='store_true',
                        help='engineer features from noun phrases')
    parser.add_argument('-t', '--token_pattern_idx', type=int,
                        default=5,
                        help='index corresponding to token_pattern in '
                             'CUSTOM_TOKEN_PATTERNS -- '
                             'see custom_token_patterns.py')
    parser.add_argument('-tf', '--tfidf', action='store_true',
                        help='apply the tfidf transformation to '
                             'lexical features')
    parser.add_argument('-txt', '--text_only', action='store_true',
                        help='derive features exclusively from body text')
    # TODO: Include an option that enables scaling and normalizing of features

    # Feature selection
    parser.add_argument('-kb', '--k_best', type=int, default=0,
                        help='use chi-square feature reduction to select '
                             'the k best features')
    parser.add_argument('-rf', '--reduce_features', action='store_true',
                        help='run RFECV to reduce features (backwards search)')

    # Output and formatting
    parser.add_argument('-avg', '--average', action='store_true',
                        help='only output the average of the per-fold metrics '
                             'computed during cross validation.')
    parser.add_argument('-f1_avg', action='store_true',
                        help='like -avg, but only print out the f1 score for '
                             'each label.')
    parser.add_argument('-tst_file', '--test_file', type=str,
                        help='test on this file, train on the data file')
    parser.add_argument('-wl', '--wordlist', type=str,
                        help='prefix of files into which word lists should be '\
                             'dumped.')

    # Persistence
    parser.add_argument('-sv', '--save', type=str,
                        help='train a classifer and persist it to disk.')

    args = parser.parse_args(args)

    if args.no_text and args.text_only:
        print 'no_text and text_only cannot both be set!'
        return

    if args.chained_guess:
        args.chained = True

    classifier = SklearnCLF(clf_name=args.classifier,
                            column=args.data_cleaner,
                            token_pattern_idx=args.token_pattern_idx,
                            text_only=args.text_only,
                            no_text=args.no_text,
                            tfidf=args.tfidf,
                            reduce_features=args.reduce_features,
                            k_best_features=args.k_best,
                            penalty=args.penalty,
                            chained=args.chained,
                            guess=args.chained_guess)
    data_cleaner = make_data_cleaner(dc=args.data_cleaner,
                                     binary=args.binary,
                                     extract_noun_phrases=args.noun_phrases,
                                     first_sentence_weight=args.first_sentence)
    X, y = examples_labels(args.data_file, data_cleaner)

    if args.save is not None:
        classifier.train(X, y)
        joblib.dump((data_cleaner, classifier), args.save)
    elif args.test_file is not None:
        test_specified_partition(classifier, X, y,
                                 args.test_file, data_cleaner)
    else: 
        cross_validation(classifier, args.data_file,
                         X, y,
                         args.average, args.f1_avg,
                         data_cleaner, args.wordlist)
    # TODO: Option to generate visuals

if __name__ == '__main__':
    main()
