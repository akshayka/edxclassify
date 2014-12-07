import numpy as np
import re
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from custom_stop_words import ENGLISH_STOP_WORDS
from ..feature_spec import FEATURE_COLUMNS


class TextToDictTransformer:
    def __init__(self, token_pattern,
                 stop_words='english',
                 binary_counts=False):
        self.token_pattern = token_pattern
        self.binary_counts = binary_counts

        if stop_words == 'english':
            stop_words = ENGLISH_STOP_WORDS
        self.stop_words = stop_words

    def fit(self, documents, y=None):
        self.keyset = set(
            [token for document in documents
             for token in re.findall(self.token_pattern, document)]
        ) - self.stop_words

    def transform(self, documents, y=None):
        countDict = dict.fromkeys(self.keyset, 0)
        for document in documents:
            for token in re.findall(self.token_pattern, document):
                if token in countDict:
                    if self.binary_counts:
                        countDict[token] = 1
                    else:
                        countDict[token] += 1
        return countDict

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class FeatureAggregator:
    def __init__(self, token_pattern,
                 stop_words='english',
                 binary_counts=False):
        self.textTransformer = TextToDictTransformer(token_pattern,
                                                     stop_words,
                                                     binary_counts)

    def fit(self, X, y=None):
        '''
        Aggregate each of the features on a per-row basis, so that transform
        can return a list of dictionaries, one per training example
        '''
        for training_example in X:
            text_idx = FEATURE_COLUMNS['text']
            self.textTransformer.fit_transform(training_example[text_idx])





def sklearn_cv(clf, examples, labels):
    X, y = np.array(examples), np.array(labels)
    skf = StratifiedKFold(labels, n_folds=10)
    precision_train = []
    recall_train = []
    f1_train = []
    precision_test = []
    recall_test = []
    f1_test = []
    for train_indices, test_indices in skf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Predict on the test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision_test.append(metrics.precision_score(y_test, y_pred,
            average=None))
        recall_test.append(metrics.recall_score(y_test, y_pred, average=None))
        f1_test.append(metrics.f1_score(y_test, y_pred, average=None))

        # Predict on the training set
        y_pred = clf.predict(X_train)
        precision_train.append(metrics.precision_score(y_train, y_pred,
            average=None))
        recall_train.append(metrics.recall_score(y_train, y_pred, average=None))
        f1_train.append(metrics.f1_score(y_train, y_pred, average=None))
    return [precision_train, recall_train, f1_train], [precision_test, recall_test, f1_test]
