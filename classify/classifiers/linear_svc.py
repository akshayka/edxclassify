'''

Linear SVC

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
import clf_util
import numpy as np
from sklearn_clf import SklearnCLF
from sklearn.svm import LinearSVC


class LinSVC(SklearnCLF):
    def __init__(self, token_pattern=r'(?u)\b\w\w+\b',
                 text_only=False,
                 no_text=False,
                 tfidf=False,
                 custom_stop_words=False,
                 C=1.0,
                 reduce_features=False,
                 k_best_features=0):
        super(LinSVC, self).__init__(token_pattern=token_pattern,
                                     text_only=text_only,
                                     no_text=no_text,
                                     tfidf=tfidf,
                                     custom_stop_words=custom_stop_words,
                                     reduce_features=reduce_features,
                                     k_best_features=k_best_features)
        self.binary_counts = True
        self.C = C
        self.normalize = True
        self.name = 'LinearSVC ' + self.name


    def train(self, X, y):
        self.make_clf(LinearSVC(C=self.C))
        self.clf.fit(X, y)


    def cross_validate(self, X, y):
        self.make_clf(LinearSVC(C=self.C))
        return clf_util.sklearn_cv(self.clf, X, y)
