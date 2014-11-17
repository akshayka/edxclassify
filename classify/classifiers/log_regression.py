'''

Logistic Regression

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
import clf_util
import numpy as np
from sklearn_clf import SklearnCLF
from sklearn.linear_model import LogisticRegression


class LogRegression(SklearnCLF):
    def __init__(self, token_pattern=r'(?u)\b\w\w+\b', tfidf=False,
                 custom_stop_words=False,
                 reduce_features=False,
                 k_best_features=0):
        super(LogRegression, self).__init__(token_pattern,
                                            tfidf,
                                            custom_stop_words,
                                            reduce_features,
                                            k_best_features)
        self.name = 'LogisticRegression ' + self.name


    def train(self, X, y):
        self.make_clf(LogisticRegression())
        self.clf.fit(X, y)


    def cross_validate(self, X, y):
        self.make_clf(LogisticRegression())
        return clf_util.sklearn_cv(self.clf, X, y)
