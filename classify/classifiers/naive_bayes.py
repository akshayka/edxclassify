'''

Multinomial Naive Bayes

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
import clf_util
import numpy as np
from sklearn_clf import SklearnCLF
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(SklearnCLF):
    def __init__(self, token_pattern=r'(?u)\b\w\w+\b',
                 text_only=False,
                 no_text=False,
                 tfidf=False,
                 reduce_features=False,
                 k_best_features=0):
        super(NaiveBayes, self).__init__(token_pattern=token_pattern,
                                         text_only=text_only,
                                         no_text=no_text,
                                         tfidf=tfidf,
                                         reduce_features=reduce_features,
                                         k_best_features=k_best_features)
        self.name = 'NaiveBayes ' + self.name

    
    def train(self, X, y):
        self.make_clf(MultinomialNB())
        self.clf.fit(X, y)


    def cross_validate(self, X, y, labels):
        self.make_clf(MultinomialNB())
        return clf_util.sklearn_cv(self.clf, X, y, labels)
