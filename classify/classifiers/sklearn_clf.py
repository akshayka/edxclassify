'''

Generic sklearn classifier

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
from abc import ABCMeta, abstractmethod
from abstract_classifier import Classifier
from custom_stop_words import CUSTOM_STOP_WORDS
import clf_util
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SklearnCLF(Classifier):
    def __init__(self, token_pattern=r'(?u)\b\w\w+\b', tfidf=False,
                 custom_stop_words=False,
                 reduce_features=False,
                 k_best_features=0):
        self.token_pattern = token_pattern
        self.tfidf = tfidf
        self.custom_stop_words = custom_stop_words
        self.reduce_features = reduce_features
        self.k_best_features = k_best_features
        self.binary_counts = False

        opts = 'token:' + token_pattern + ' '
        if tfidf:
            opts = opts + 'tfidf '
        if custom_stop_words:
            opts = opts + 'custom_stop_words '
        if reduce_features:
            opts = opts + 'reduce_features '
        if k_best_features:
            opts = opts + 'k_best_features '
        self.name = opts


    def make_clf(self, clf):
        stop_words='english'
        if self.custom_stop_words:
           stop_words=CUSTOM_STOP_WORDS 

        if self.binary_counts:
            pipeline = [CountVectorizer(token_pattern=self.token_pattern,
                                        stop_words=stop_words, binary=True)]
        else:
            pipeline = [CountVectorizer(token_pattern=self.token_pattern,
                                        stop_words=stop_words)]

        if self.tfidf:
            pipeline = pipeline + [TfidfTransformer()]
        if self.scale:
            pipeline = pipeline + [StandardScaler(with_mean=False)]
        if self.k_best_features > 0:
            pipeline = pipeline + [SelectKBest(chi2, k=self.k_best_features)]
        if self.reduce_features:
            pipeline = pipeline + [RFECV(clf, step=1, cv=2)]
        else:
            pipeline = pipeline + [clf]
        self.clf = make_pipeline(*pipeline)
    
    
    @abstractmethod
    def train(self, X, y):
        pass


    def test(self, X, y):
        predictions = self.clf.predict(X)
        accuracy = np.mean(predictions == y)
        return (zip(X, predictions, y), accuracy)


    @abstractmethod
    def cross_validate(self, X, y):
        pass
