'''

Generic sklearn classifier

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
from abc import ABCMeta, abstractmethod
from abstract_classifier import Classifier
from custom_stop_words import CUSTOM_STOP_WORDS
import classify.classifiers.clf_util as clf_util
from classify.classifiers.feature_aggregator import *
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import FeatureUnion, Pipeline, make_union, make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


class SklearnCLF(Classifier):
    def __init__(self, token_pattern=r'(?u)\b\w\w+\b',
                 text_only=False,
                 no_text=False,
                 tfidf=False,
                 custom_stop_words=False,
                 reduce_features=False,
                 k_best_features=0,
                 use_dict_vectorizer=True):
        self.token_pattern = token_pattern
        self.text_only = text_only
        self.no_text = no_text
        self.tfidf = tfidf
        self.custom_stop_words = custom_stop_words
        self.reduce_features = reduce_features
        self.k_best_features = k_best_features
        self.binary_counts = False
        self.normalize = False

        opts = 'token:' + token_pattern + ' '
        if self.text_only:
            opts = opts + 'text_only '
        if self.tfidf:
            opts = opts + 'tfidf '
        if self.custom_stop_words:
            opts = opts + 'custom_stop_words '
        if self.reduce_features:
            opts = opts + 'reduce_features '
        if self.k_best_features:
            opts = opts + 'k_best_features '
        self.name = opts


    def make_clf(self, clf):
        stop_words='english'
        if self.custom_stop_words:
           stop_words=CUSTOM_STOP_WORDS

        counter = None
        if self.tfidf:
            counter = TfidfVectorizer(token_pattern=self.token_pattern,
                                      stop_words=stop_words,
                                      binary=self.binary_counts)
        else:
            counter = CountVectorizer(token_pattern=self.token_pattern,
                                      stop_words=stop_words,
                                      binary=self.binary_counts)
        features = []
        if not self.no_text:
            features = [
                ('text_document', Pipeline([
                    ('selector', FeatureExtractor('text')),
                    ('count', counter),
                ])),
            ]

        if not self.text_only:
            features = \
                features + [
                    ('up_counts', Pipeline([
                        ('selector', FeatureExtractor('up_count')),
                        ('curate', FeatureCurator('up_count', clf_util.to_int)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('anonymous', Pipeline([
                        ('selector', FeatureExtractor('anonymous')),
                        ('curate', FeatureCurator('anonymous',
                                    clf_util.is_anonymous)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('anonymous_to_peers', Pipeline([
                        ('selector', FeatureExtractor('anonymous_to_peers')),
                        ('curate', FeatureCurator('anonymous_to_peers',
                                    clf_util.is_anonymous)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('post_type', Pipeline([
                        ('selector', FeatureExtractor('post_type')),
                        ('curate', FeatureCurator('post_type',
                                    clf_util.is_comment_thread)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('reads', Pipeline([
                        ('selector', FeatureExtractor('reads')),
                        ('curate', FeatureCurator('reads', clf_util.to_int)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('cum_attempts', Pipeline([
                        ('selector', FeatureExtractor('cum_attempts')),
                        ('curate', FeatureCurator('cum_attemtps', clf_util.to_int)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('cum_grade', Pipeline([
                        ('selector', FeatureExtractor('cum_grade')),
                        ('curate', FeatureCurator('cum_grade', clf_util.to_float)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                ]
        pipeline = [FeatureUnion(features)]
        #if self.normalize:
        #    pipeline = pipeline + [StandardScaler(with_mean=False)]
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
        p = metrics.precision_score(y, predictions, average=None)
        r = metrics.recall_score(y, predictions, average=None)
        f = metrics.f1_score(y, predictions, average=None)
        return (predictions, [p, r, f])


    @abstractmethod
    def cross_validate(self, X, y, labels):
        pass
