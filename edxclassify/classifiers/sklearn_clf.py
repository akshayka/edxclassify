'''

Generic sklearn classifier

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
from edxclassify.classifiers.abstract_classifier import Classifier
from edxclassify.classifiers.clf_util import *
from edxclassify.classifiers.custom_token_patterns import CUSTOM_TOKEN_PATTERNS
from edxclassify.classifiers.feature_generation import *

import math
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline, make_union, make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
import skll

from edxclassify.classifiers.word_lists import *


class SklearnCLF(Classifier):
    supported_classifiers = 'naive_bayes\nlogistic\nlin_svc'

    def __init__(self, clf_name='',
                 column='', 
                 token_pattern_idx=5,
                 text_only=False,
                 no_text=False,
                 tfidf=False,
                 reduce_features=False,
                 k_best_features=0,
                 penalty=1.0,
                 chained=False,
                 guess=False):
        
        if token_pattern_idx >= len(CUSTOM_TOKEN_PATTERNS):
            raise NotImplementedError('Token pattern %d not implemented'
                                      % token_pattern_idx)
        self.clf_name = clf_name.lower()
        self.column = column
        self.token_pattern = CUSTOM_TOKEN_PATTERNS[token_pattern_idx]
        self.token_pattern_idx = token_pattern_idx
        self.text_only = text_only
        self.no_text = no_text
        self.tfidf = tfidf
        self.reduce_features = reduce_features
        self.k_best_features = k_best_features
        self.penalty = penalty
        self.chained = chained
        self.guess = guess

        if self.chained:
            self.chain_args = [self.clf_name, '', self.token_pattern_idx,
                              self.text_only, self.no_text, self.tfidf,
                              self.reduce_features, self.k_best_features,
                              self.penalty, False, False]

        # TODO: Understand the mathematical implications for
        # each of these options
        self.binary_counts = False
        self.normalize = False

        opts = 'token:' + self.token_pattern + ' '
        if self.text_only:
            opts = opts + 'text_only '
        if self.tfidf:
            opts = opts + 'tfidf '
        if self.reduce_features:
            opts = opts + 'reduce_features '
        if self.k_best_features:
            opts = opts + 'k_best_features '
        self.name = self.clf_name + opts

        if self.clf_name == 'naive_bayes':
            self._make_clf(MultinomialNB())
        elif self.clf_name == 'logistic':
            self.binary_counts = True
            self._make_clf(LogisticRegression(C=self.penalty, penalty='l2'))
        elif self.clf_name == 'lin_svc':
            self.binary_counts = True
            self.normalize = True
            self._make_clf(LinearSVC(C=self.penalty))
        else:
            raise NotImplementedError('Classifier %s not supported; choose from:\n'
                                      '%s' % (self.clf_name, self.supported_classifiers))

    def _make_chained(self, column):
        if self.column != column:
            chain = ChainedClassifier(
                clf=SklearnCLF(*self.chain_args),
                column=column,
                guess=self.guess,
            )
            return [(column, Pipeline([
                        (column + '_guess', chain),
                        ('dict_vect', DictVectorizer()),
                    ]))]
        else:
            return []

    def _make_clf(self, clf):
        counter = None
        if self.tfidf:
            counter = TfidfVectorizer(token_pattern=self.token_pattern,
                                      stop_words=CUSTOM_STOP_WORDS,
                                      binary=self.binary_counts)
        else:
            counter = CountVectorizer(token_pattern=self.token_pattern,
                                      stop_words=CUSTOM_STOP_WORDS,
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
                        ('curate', FeatureCurator('up_count', to_int)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('anonymous', Pipeline([
                        ('selector', FeatureExtractor('anonymous')),
                        ('curate', FeatureCurator('anonymous',
                                    is_anonymous)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('anonymous_to_peers', Pipeline([
                        ('selector', FeatureExtractor('anonymous_to_peers')),
                        ('curate', FeatureCurator('anonymous_to_peers',
                                    is_anonymous)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('post_type', Pipeline([
                        ('selector', FeatureExtractor('post_type')),
                        ('curate', FeatureCurator('post_type',
                                    is_comment_thread)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    ('reads', Pipeline([
                        ('selector', FeatureExtractor('reads')),
                        ('curate', FeatureCurator('reads', to_int)),
                        ('dict_vect', DictVectorizer()),
                    ])),
                    # TODO: Use cum_grade?
                    #       (curator --> to_float)
                ]

            # Affect specific features
            if self.column == 'sentiment':
                features =\
                     features + [
                        ('negative_words', Pipeline([
                            ('selector', FeatureExtractor('text')),
                            ('curate', FeatureCurator('negative_count',
                                count_negative_words,
                                CUSTOM_TOKEN_PATTERNS[self.token_pattern_idx])),
                            ('dict_vect', DictVectorizer()),
                        ])),
                    ]
            # TODO: Is this discrimination advisable?
            if self.column == 'confusion' or\
                self.column == 'urgency' or\
                self.column == 'question' or\
                self.column == 'answer':
                features =\
                    features + [
                        ('#?', Pipeline([
                            ('selector', FeatureExtractor('text')),
                            ('curate', FeatureCurator('question_mark_count',
                                count_question_marks)),
                            ('dict_vect', DictVectorizer()),
                        ])),
                    ]
             # TODO: Count urgent words?
            if self.column == 'urgency':
                features =\
                    features + [
                        ('#urgent', Pipeline([
                            ('selector', FeatureExtractor('text')),
                            ('curate', FeatureCurator('urgent_count',
                                count_urgent_words,
                                CUSTOM_TOKEN_PATTERNS[self.token_pattern_idx])),
                            ('dict_vect', DictVectorizer()),
                        ])),
                    ]
                
        # TODO: More intelligent selection of chains based on correlations
        if self.chained:
            # NB: Features only grows if self.column != <param-for-make_chained>
            features = features + self._make_chained('question')
            features = features + self._make_chained('answer')
            features = features + self._make_chained('opinion')
            features = features + self._make_chained('sentiment')
            features = features + self._make_chained('urgency')
            features = features + self._make_chained('confusion')
        pipeline = [FeatureUnion(features)]

        #if self.normalize:
        #    pipeline = pipeline + [StandardScaler(with_mean=False)]
        if self.k_best_features > 0:
            pipeline = pipeline + [SelectKBest(chi2, k=self.k_best_features)]
        if self.reduce_features:
            pipeline = pipeline + [RFE(clf)]
        else:
            pipeline = pipeline + [clf]
        self.clf = make_pipeline(*pipeline)


    def train(self, X, y):
        self.clf.fit(X, y)


    def test(self, X, y=None):
        predictions = self.clf.predict(X)
        if y is not None:
            p = metrics.precision_score(y, predictions, average=None)
            r = metrics.recall_score(y, predictions, average=None)
            f = metrics.f1_score(y, predictions, average=None)
            K = skll.metrics.kappa(y, predictions)
            return (predictions, [p, r, f, K])
        else:
            return predictions

    def cross_validate(self, X, y):
        return sklearn_cv(self.clf, X, y)


    """
    Retrieve most relevant features
    parameters:
    ----------
    X          - a list of feature vectors
    y          - a list of labels, with y[i] the label for X[i]
    labels     - a list of string identifiers, one for each label
    num_top    - number of features to retrieve

    returns:
    -------
    relevant_features: A dictionary describing informative features.
                    In particular, if clf is a multiclass classifier,
                    relevant_features maps each string label to a list
                    containing the 600 features weighted most highly by the
                    classifier's decision function(s).
                    If clf is a binary class, then a dictionary mapping
                    the word 'informative' to the list of num_top features weighted
                    most highly by the classifier's decision function.
    """
    def relevant_features(self, X, y, labels, num_top=600):
        relevant_features = {}
        self.clf.fit(X, y)

        # A bit of a hack -- we index into the pipeline in order to
        # retrieve the actual estimator.
        classifier = self.clf.steps[-1][-1]
        feature_names = extract_feature_names(self.clf.steps[0][-1])

        # Another hack -- if it's a binary problem, then len(coef_) == 1
        if len(labels) == 2:
            labels = ['informative']

        if feature_names is not None and hasattr(classifier, 'coef_'):
            for i, label in enumerate(labels):
                top = np.argsort(classifier.coef_[i])[::-1][:num_top]
                coefs = classifier.coef_[i][top]
                for t, c in zip(top, coefs):
                    relevant_features.setdefault(label,
                        []).append(feature_names[t] + ' : ' + ('%.2f' %
                                    math.exp(c)))

        return relevant_features
