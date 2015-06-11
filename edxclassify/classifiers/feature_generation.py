from edxclassify.feature_spec import FEATURE_COLUMNS
from edxclassify.classifiers.word_lists import *
from edxclassify.data_cleaners.dc_util import compress_likert
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def to_int(value, aux=None):
    if value == '':
        return 0
    return int(value)

def to_float(value, aux=None):
    if value == '':
        return 0
    return 1 if float(value) > 0.94 else 0

def is_anonymous(value, aux=None):
    return 1 if value.lower() == 'true' else 0

def is_comment_thread(value, aux=None):
    return 1 if value.lower() == 'commentthread' else 0

def count_question_marks(document, aux=None):
    count = 0
    for c in document:
        if c == '?':
            count = count + 1
    return count

# TODO: How do these play with logistic regression?
# TODO: Idea -- feature for sentiment ~ 1 iff #pos > #neg
def count_negative_words(document, token_patrn):
    words = re.findall(token_patrn, document)
    count = 0
    for w in words:
        if w in NEGATIVE_WORDS:
            count = count + 1
    return count

def count_urgent_words(document, token_patrn):
    words = re.findall(token_patrn, document)
    count = 0
    for w in words:
        if w in URGENT_WORDS:
            return 1
    return 0

def count_opinion_words(document, token_patrn):
    words = re.findall(token_patrn, document)
    count = 0
    for w in words:
        if w in OPINION_WORDS:
            count = count + 1
    return count

def count_nouns(document, aux=None):
    tagged_words = []
    for s in sent_tokenize(document.decode('utf-8')):
        tagged_words.extend(nltk.pos_tag(word_tokenize(s)))
    count = 0
    for word, tag in tagged_words:
        if tag == 'NN':
            count = count + 1
    return count

# TODO: We might want to discretize the grades and number of attempts
class FeatureExtractor:
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        idx = FEATURE_COLUMNS[self.feature_name]
        return [row[idx] for row in X]


class FeatureCurator:
    def __init__(self, feature_name, curate_function, aux=None):
        self.feature_name = feature_name
        self.curate = curate_function
        self.aux=aux

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [{self.feature_name + ' feature': self.curate(value, self.aux)}
                for value in X]

    def fit_transform(self, X, y=None):
        return self.transform(X)

class ChainedClassifier:
    def __init__(self, clf, column, guess):
        self.clf = clf
        self.column = column
        self.y_chain = None
        self.guess = guess

    def fit(self, X, y=None):
        # Note that the extracted values will be in 
        # [0, 2] for non-binary variables (confusion,
        # sentiment, urgency), {0, 1} otherwise.
        if self.column == 'confusion' or\
            self.column == 'sentiment' or\
            self.column == 'urgency':
            self.y_chain = [compress_likert(
                                record[FEATURE_COLUMNS[self.column]],
                                    binary=False)\
                            for record in X]
        else:
            self.y_chain = [int(record[FEATURE_COLUMNS[self.column]])\
                            for record in X]

        self.clf.train(X, self.y_chain)

    def transform(self, X, y=None):
        if self.y_chain is not None and not self.guess:
            predictions = self.y_chain
            # This is critical -- it ensures
            # that we don't use the gold set values when
            # predicting.
            self.y_chain = None
        else:
            predictions = self.clf.test(X)
        return [{self.column + ' prediction': value} for value in predictions]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
