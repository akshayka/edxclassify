'''

Uses naive Bayes to predict likert value.

Maps scale 1-7 to -1, 0, 1

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
from abstract_classifier import Classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class NaiveBayesLikert(Classifier):
    def __init__(self, tfidf=False):
        self.use_tfidf=tfidf
        self.name = 'NaiveBayesLikert'

    # Compress a likert scale with 7 values to one with 3.
    @classmethod
    def compress_likert(cls, score):
        if score <= 3:
            return -1
        elif score == 4:
            return 0
        else:
            return 1

    @classmethod
    def unpack_examples(cls, examples):
        documents, labels = zip(*examples)
        labels = map(NaiveBayesLikert.compress_likert, labels)
        return (documents, labels)

    def labels(self):
        return (-1, 0, 1)

    def train(self, training_examples):
        # Have: [ [Document, label], ... ]
        # Need: [[Features], ... ], [ label, ... ]
        documents, labels = NaiveBayesLikert.unpack_examples(
            training_examples)
        self.label_counts = [0, 0, 0]
        self.label_counts[0] = labels.count(-1)
        self.label_counts[1] = labels.count(0)
        self.label_counts[2] = labels.count(1)
        if self.use_tfidf:
            self.likert_clf = make_pipeline(
                CountVectorizer(stop_words='english'),
                TfidfTransformer(),
                MultinomialNB())
        else:
            self.likert_clf = make_pipeline(
                CountVectorizer(stop_words='english'),
                MultinomialNB())
        self.likert_clf.fit(documents, labels)

    def test(self, test_examples):
        documents, labels = NaiveBayesLikert.unpack_examples(test_examples)
        self.label_counts[0] = self.label_counts[0] + labels.count(-1)
        self.label_counts[1] = self.label_counts[1] + labels.count(0)
        self.label_counts[2] = self.label_counts[2] + labels.count(1)
        predictions = self.likert_clf.predict(documents)
        accuracy = np.mean(predictions == labels)
        return (zip(documents, predictions, labels), accuracy)
