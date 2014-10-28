'''

Uses naive Bayes to predict sentiment.

Training, test data format:
[ [ TEXT, SENTIMENT_SCORE] ... [TEXT, SENTIMENT_SCORE]

TODO: docs and docs and docs

'''
from abstract_classifier import Classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class NaiveBayesSentiment(Classifier):
    # Compress a likert scale with 7 values to one with 3.
    @classmethod
    def compress_likert(cls, score):
        if score <= 3:
            return 0
        elif score == 4:
            return 1
        else:
            return 2

    @classmethod
    def unpack_examples(cls, examples):
        documents, labels = zip(*examples)
        labels = map(NaiveBayesSentiment.compress_likert, labels)
        return (documents, labels)

    def labels(self):
        return (0, 1, 2)

    def train(self, training_examples):
        # Have: [ [Document, label], ... ]
        # Need: [[Features], ... ], [ label, ... ]
        documents, labels = NaiveBayesSentiment.unpack_examples(
            training_examples)
        self.sentiment_clf = make_pipeline(
            CountVectorizer(stop_words='english'),
            MultinomialNB())

        # Train!
        print 'Training ...'
        self.sentiment_clf.fit(documents, labels)

    def test(self, test_examples):
        documents, labels = NaiveBayesSentiment.unpack_examples(test_examples)

        print 'Testing ... '
        predictions = self.sentiment_clf.predict(documents)
        accuracy = np.mean(predictions == labels)
        return (zip(documents, predictions, labels), accuracy)
