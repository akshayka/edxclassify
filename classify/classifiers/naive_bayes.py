'''

Uses naive Bayes to predict likert value.

Maps scale 1-7 to -1, 0, 1

Training, test data format:
[ [ TEXT, LIKERT_SCORE] ... [TEXT, LIKERT_SCORE]

TODO: docs and docs and docs

'''
from abstract_classifier import Classifier
import clf_util
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class NaiveBayes(Classifier):
    def __init__(self, tfidf=False):
        self.use_tfidf=tfidf
        self.name = 'NaiveBayes'


    def make_clf(self):
        if self.use_tfidf:
            self.clf = make_pipeline(
                CountVectorizer(stop_words='english'),
                TfidfTransformer(),
                MultinomialNB())
        else:
            self.clf = make_pipeline(
                CountVectorizer(stop_words='english'),
                MultinomialNB())
    
    def train(self, training_examples):
        documents, labels = zip(*examples)

        self.make_clf()
        self.clf.fit(documents, labels)

    def test(self, test_examples):
        documents, labels = zip(*test_examples)
        predictions = self.clf.predict(documents)
        accuracy = np.mean(predictions == labels)
        return (zip(documents, predictions, labels), accuracy)

    def cross_validate(self, examples):
        documents, labels = zip(*examples)

        self.make_clf()
        return clf_util.sklearn_cv(self.clf, documents, labels)
