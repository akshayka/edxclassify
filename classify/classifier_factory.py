'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.naive_bayes_sentiment import NaiveBayesSentiment


def make_classifier(key):
    if key.lower() == 'naive_bayes_sentiment':
        return NaiveBayesSentiment()
    else:
        raise NotImplementedError('Classifier %s not supported.' % key)
