'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.naive_bayes import NaiveBayes


def make_classifier(key):
    if key.lower() == 'naive_bayes':
        return NaiveBayes()
    elif key.lower() == 'naive_bayes_tfidf':
        return NaiveBayes(tfidf=True)
    else:
        raise NotImplementedError('Classifier %s not supported.' % key)
