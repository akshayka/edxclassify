'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.naive_bayes import NaiveBayes


#
#
#
def make_classifier(clf, token_pattern, tfidf=False,
    custom_stop_words=False):
    if clf == 'naive_bayes':
        return NaiveBayes(token_pattern=token_pattern, tfidf=tfidf,
                          custom_stop_words=custom_stop_words)
    else:
        raise NotImplementedError('Classifier %s not supported.' % key)
