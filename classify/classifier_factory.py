'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.naive_bayes import NaiveBayes
from classifiers.custom_token_patterns import CUSTOM_TOKEN_PATTERNS


#
#
#
def make_classifier(clf, token_pattern_idx=0, tfidf=False,
    custom_stop_words=False):

    if token_pattern_idx >= len(CUSTOM_TOKEN_PATTERNS):
        raise NotImplementedError('Token pattern %d not implemented'
                                  % token_pattern_idx)

    if clf == 'naive_bayes':
        return NaiveBayes(token_pattern=CUSTOM_TOKEN_PATTERNS[token_pattern_idx],
                          tfidf=tfidf,
                          custom_stop_words=custom_stop_words)
    else:
        raise NotImplementedError('Classifier %s not supported.' % clf)
