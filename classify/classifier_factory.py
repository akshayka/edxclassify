'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.linear_svc import LinSVC
from classifiers.log_regression import LogRegression
from classifiers.naive_bayes import NaiveBayes
from classifiers.custom_token_patterns import CUSTOM_TOKEN_PATTERNS


#
#
#
def make_classifier(clf, token_pattern_idx=0, tfidf=False,
    custom_stop_words=False, penalty=1.0):

    if token_pattern_idx >= len(CUSTOM_TOKEN_PATTERNS):
        raise NotImplementedError('Token pattern %d not implemented'
                                  % token_pattern_idx)

    clf = clf.lower()
    if clf == 'naive_bayes':
        return NaiveBayes(token_pattern=CUSTOM_TOKEN_PATTERNS[token_pattern_idx],
                          tfidf=tfidf,
                          custom_stop_words=custom_stop_words)
    elif clf == 'logistic':
        return LogRegression(
                token_pattern=CUSTOM_TOKEN_PATTERNS[token_pattern_idx],
                tfidf=tfidf,
                custom_stop_words=custom_stop_words)
    elif clf == 'lin_svc':
        return LinSVC(
                token_pattern=CUSTOM_TOKEN_PATTERNS[token_pattern_idx],
                tfidf=tfidf,
                custom_stop_words=custom_stop_words,
                C=penalty)
    else:
        raise NotImplementedError('Classifier %s not supported.' % clf)
