'''

INPUT: string --> classifier
OUTPUT: subclass instance of Classifier

'''

from classifiers.naive_bayes_likert import NaiveBayesLikert


def make_classifier(key):
    if key.lower() == 'naive_bayes_likert':
        return NaiveBayesLikert()
    elif key.lower() == 'naive_bayes_likert_tfidf':
        return NaiveBayesLikert(tfidf=True)
    else:
        raise NotImplementedError('Classifier %s not supported.' % key)
