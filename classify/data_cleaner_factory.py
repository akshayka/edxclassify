'''

INPUT: string --> data cleaner
OUTPUT: subclass instance of DataCleaner

'''

from data_cleaners.edx_confusion import EdxConfusion
from data_cleaners.edx_sentiment import EdxSentiment
from data_cleaners.edx_urgency import EdxUrgency


def make_data_cleaner(key):
    key = key.lower()
    if key == 'edx_confusion':
        return EdxConfusion()
    if key == 'edx_confusion_binary':
        return EdxConfusion(binary=True)
    elif key == 'edx_sentiment':
        return EdxSentiment()
    elif key == 'edx_sentiment_binary':
        return EdxSentiment(binary=True)
    elif key == 'edx_urgency':
        return EdxUrgency()
    elif key == 'edx_urgency_binary':
        return EdxUrgency(binary=True)
    else:
        raise NotImplementedError('DataCleaner %s not supported.' % key)
