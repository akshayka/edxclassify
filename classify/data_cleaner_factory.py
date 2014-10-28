'''

INPUT: string --> data cleaner
OUTPUT: subclass instance of DataCleaner

'''

from data_cleaners.edx_sentiment import EdxSentiment


def make_data_cleaner(key):
    if key.lower() == 'edx_sentiment':
        return EdxSentiment()
    else:
        raise NotImplementedError('DataCleaner %s not supported.' % key)
