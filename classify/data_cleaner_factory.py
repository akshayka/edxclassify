'''

INPUT: string --> data cleaner
OUTPUT: subclass instance of DataCleaner

'''

from data_cleaners.edx_confusion import EdxConfusion
from data_cleaners.edx_sentiment import EdxSentiment
from data_cleaners.edx_urgency import EdxUrgency


def make_data_cleaner(dc, binary=False,
                          collapse_numbers=False,
                          extract_noun_phrases=False,
                          upweight_first_sentence=False):
    dc = dc.lower()
    if dc == 'edx_confusion':
        return EdxConfusion(binary, collapse_numbers,
                            extract_noun_phrases,
                            upweight_first_sentence)
    elif dc == 'edx_sentiment':
        # TODO
        return EdxSentiment(binary)
    elif dc == 'edx_urgency':
        # TODO
        return EdxUrgency(binary)
    else:
        raise NotImplementedError('DataCleaner %s not supported.' % dc)
