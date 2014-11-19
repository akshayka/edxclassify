'''

INPUT: string --> data cleaner
OUTPUT: subclass instance of DataCleaner

'''

from data_cleaners.edx_confusion import EdxConfusion
from data_cleaners.edx_sentiment import EdxSentiment
from data_cleaners.edx_urgency import EdxUrgency
from data_cleaners.edx_answer import EdxAnswer
from data_cleaners.edx_opinion import EdxOpinion
from data_cleaners.edx_question import EdxQuestion


def make_data_cleaner(dc, binary=False,
                          collapse_numbers=False,
                          extract_noun_phrases=False,
                          first_sentence_weight=1):
    dc = dc.lower()
    if dc == 'edx_confusion':
        return EdxConfusion(binary, collapse_numbers,
                            extract_noun_phrases,
                            first_sentence_weight)
    elif dc == 'edx_sentiment':
        return EdxSentiment(binary, collapse_numbers,
                            extract_noun_phrases,
                            first_sentence_weight)
    elif dc == 'edx_urgency':
        return EdxUrgency(binary, collapse_numbers,
                          extract_noun_phrases,
                          first_sentence_weight)
    elif dc == 'edx_opinion':
        return EdxOpinion(collapse_numbers,
                          extract_noun_phrases,
                          first_sentence_weight)
    elif dc == 'edx_question':
        return EdxQuestion(collapse_numbers,
                          extract_noun_phrases,
                          first_sentence_weight)
    elif dc == 'edx_answer':
        return EdxAnswer(collapse_numbers,
                          extract_noun_phrases,
                          first_sentence_weight)
    else:
        raise NotImplementedError('DataCleaner %s not supported.' % dc)
