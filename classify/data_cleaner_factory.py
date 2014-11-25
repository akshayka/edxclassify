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


supported_data_cleaners='edx_confusion\nedx_sentiment\nedx_urgency\n' \
                        'edx_opinion\nedx_question\nedx_answer'
def make_data_cleaner(dc, binary=False,
                          collapse_numbers=False,
                          latex=True,
                          extract_noun_phrases=False,
                          first_sentence_weight=1):
    dc = dc.lower()
    if dc == 'edx_confusion':
        return EdxConfusion(binary=binary,
                            collapse_numbers=collapse_numbers,
                            latex=latex,
                            extract_noun_phrases=extract_noun_phrases,
                            first_sentence_weight=first_sentence_weight)
    elif dc == 'edx_sentiment':
        return EdxSentiment(binary=binary,
                            collapse_numbers=collapse_numbers,
                            latex=latex,
                            extract_noun_phrases=extract_noun_phrases,
                            first_sentence_weight=first_sentence_weight)
    elif dc == 'edx_urgency':
        return EdxUrgency(binary, collapse_numbers,
                          latex=latex,
                          extract_noun_phrases=extract_noun_phrases,
                          first_sentence_weight=first_sentence_weight)
    elif dc == 'edx_opinion':
        return EdxOpinion(collapse_numbers=collapse_numbers,
                          latex=latex,
                          extract_noun_phrases=extract_noun_phrases,
                          first_sentence_weight=first_sentence_weight)
    elif dc == 'edx_question':
        return EdxQuestion(collapse_numbers=collapse_numbers,
                          latex=latex,
                          extract_noun_phrases=extract_noun_phrases,
                          first_sentence_weight=first_sentence_weight)
    elif dc == 'edx_answer':
        return EdxAnswer(collapse_numbers,
                         latex=latex,
                         extract_noun_phrases=extract_noun_phrases,
                         first_sentence_weight=first_sentence_weight)
    else:
        raise NotImplementedError('DataCleaner %s not supported; choose from:'
                                  '%s' (dc, supported_data_cleaners))
