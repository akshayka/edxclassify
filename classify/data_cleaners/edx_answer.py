from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxAnswer(Edx):
    def __init__(self,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxAnswer, self).__init__(True, collapse_numbers,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxAnswer ' + self.name

    def labels(self):
        return ['not-answer', 'answer']

    def process_doc(self, document):
        return super(EdxAnswer, self).process_doc(document)

    # The first entry in each record is the document;
    # the fourth entry in each record is the sentiment likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[3])), True, 0))
                for record in records]
