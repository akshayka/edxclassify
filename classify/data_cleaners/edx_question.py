from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxQuestion(Edx):
    def __init__(self,
                 collapse_numbers=False,
                 latex=True,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxQuestion, self).__init__(True, collapse_numbers, latex,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxQuestion ' + self.name

    def labels(self):
        return ['not-question', 'question']

    def process_doc(self, document):
        return super(EdxQuestion, self).process_doc(document)

    # The first entry in each record is the document;
    # the third entry in each record is the sentiment likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[2])), True, 0))
                for record in records]
