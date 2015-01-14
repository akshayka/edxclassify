from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxQuestion(Edx):
    def __init__(self,
                 collapse_numbers=False,
                 latex=False,
                 url=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxQuestion, self).__init__(True, collapse_numbers, latex,
                                           url,
                                           extract_noun_phrases,
                                           first_sentence_weight)
        self.name = 'EdxQuestion ' + self.name

    def labels(self):
        return ['not-question', 'question']

    def process_doc(self, document):
        return super(EdxQuestion, self).process_doc(document)

    # LIST(TUPLE(LIST(features), label))
    def process_records(self, records):
        return [([self.process_doc(record[self.columns['text']])] +\
                record[1:], int(record[self.columns['question']]))\
                for record in records]
