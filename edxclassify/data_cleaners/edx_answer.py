from abstract_data_cleaner import DataCleaner
import dc_util
from edx import Edx


class EdxAnswer(Edx):
    def __init__(self,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        super(EdxAnswer, self).__init__(True,
                                        extract_noun_phrases,
                                        first_sentence_weight)
        self.name = 'EdxAnswer ' + self.name

    def labels(self):
        return ['not-answer', 'answer']

    def process_doc(self, document):
        return super(EdxAnswer, self).process_doc(document)

    # LIST(TUPLE(LIST(features), label))
    def process_records(self, records):
        return [([self.process_doc(record[self.columns['text']])] +\
                record[1:], int(record[self.columns['answer']]))\
                for record in records]
