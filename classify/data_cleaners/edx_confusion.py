from abstract_data_cleaner import DataCleaner
import dc_util


class EdxConfusion(DataCleaner):
    def __init__(self, binary=False):
        self.name = 'EdxConfusion'
        self.binary = binary

    def labels(self):
        if self.binary:
            return ['not-confused', 'confused']
        else:
            return ['knowledgeable', 'neutral', 'confused']

    def process_doc(self, document,
                    collapse_numbers=False,
                    extract_noun_phrases=False,
                    upweight_first_sentence=False):
        document = document.lower()
        # document = dc_util.extract_noun_phrases(document)
        # document = dc_util.upweight_first_sentence(document)
        return document

    # The first entry in each record is the document;
    # the sixth entry in each record is the likert score.
    def process_records(self, records,
                        collapse_numbers=False,
                        extract_noun_phrases=False,
                        upweight_first_sentence=False):
        return [(self.process_doc(record[0],
                                  collapse_numbers,
                                  extract_noun_phrases,
                                  upweight_first_sentence),
                dc_util.compress_likert(int(float(record[5])), self.binary, 4))
                for record in records]
