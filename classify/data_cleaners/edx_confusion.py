from abstract_data_cleaner import DataCleaner
import dc_util
from nltk.corpus import conll2000
from chunk_parser import ChunkParser


class EdxConfusion(DataCleaner):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 upweight_first_sentence=False):
        self.name = 'EdxConfusion'
        self.binary = binary
        self.collapse_numbers = collapse_numbers
        self.extract_noun_phrases = extract_noun_phrases
        self.upweight_first_sentence = upweight_first_sentence
        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
        self.chunker = ChunkParser(train_sents)

    def labels(self):
        if self.binary:
            return ['not-confused', 'confused']
        else:
            return ['knowledgeable', 'neutral', 'confused']

    def process_doc(self, document):
        document = document.lower()
        if self.collapse_numbers:
            document = dc_util.collapse_numbers(document)
        if self.extract_noun_phrases:
            document = dc_util.extract_noun_phrases(document, self.chunker)
        if self.upweight_first_sentence:
            document = dc_util.upweight_first_sentence(document)
        return document

    # The first entry in each record is the document;
    # the sixth entry in each record is the likert score.
    def process_records(self, records):
        return [(self.process_doc(record[0]),
                dc_util.compress_likert(int(float(record[5])), self.binary, 4))
                for record in records]
