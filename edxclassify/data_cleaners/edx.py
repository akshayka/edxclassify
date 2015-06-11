from abc import ABCMeta, abstractmethod
from edxclassify.data_cleaners.abstract_data_cleaner import DataCleaner
from edxclassify.chunk_parser import ChunkParser
import edxclassify.data_cleaners.dc_util as dc_util
from nltk.corpus import conll2000
from edxclassify.feature_spec import FEATURE_COLUMNS


class Edx(DataCleaner):
    def __init__(self,
                 binary=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):

        self.columns = FEATURE_COLUMNS
        self.binary = binary
        self.extract_noun_phrases = extract_noun_phrases
        self.first_sentence_weight = first_sentence_weight
        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
        self.chunker = ChunkParser(train_sents)
        opts = ''
        if binary:
            opts = opts + 'binary '
        if extract_noun_phrases:
            opts = opts + 'extract_noun_phrases '
        if first_sentence_weight > 1:
            opts = opts + 'upweight_first_sentence '
        self.name = opts


    @abstractmethod
    def labels(self):
        pass

    def process_doc(self, document):
        document = document.lower()
        document = dc_util.collapse_numbers(document)
        document = dc_util.replace_latex(document)
        document = dc_util.replace_url(document)

        if self.extract_noun_phrases:
            document = dc_util.extract_noun_phrases(document, self.chunker)
        if self.first_sentence_weight > 1:
            document = dc_util.upweight_first_sentence(document,
                self.first_sentence_weight)
        return document

    def process_records_without_labels(self, records):
        return [ [self.process_doc(record[self.columns['text']])] +\
                  record[1:] for record in records ]

    # Idempotent
	@abstractmethod
	def process_records(self, records):
		pass
