from abc import ABCMeta, abstractmethod
from abstract_data_cleaner import DataCleaner
from chunk_parser import ChunkParser
import dc_util
from nltk.corpus import conll2000


class Edx(DataCleaner):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):
        self.binary = binary
        self.collapse_numbers = collapse_numbers
        self.extract_noun_phrases = extract_noun_phrases
        self.first_sentence_weight = first_sentence_weight
        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
        self.chunker = ChunkParser(train_sents)
        opts = ''
        if binary:
            opts = opts + 'binary '
        if collapse_numbers:
            opts = opts + 'collapse_numbers '
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
        if self.collapse_numbers:
            document = dc_util.collapse_numbers(document)
        if self.extract_noun_phrases:
            document = dc_util.extract_noun_phrases(document, self.chunker)
        if self.first_sentence_weight > 1:
            document = dc_util.upweight_first_sentence(document,
                self.first_sentence_weight)
        return document

	@abstractmethod
	def process_records(self, records):	
		pass
