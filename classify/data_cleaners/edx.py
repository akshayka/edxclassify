from abc import ABCMeta, abstractmethod
from abstract_data_cleaner import DataCleaner
from chunk_parser import ChunkParser
import dc_util
from nltk.corpus import conll2000


class Edx(DataCleaner):
    def __init__(self,
                 binary=False,
                 collapse_numbers=False,
                 latex=True,
                 extract_noun_phrases=False,
                 first_sentence_weight=1):

        self.columns = { 'text': 0,
                         'opinion': 1,
                         'question': 2,
                         'answer': 3,
                         'sentiment': 4,
                         'confusion': 5,
                         'urgency': 6,
                         'poster_identifiable': 7,
                         'course_type': 8,
                         'forum_pid': 9,
                         'course_name': 10,
                         'forum_uid': 11,
                         'date': 12,
                         'post_type': 13,
                         'anonymous': 14,
                         'anonymous_to_peers': 15,
                         'up_count': 16,
                         'comment_thread_id': 17,
                         'reads': 18 }

        self.binary = binary
        self.collapse_numbers = collapse_numbers
        self.latex = latex
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
        if self.latex:
            document = dc_util.replace_latex(document)
        if self.extract_noun_phrases:
            document = dc_util.extract_noun_phrases(document, self.chunker)
        if self.first_sentence_weight > 1:
            document = dc_util.upweight_first_sentence(document,
                self.first_sentence_weight)
        return document

	@abstractmethod
	def process_records(self, records):	
		pass
