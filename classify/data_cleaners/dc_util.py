import nltk
from nltk.corpus import conll2000
from chunk_parser import ChunkParser


def compress_likert(score, binary=False, bin_threshold=4):
    if binary:
        if score <= bin_threshold:
            return 0
        else:
            return 1

    if score <= 3:
        return 0
    elif score == 4:
        return 1
    else:
        return 2

def upweight_first_sentence(document):
    sentences = nltk.sent_tokenize(document)
    document = ' '.join(sentences + [sentences[0]])
    return document


def extract_noun_phrases(document):
    # Process the document into pos-tagged sentences
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    # Build an 'NP' chunk parser
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    cp = ChunkParser(train_sents)

    # Parse the document
    parsed_doc = [cp.parse(sentence) for sentence in sentences]
    result = []
    for tree in parsed_doc:
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                result = result + [' '.join([w for (w,p) in subtree.leaves()])]

    # For now, just upweight the noun phrases
    document = ' '.join([document] + result)
    return document
