import nltk
from nltk.corpus import conll2000
from chunk_parser import ChunkParser


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


def test():
    document = """I believe that there are many types of
    love in the world. However, my good friend Akshay disagrees with me!
    My parents jumped over the couch, but thankfully no one was hurt. Do you want
    to go to Elitch Gardens with my sister and me?"""

    print("Upweighting first sentence:")
    upweight_first_sentence(document)

    print("Upweighting noun phrases:")
    edoc = extract_noun_phrases(document)

    print("Doing both (must upweight noun phrases first):")
    upweight_first_sentence(edoc)


if __name__ == '__main__':
    test()
