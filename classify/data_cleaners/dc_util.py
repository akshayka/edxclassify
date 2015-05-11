import nltk
import re


def compress_likert(score, binary=False, bin_threshold=4):
    score = float(score)
    if binary:
        if score <= bin_threshold:
            return 0
        else:
            return 1

    if score < 4:
        return 0
    elif score == 4:
        return 1
    else:
        return 2

def upweight_first_sentence(document, weight):
    sentences = nltk.sent_tokenize(document.decode("utf8"))
    document = ' '.join([sentences[0]] * weight + sentences[1:])
    return document.encode("utf8")


def extract_noun_phrases(document, cp):
    # Process the document into pos-tagged sentences
    document = document.decode("utf8")
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    # Parse the document
    parsed_doc = [cp.parse(sentence) for sentence in sentences]
    result = []
    for tree in parsed_doc:
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                result = result + [' '.join([w for (w,p) in subtree.leaves()])]

    # For now, just upweight the noun phrases
    document = ' '.join([document] + result)
    return document.encode("utf8")


def collapse_numbers(document):
    return re.sub(r'\b[0-9]+\b', '1', document)


def replace_latex(document):
    return re.sub(r'\${1,2}[^\$]*\${1,2}', 'clf_latex_eqn_tok', document)

def replace_url(document):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'clf_url_tok', document)

