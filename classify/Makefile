TARGETS=original tfidf first_sentence number custom_token1 custom_token5 custom_stopwords noun_phrase 
PRG=python harness.py
GOLD_DIR=../../Confusion\ Datasets
CLASS=medicine
CLASSIFIER=logistic
LOGFILE=$(CLASSIFIER)_$(CLASS)_log.mkd
ARGS=$(GOLD_DIR)/$(CLASS)_gold edx_confusion $(CLASSIFIER) -avg -tr_tst

define newline


endef

define OVERVIEW
% Raw Simulation Results for SVM-Modeled Text Classification
% Akshay Agrawal, Shane Leonard
% November 14, 2014 

# Overview

The following tables outline the training and testing results applied to the
`$(CLASS)_gold` dataset, using classifier `$(CLASSIFIER)` and the `EdxConfusion`
data cleaner. Each section reports the result of processing the raw document
in a different way before classifying the processed text as a bag of words.

endef

define original_DESC
# Original Bag-of-Words Approach

The raw input documents were passed directly to the classifiers with no 
preprocessing.

endef

define tfidf_DESC
# TF-IDF

`tf-idf` was used to reduce the effect of words that were common across all
documents. 

endef

define first_sentence_DESC
# Upweighting Words in the First Sentence

The first sentence of the document was duplicated to increase the importance of
words in that sentence.

endef

define numbers_DESC
# Special Tokens for Numbers

All numbers in the document were replaced by `1` so that they were counted as
the same token.

endef

define custom_token1_DESC
# Custom Tokenizer: Punctuation (.,;!?)

Punctuation (periods, commas, semicolons, exclamation marks, and question marks)
were included as individual tokens.

endef

define custom_token5_DESC
# Custom Tokenizer: Punctuation and Single-Letter Words

Punctuation and single-letter words (I, a) were included as individual tokens.

endef

define custom_stopwords_DESC
# Custom Stop Words

The word 'I' was excluded from the standard English stop words list.

endef

define noun_phrase_DESC
# Upweighting of Noun Phrases

An NLP toolkit was used to extract the noun phrases from the document. Each word
that occurred in a noun phrase was duplicated to increase the importance of those
words.

endef

define log_recipe
echo -e '$(subst $(newline),\n,${DESC})' >> $(LOGFILE) 
$(PRG) $(ARGS) $(OPTS) >> $(LOGFILE)
echo "" >> $(LOGFILE)
endef

all: init $(TARGETS)

compact: init original full full2

init:
	echo -e '$(subst $(newline),\n,${OVERVIEW})' > $(LOGFILE)

original: DESC=$(original_DESC)
original: OPTS=

tfidf: DESC=$(tfidf_DESC)
tfidf: OPTS=-tf

first_sentence: DESC=$(first_sentence_DESC)
first_sentence: OPTS=-fs

number: DESC=$(numbers_DESC)
number: OPTS=-n

custom_token1: DESC=$(custom_token1_DESC)
custom_token1: OPTS=-t 1

custom_token5: DESC=$(custom_token5_DESC)
custom_token5: OPTS=-t 5

custom_stopwords: DESC=$(custom_stopwords_DESC)
custom_stopwords: OPTS=-c

noun_phrase: DESC=$(noun_phrase_DESC)
noun_phrase: OPTS=-np

full: OPTS=-kb 700 -t 5 -c -n -p 0.5 
full: DESC=Full

full2: OPTS=-kb 700 -t 5 -c -n -p 0.5 -fs 
full2: DESC=Full2

%:
	$(log_recipe)
