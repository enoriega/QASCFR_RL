""" This file has NLP utilities, such as tokenization, embedding lookups, etc"""
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union, Mapping, Iterable, Optional, cast, FrozenSet, List, Type, Set

import numpy as np
import spacy
from gensim.models import KeyedVectors
from nltk import PorterStemmer, SnowballStemmer, word_tokenize
from numpy.typing import ArrayLike
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Doc

import utils
from machine_reading.ir.frequency_counter import FrequencyCounter


@lru_cache(maxsize=5)
def load_glove(path: str) -> KeyedVectors:
    return cast(KeyedVectors, KeyedVectors.load(path))


@lru_cache(maxsize=1)
def load_stop_words() -> FrozenSet[str]:
    """ Return a set of stop words. Cache the result for efficiency """
    stop_words = set(STOP_WORDS)

    # # Remove some words that could be useful
    # for w in ['do', 'it', 'yourself',
    #           'the', 'who', 'front',
    #           'five', 'ten', 'rather',
    #           'be', 'next']:
    #     stop_words.remove(w)

    # Add punctuation marks as stop words
    for w in ['&', "'", '(', ')', ',', '-', '.',
              '/', ':', '–', '!', '"', '#', '$',
              '%', "''", '*', '+', "..", "...",
              ";", '[', ']', '<', '>', '=', '?',
              '@', '`']:
        stop_words.add(w)

    stop_words.add('the')

    return frozenset(stop_words)


@lru_cache(maxsize=1)
def stemmed_stop_words() -> FrozenSet[str]:
    """ Return the same stop words as above, stemmed with Snowball """
    stemmer = SnowballStemmer(language="english")
    return frozenset(stemmer.stem(w) for w in load_stop_words())


# @lru_cache(maxsize=10000)
def preprocess(text: Union[str, Iterable[str]], nlp: Language, stem: Optional[bool] = False) -> Union[Sequence[str], Iterable[Sequence[str]]]:
    """ Splits the input string into a sequence of tokens. Everything is done lazily """

    return_str = False
    # Make sure we do the "batched" version even if its a single input text
    if type(text) == str:
        return_str = True
        text = [text]

    # Normalize and tokenize. Replace the underscores of the entities for white spaces
    text = (t.lower().strip().replace("_", " ")\
            .replace("-", " ")\
            .replace("?", "")\
            .replace("!", "")\
            .replace(".", "")\
            .replace(",", "")\
            .replace(":", "")\
            .replace(";", "") for t in text)
            # for t in text)

    # # Pipe the text for efficient tokenization. Disable the tagger and parser for now
    # docs = nlp.pipe(text, disable=['parser'])
    #
    # # Fetch the stop words
    # stop_words = load_stop_words()
    #
    # # Do some stemming with this
    # stemmer = SnowballStemmer(language='english')
    #
    # # Return a generator that will return the tokens as long as they're not a stop word
    # res = [[stemmer.stem(w) if stem else w for w in (token.text for token in doc) if w not in stop_words] for doc in docs]

    # Pipe the text for efficient tokenization. Disable the tagger and parser for now
    docs =[word_tokenize(t) for t in text]

    # Fetch the stop words
    stop_words = load_stop_words()

    # Do some stemming with this
    stemmer = SnowballStemmer(language='english')

    # Return a generator that will return the tokens as long as they're not a stop word
    res = [[stemmer.stem(w) if stem else w for w in doc if w not in stop_words] for doc in
           docs]
    if return_str:
        return res[0]
    else:
        return res


def average_embedding(tokens: Sequence[str],
                      nlp: Language) -> np.ndarray:
    """ Returns the average embedding of an entity based on its tokens """
    # Build a document from the pre-tokenized input
    spaces = [True] * (len(tokens) - 1) + [False]
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

    # Delegate the vector representation to spaCy. It will compute the average itself
    ret = doc.vector

    return ret


def air_align(anchor_term: str, phrase_terms: Union[ArrayLike, List[str]], model: KeyedVectors) -> float:
    """
    Max-pooled cosine similarity from the cartesian product of the arguments.
    See AIR, Sec 3.1
    """

    anchor = model[anchor_term]

    # In case we don't have the matrix of embeddings yet
    if type(phrase_terms) == list:
        phrase_terms = model[phrase_terms]

    similarities = cast(ArrayLike, KeyedVectors.cosine_similarities(anchor, phrase_terms))

    pooled = similarities.max()

    return pooled


def idf(term: str) -> float:
    frequencies = load_term_frequencies()
    return frequencies.idf.get(term, 0.)


@lru_cache(maxsize=1)
def load_term_frequencies() -> FrequencyCounter:
    """ Loads lazily the pre-computed FrequencyCounter """
    config = utils.read_config()
    path = Path(config['files']['frequencies_path'])
    with path.open('rb') as f:
        freqs = pickle.load(f)

    return freqs


def air_s(query_terms: Iterable[str], phrase_terms: Iterable[str], model: KeyedVectors) -> float:
    # Filter out the terms not in the model to avoid triggering an exception
    query_terms = [t for t in query_terms if t in model]
    phrase_terms = [t for t in phrase_terms if t in model]
    phrase_matrix = model[phrase_terms]
    return sum(idf(q) * air_align(q, phrase_matrix, model) for q in query_terms)


def air_remaining(query_terms: Iterable[str], explanations_terms: Iterable[Iterable[str]], model: KeyedVectors) -> Set[str]:
    """ Returns the set of terms not yet covered by an explanation """

    query = set(query_terms)
    explanation = set()
    for exp in explanations_terms:
        explanation |= set(exp)

    phrase_matrix = model[[e for e in explanation if e in model]]
    # filtered_query = {q for q in query if air_align(q, phrase_matrix, model) >= 0.70}

    return query - explanation


def air_coverage(query_terms: Iterable[str], explanations_terms: Iterable[Iterable[str]]) -> float:
    """ Returns the coverage of terms in the query by the explanations' terms """
    query = set(query_terms)

    intersections = [(query & set(exp)) for exp in explanations_terms]
    union = set()
    for intersection in intersections:
        union |= intersection

    return len(union) / len(query)

