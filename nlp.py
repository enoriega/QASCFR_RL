""" This file has NLP utilities, such as tokenization, embedding lookups, etc"""
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union, Mapping, Iterable, Optional, cast, FrozenSet, List, Type, Set

import numpy as np
import spacy
from gensim.models import KeyedVectors
from nltk import PorterStemmer, SnowballStemmer
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


@lru_cache(maxsize=10000)
def preprocess(text: Union[str, Iterable[str]], nlp: Language) -> Iterable[Sequence[str]]:
    """ Splits the input string into a sequence of tokens. Everything is done lazily """

    # Make sure we do the "batched" version even if its a single input text
    if type(text) == str:
        text = [text]

    # Normalize and tokenize. Replace the underscores of the entities for white spaces
    text = (t.lower().strip().replace("_", " ") for t in text)

    # Pipe the text for efficient tokenization. Disable the tagger and parser for now
    docs = nlp.pipe(text, disable=['parser'])

    # Fetch the stop words
    stop_words = load_stop_words()

    # Do some stemming with this
    stemmer = SnowballStemmer(language='english')

    # Return a generator that will return the tokens as long as they're not a stop word
    return [[w for w in (stemmer.stem(token.text) for token in doc) if w not in stop_words] for doc in docs]


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


def air_s(query_terms: List[str], phrase_terms: List[str], model: KeyedVectors) -> float:
    phrase_matrix = model[phrase_terms]
    return sum(idf(q) * air_align(q, phrase_matrix, model) for q in query_terms)


def air_remaining(query_terms: Sequence[str], explanations_terms: Sequence[Sequence[str]]) -> Set[str]:
    """ Returns the set of terms not yet covered by an explanation """

    query = set(query_terms)
    explanation = set()
    for exp in explanations_terms:
        explanation |= set(exp)

    return query - explanation


def air_coverage(query_terms: Sequence[str], explanations_terms: Sequence[Sequence[str]]) -> float:
    """ Returns the coverage of terms in the query by the explanations' terms """
    query = set(query_terms)

    intersections = sum(len(query & set(exp)) for exp in explanations_terms)
    return intersections / len(query)

