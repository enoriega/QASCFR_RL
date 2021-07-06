""" This file has NLP utilities, such as tokenization, embedding lookups, etc"""
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union, Mapping, Iterable, Optional, cast, FrozenSet

import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Doc

import utils


@lru_cache(maxsize=1)
def load_stop_words() -> FrozenSet[str]:
    """ Return a set of stop words. Cache the result for efficiency """
    stop_words = set(STOP_WORDS)

    # Remove some words that could be useful
    for w in ['do', 'it', 'yourself',
              'the', 'who', 'front',
              'five', 'ten', 'rather',
              'be', 'next']:
        stop_words.remove(w)

    # Add punctuation marks as stop words
    for w in ['&', "'", '(', ')', ',', '-', '.',
              '/', ':', '–', '!', '"', '#', '$',
              '%', "''", '*', '+', "..", "...",
              ";", '[', ']', '<', '>', '=', '?',
              '@', '`']:
        stop_words.add(w)

    return frozenset(stop_words)


@lru_cache(maxsize=10000)
def preprocess(text: Union[str, Sequence[str]], nlp: Language) -> Iterable[Sequence[str]]:
    """ Splits the input string into a sequence of tokens. Everything is done lazily """

    # Make sure we do the "batched" version even if its a single input text
    if type(text) == str:
        text = [text]

    # Normalize and tokenize. Replace the underscores of the entities for white spaces
    text = (t.lower().strip().replace("_", " ") for t in text)

    # Pipe the text for efficient tokenization. Disable the tagger and parser for now
    docs = nlp.pipe(text, disable=['tagger', 'parser'])

    # Fetch the stop words
    stop_words = load_stop_words()

    # Do some stemming
    # stemmer = PorterStemmer()  # TODO: Do something better about stemming/lemmatization

    # Return a generator that will return the tokens as long as they're not a stop word
    return [[w for w in (token.text for token in doc) if w not in stop_words] for doc in docs]


def preprocess_entities(gt_path: Union[Path, str],
                        nlp: Optional[Language] = None) -> Mapping[str, Sequence[str]]:
    """
    Reads the entities from the shelf, returns a preprocessed and tokenized dict with the
    entities' tokens
    """

    # Let's read the entities from the sample paths
    _, inverted_index = utils.build_indices(gt_path)

    # The entities come from the keys of the inverted index, discard the paris
    entities = [cast(str, e) for e in inverted_index.keys() if type(e) == str]

    # If no language pipeline is provided yet
    if nlp is None:
        # Load a spaCy english language, don't need the models for now
        nlp = spacy.blank("en")

    # Preprocess and tokenize those entities
    tokenized_entities = preprocess(tuple(entities), nlp)

    # Generate a dictionary that maps the inputs to the outputs
    ret = dict(zip(entities, tokenized_entities))

    return ret


def average_embedding(tokens: Sequence[str],
                      nlp: Language) -> np.ndarray:
    """ Returns the average embedding of an entity based on its tokens """
    # Build a document from the pre-tokenized input
    spaces = [True] * (len(tokens) - 1) + [False]
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

    # Delegate the vector representation to spaCy. It will compute the average itself
    ret = doc.vector

    return ret


class EmbeddingSpaceHelper:
    """ This class handles mostly cosine similarity operations """

    def __init__(self, gt_path: Union[Path, str], nlp: Language) -> None:
        """
        Initialize the embeddings matrix from the entities in the GT sample
        :param gt_path: Path to the GT sample
        :param nlp: spaCy pipeline to fetch entity vectors
        """
        self.nlp = nlp

        # Preprocess the entities
        tokenized_entities = preprocess_entities(gt_path, nlp)
        self._tokenized_entities = tokenized_entities

        # Build the embeddings matrix from the pre-processed entities
        embedding_matrix = np.vstack(
            [average_embedding(t, nlp) for t in tokenized_entities.values()])

        # Pre-normalize the embeddings for efficient cosine similarity queries
        norms = np.linalg.norm(embedding_matrix, axis=1)
        # Add a small residual to the entries where the norm was zero to avoid a division by zero
        norms = np.where(norms != 0, norms, 1e-10)
        # Normalize the matrix
        normalized_embedding_matrix = embedding_matrix / norms.reshape(-1, 1)

        self._matrix = embedding_matrix
        self._normalized_matrix = normalized_embedding_matrix

        self._similarities = normalized_embedding_matrix @ np.transpose(normalized_embedding_matrix)

        # Build index and inverted index of the entities to efficiently address the embedding matrices
        self._entity_ix = {e: ix for ix, e in enumerate(tokenized_entities.keys())}
        self._inv_entity_ix = {ix: e for ix, e in enumerate(tokenized_entities.keys())}

    def get(self, items: Union[str, Sequence[str]], normalized: bool = False) -> np.ndarray:
        """ Fetches the entity average embedding may return the normalized version if requested by the caller"""

        if type(items) == str:
            items = [items]

        for item in items:
            if item not in self._tokenized_entities:
                # If the entity is not in the index fail
                raise IndexError(f"{item} not among the embeddings")
        else:
            # Select the appropriate matrix to address
            if normalized:
                matrix = self._normalized_matrix
            else:
                matrix = self._matrix

            # Use the entity index to slice the matrix and return the embedding vector
            return matrix[[self._entity_ix[i] for i in items]]

    def __getitem__(self, items: Union[str, Sequence[str]]) -> np.array:
        """ Syntactic sugar to get method. Will return the un-normalized matrix """
        return self.get(items)

    # def similarity(self, a: Union[str, Sequence[str]], b: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
    #     """
    #     Computes the cosine similarity of an entity with respect to all other entities or a subset of entities
    #     :param a: Key or keys to our query to compute the cosine similarity
    #     :param b: Another entity of subset of entities to compute the cosine similarity with a.
    #               If None, then result a vector with all the similarities
    #     :return: Array with the numpy similarity wrt
    #     """
    #
    #     if type(a) == str:
    #         a = [a]
    #
    #     # Get the normalized version  of the key vector. Reshape it to make it a column vector
    #     va = self.get(a, normalized=True)  # 300 is the embedding size
    #
    #     # Fetch the normalized matrix
    #     matrix = self._normalized_matrix
    #
    #     # Since both operands are normalized with L2, cosine similarity reduces to a dot product
    #     similarities = matrix @ va.transpose()
    #
    #     # If a subset was requested, slice the similarities vector to select the appropriate indices
    #     if b is not None:
    #         # If the second entity is just one, make a list with as single elements
    #         if type(b) == str:
    #             b = [b]
    #         # Fetch the indices in the matrix of the requested entities
    #         indices = [self._entity_ix[e] for e in b]
    #         # Return the appropriate subset of the similarities vector
    #         return similarities[indices, range(len(indices))]  # .reshape(1, )
    #     else:
    #         # Return all the similarities vector
    #         return similarities

    def similarity(self, a: Union[str, Sequence[str]], b: Optional[Union[str, Sequence[str]]] = None) -> np.ndarray:
        """
        Computes the cosine similarity of an entity with respect to all other entities or a subset of entities
        :param a: Key or keys to our query to compute the cosine similarity
        :param b: Another entity of subset of entities to compute the cosine similarity with a.
                  If None, then result a vector with all the similarities
        :return: Array with the numpy similarity wrt
        """

        if type(a) == str:
            a = [a]

        # Fetch the normalized matrix
        similarities = self._similarities

        # If a subset was requested, slice the similarities vector to select the appropriate indices
        if b is not None:
            # If the second entity is just one, make a list with as single elements
            if type(b) == str:
                b = [b]
            # Fetch the indices in the matrix of the requested entities
            indices_a = [self._entity_ix[e] for e in a]
            indices_b = [self._entity_ix[e] for e in b]
            # Return the appropriate subset of the similarities vector
            return similarities[indices_a, indices_b]  # .reshape(1, )
        else:
            # Return all the similarities vector
            return similarities

    def top_k(self, a: str, k: int):
        """ Return the top k neighbors to the requested entity """

        # Compute the similarities with respect to all the entities
        sims = self.similarity(a)
        # Rank them by their value
        ranked = np.argsort(sims, axis=0)
        # Since the most similar elements are those with the highest values, select the last K members, and ignore
        # The last element which is the key entity itself
        selected = ranked[-k - 1:-2].squeeze()
        # Resolve the identity of the neighbors and reverse the order to return them with the expected order
        return list(reversed([self._inv_entity_ix[s] for s in selected]))
