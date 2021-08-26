import itertools as it
from collections import defaultdict
from functools import lru_cache
from typing import Set, Tuple, List, Optional, AbstractSet, FrozenSet, Collection, Dict, NamedTuple, \
    cast, Sequence, Iterable
from weakref import WeakValueDictionary

# import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
# from networkx import Graph, NetworkXNoPath
from numpy import ndarray
from scipy import stats
from spacy.language import Language

import nlp
import utils
from actions import Query, QueryType
from parsing import QASCItem, Pair, EsHit

from machine_reading.ie import RedisWrapper
from machine_reading.ir.es import QASCIndexSearcher
from nlp import preprocess, load_stop_words, air_coverage
from tfidf import TfIdfHelper
from topic_modeling import TopicsHelper


class CandidateEntity(NamedTuple):
    entity: Optional[str]
    score: float


class CandidatePair(NamedTuple):
    pair: Tuple[str, str]
    score: float



def get_terms(phrase: str) -> str:
    # Lower case and tokenize
    terms = phrase.lower().split()

    # filter stop wrods
    kept = {t for t in terms if t not in load_stop_words()}

    return " ".join(kept)


class QASCInstanceEnvironment:
    """ Represents the environment on which a problem is operated """

    # Store the assembled knowledge graphs here to avoid hitting the indices repeatedly
    _kg_cache = WeakValueDictionary()

    def __init__(self, item: QASCItem, max_iterations: int, use_embeddings: bool, num_top_entities: int,
                 seed: int, ir_index: QASCIndexSearcher, redis: RedisWrapper, embeddings: KeyedVectors,
                 # vector_space: EmbeddingSpaceHelper  #, topics_helper: TopicsHelper,
                 # tfidf_helper: TfIdfHelper,
                 language: Language, doc_universe: Sequence[EsHit]) -> None:
                 # ) -> None:
        self.item = item
        self.ir_index = ir_index
        self.redis = redis
        self.embeddings: KeyedVectors = embeddings
        self._language = language
        self._seed = seed
        self._use_embeddings = use_embeddings
        self.max_iterations = max_iterations
        self.num_top_entities = num_top_entities
        self.doc_universe = doc_universe

        self.doc_set: Set[str] = set()
        self.iterations = 0
        self._latest_docs = None
        self._singleton = None
        self._and = None
        self._or = None
        self._obs = None
        self._and_log: Set[FrozenSet[str]] = set()
        self._or_log: Set[FrozenSet[str]] = set()
        self._singleton_log: Set[str] = set()
        self._rng = utils.build_rng(seed)
        # self.prev_kg: Graph = self.kg
        self._introductions: Dict[str, int] = dict()
        self._disable_topic_features = False
        self._disable_query_features = False
        self._disable_search_state_features = False

        self.explanation: List[str] = list()
        self._query: Set[str] = set()
        self._curr_remaining: Set[str] = set()
        self._prev_remaining: Set[str] = set()

        self._prev_score: float = 0.

        # This is to do the ablation tests
        config = utils.read_config()

        if "ablation" in config:
            ablation = config['ablation']
            self._disable_topic_features = ablation['disable_topic'].lower() == 'true'
            self._disable_query_features = ablation['disable_query'].lower() == 'true'
            self._disable_search_state_features = ablation['disable_search'].lower() == 'true'


    def expand_query(self):
        if len(self.explanation) > 0:
            remaining = self.remaining
            if len(remaining) <= 4:
                # Query expansion
                explanation_terms = set(it.chain.from_iterable(preprocess(self.explanation, self._language, stem=True)))
                self._query |= explanation_terms


    @property
    def query(self) -> Set[str]:

        # Terms of the question and answer:
        if len(self.explanation) == 0:
            item = self.item
            q_terms = preprocess(item.question, self._language, stem=True)
            a_terms = preprocess(item.answer, self._language, stem=True)
            self._query = set(it.chain(q_terms, a_terms))

        return self._query

    @property
    def remaining(self):

        explanation_terms = set(it.chain.from_iterable(preprocess(self.explanation, self._language, stem=True)))
        return self.query - explanation_terms

    @property
    def num_docs(self) -> int:
        """ Returns the size of the document set """
        return len(self.doc_set)

    @property
    def changed_remaining(self):

        prev = self._prev_remaining
        curr = self.remaining

        return curr != prev

    def _determine_outcome(self, query: Set[str], explanation: Sequence[str], language: Language) -> bool:
        """ Factored out this method to leverage the LRU decorator """
        # Determine if the problem is finished by the query coverage metric

        explanation_terms = preprocess(explanation, language, stem=True)

        coverage = air_coverage(query, explanation_terms)

        if coverage == 1.0:
            return True
        else:
            return False

    @property
    def status(self) -> bool:
        """
        Determines whether there's a path connecting the endpoints of the problem in the current document set

        Returns:
            True, and the shortest path if the path exists in the KG
            False, None, if there's no path in the KG
        """

        changed_remaining = self.changed_remaining
        successful = self.success

        finished = not changed_remaining or successful

        # Time out when the max number of iterations is reached
        if not finished and self.iterations == self.max_iterations:
            finished = True

        return finished

    @property
    def success(self):
        return self._determine_outcome(self.query, self.explanation, self._language)

    def add_explanation(self, phrase: str) -> None:
        # Keep track of remaining terms
        self._prev_remaining = self.remaining
        self.explanation.append(phrase)

    @property
    def path(self) -> Optional[Sequence[str]]:
        """ Returns the path connecting the endpoints of the problem or None if it is non existent """
        if self.status:
            return self.explanation
        else:
            return None

    def add_docs(self, new_docs: Collection[str]) -> None:
        """ Adds the documents to the knowledge graph.
         If query is specified, we keep track of the endpoints
         """

        # Add the new docs
        self.doc_set |= new_docs
        # Keep track of which where the latest doc added to the graph
        self._latest_docs = frozenset(new_docs)

        # Reset the sampled entities, as the KG changed
        if len(new_docs) > 0:
            self._singleton = None
            self._and = None
            self._or = None

        self._obs = None

    # Syntactic sugars
    def __iadd__(self, other: Collection[str]) -> None:
        """ Alias for add_docs """
        self.add_docs(other)

    def __len__(self) -> int:
        """ Alias for num_docs"""
        return self.num_docs

    def __bool__(self) -> bool:
        """ Returns only whether the environment has a path connecting the endpoints of the problem,
        without returning the path """
        outcome = self.status
        return outcome

    ###################

    @property
    def latest_docs(self) -> FrozenSet[str]:
        """ Return the latest batch of docs added to the knowledge graph """
        if self._latest_docs is None:
            return frozenset()
        else:
            return self._latest_docs

    def fetch_docs(self, terms: Iterable[str]) -> AbstractSet[str]:
        """ Get the documents returned by the query and just return the new ones based on the current state of this
        environment """
        ir_index = self.ir_index
        try:
            result = ir_index.search(' '.join(terms), 20)
        except Exception as ex:
            print(ex)
            result = list()

        docs = {doc for doc, score in result}

        # Get the incremental documents
        new_docs = docs - self.doc_set

        return new_docs

    def reset(self) -> None:
        """ Restarts the state of the environment to construction state """
        self.doc_set = set()
        self.iterations = 0
        self._latest_docs = None
        self._singleton = None
        self._and = None
        self._or = None
        self._obs = None
        self._and_log = set()
        self._or_log = set()
        self._singleton_log = set()
        self._rng = utils.build_rng(self._seed)
        self._query = set()
        self.explanation = list()

    def ranked_docs(self) -> Sequence[Tuple[str, float]]:
        """ Rank the documents by their alignment score to the current query """
        docs = [doc.text for doc in self.doc_universe if doc not in self.explanation]  # Find the eligible docs

        # Rank them by alignment
        scores = [nlp.air_s(self.remaining, e, self.embeddings) for e in preprocess(docs, self._language, stem=True)]

        return list(sorted(zip(docs, scores), key=lambda s: s[1]))

    def fr_score(self, normalize: bool = False) -> float:
        score = nlp.air_coverage(self.query,
                         nlp.preprocess(self.explanation, self._language, stem=True))

        if normalize:
            denominator = len(self.explanation) if len(self.explanation) > 0 else 1
            score /= denominator

        return score

    def rl_reward(self) -> float:
        """ Returns the RL reward signal """

        prev = self._prev_score
        current = self.fr_score(normalize=True)

        return current - prev

    @property
    def observe(self) -> np.ndarray:
        """
        # Entity A embedding
        # Entity B embedding
        ## Entity A type
        ## Entity B type
        # Normalized iteration number (to keep it relative to the potential length of the trial)
        # Number of edges in graph
        # Number of vertices in graph
        # Topic distribution of the graph
        # Relative Delta entropy for: exploration, exploitation, singleton
        # Relative divergence for: exploration, exploitation, singleton
        # Num new docs retrieved by: exploration, exploitation, singleton
        Returns the current state observation vector to be used by a policy and for RL

        :return: Numpy array with the observation values
        """

        # Cache the observation for as long as there is no mutation
        if self._obs is not None:
            return self._obs
        else:

            item = self.item
            q_emb = self.encode_phrase(item.question)
            a_emb = self.encode_phrase(item.answer)

            if len(self.explanation) > 0:
                explanation_embs = list()
                for phrase in self.explanation:
                    explanation_embs.append(self.encode_phrase(phrase))

                explanation_embs = np.concatenate(explanation_embs, axis=0)

                ex_emb = np.mean(explanation_embs, axis=0).reshape(1, -1)
            else:
                ex_emb = np.zeros_like(q_emb)


            one_hot_iterations = np.zeros(self.max_iterations + 1)
            one_hot_iterations[self.iterations] = 1


            # Put the features together
            components = [q_emb, a_emb, ex_emb]

            components.append(np.asarray(one_hot_iterations).reshape((1, -1)))


            obs = np.concatenate(components, axis=1).astype('float32').reshape(-1)

            self._obs = obs

            return obs

    def encode_phrase(self, phrase: str) -> ndarray:
        embeddings = self.embeddings
        embs = list()
        for w in nlp.preprocess(phrase, self._language, stem=True):

            if w in embeddings:
                embs.append(embeddings[w].reshape(1, 300))
            else:
                embs.append(embeddings['OOV'].reshape(1, 300))

        if len(embs) > 0:
            avg_emb = np.mean(np.stack(embs), axis=0)
        else:
            avg_emb = embeddings['OOV'].reshape(1, 300)

        return avg_emb
