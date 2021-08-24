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


# TODO Clean up this file

# @lru_cache(maxsize=100)
# def _get_eligible_pairs(kg: Graph, log: FrozenSet[FrozenSet[str]], singletons: FrozenSet[str]):
#     """
#     Generate the pairs to be evaluated for sampling
#     :param log: If a pair is contained here, it is not included
#     :return: List with the pairs eligible to be part of a query
#     """
#
#     eligible_pairs = list()
#     seen = set()
#     for a in kg.nodes:
#         for b in kg.nodes:
#             if a != b and a not in singletons and b not in singletons:
#                 pair = frozenset((a, b))
#                 if pair not in log:
#                     if pair not in seen:
#                         eligible_pairs.append(pair)
#                         seen.add(pair)
#     return eligible_pairs


# @lru_cache(maxsize=128)



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

    def __init__(self, problem: QASCItem, max_iterations: int, use_embeddings: bool, num_top_entities: int,
                 seed: int, ir_index: QASCIndexSearcher, redis: RedisWrapper, embeddings: KeyedVectors,
                 # vector_space: EmbeddingSpaceHelper  #, topics_helper: TopicsHelper,
                 # tfidf_helper: TfIdfHelper,
                 language: Language, doc_universe: Sequence[EsHit]) -> None:
                 # ) -> None:
        self.problem = problem
        self.ir_index = ir_index
        self.redis = redis
        self.embeddings = embeddings
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
                # self._query = (remaining | (explanation_terms - self._query))
                # remaining = nlp.air_remaining(self._query, preprocess(self.explanation, self._language, stem=True),
                #                               self.embeddings)
                # self._curr_remaining = remaining

    @property
    def query(self) -> Iterable[str]:

        # Terms of the question and answer:
        if len(self.explanation) == 0:
            item = self.problem
            q_terms = preprocess(item.question, self._language, stem=True)
            a_terms = preprocess(item.answer, self._language, stem=True)
            self._query = set(it.chain(q_terms, a_terms))

        return self._query

    @property
    def remaining(self):
        # if len(self.explanation) == 0:
        #     return set(self.query)
        # else:
        #     return nlp.air_remaining(self.query, preprocess(self.explanation, self._language, stem=True), self.embeddings)
        explanation_terms = set(it.chain.from_iterable(preprocess(self.explanation, self._language, stem=True)))
        return self.query - explanation_terms

    @property
    def num_docs(self) -> int:
        """ Returns the size of the document set """
        return len(self.doc_set)

    @property
    def changed_remaining(self):
        if self.iterations == 1:
            return True
        elif len(self.doc_set) == 0:
            return False
        else:
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
        docs = [doc for doc in self.doc_set if doc not in self.explanation]  # Find the eligible docs

        # Rank them by alignment
        scores = [nlp.air_s(self.remaining, e, self.embeddings) for e in preprocess(docs, self._language, stem=True)]

        return list(sorted(zip(docs, scores), key=lambda s: s[1]))

    @property
    def fr_score(self) -> float:
        # explanation_terms = set(it.chain.from_iterable(preprocess(self.explanation, self._language)))
        # remaining = self._curr_remaining
        # qa_terms = set(preprocess(self.problem.question, self._language)) | \
        #           set(preprocess(self.problem.answer, self._language))
        #
        # explanation_coverage = len(explanation_terms - remaining) / len(explanation_terms)
        # original_coverage = len(qa_terms - explanation_terms) / len(qa_terms)
        #
        # return explanation_coverage + original_coverage
        return nlp.air_coverage(self.query,
                         nlp.preprocess(self.explanation, self._language, stem=True))

    def rl_reward(self) -> float:
        """ Returns the RL reward signal """

        prev = self._prev_score
        current = self.fr_score

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

            # vs = self._vector_space
            # topics = self._topics_helper
            # normalized_iterations = self.iterations / self.max_iterations
            one_hot_iterations = np.zeros(self.max_iterations + 1)
            one_hot_iterations[self.iterations] = 1
            # num_edges = len(self.kg.edges)
            # num_vertices = len(self.kg.nodes)
            # graph_topics_dist = topics.compute_topic_dist(frozenset(self.doc_set))

            num_entities = self.num_top_entities

            # AND query features
            # and_features = np.concatenate([self.query_features(QueryType.And, ix, graph_topics_dist)
            #                                for ix in range(num_entities)])
            #
            # # OR query features
            # or_features = np.concatenate([self.query_features(QueryType.Or, ix, graph_topics_dist)
            #                               for ix in range(num_entities)])
            #
            # # Singleton query features
            # singleton_features = np.concatenate([self.query_features(QueryType.Singleton, ix, graph_topics_dist)
            #                                      for ix in range(num_entities)])
            #
            # # Put the features together
            components = list()
            # for feats in [and_features, or_features, singleton_features]:
            #     components.append(feats)
            #
            if not self._disable_search_state_features:
                components.append(one_hot_iterations)
                # components.append([num_edges, num_vertices])

            obs = np.concatenate(components).astype('float32')
            #
            # # If the embeddings where requested, use them
            # if self._use_embeddings:
            #     embeddings = vs[[self.problem.question, self.problem.answer]]
            #     obs = np.concatenate([embeddings[0], embeddings[1], obs])

            self._obs = obs

            return obs

    # def query_features(self, query_type: QueryType, entity_index: int, graph_topics_dist: np.ndarray) -> np.ndarray:
    #     if query_type == QueryType.Or:
    #         log = self._or_log
    #         eligible_entities = self.or_entities
    #     elif query_type == QueryType.And:
    #         log = self._and_log
    #         eligible_entities = self.and_entities
    #     elif query_type == QueryType.Singleton:
    #         log = self._singleton_log
    #         eligible_entities = self.singleton_entity
    #     else:
    #         raise ValueError("Un supported query type")
    #
    #     if entity_index < len(eligible_entities):
    #         if query_type == QueryType.Singleton and (len(self.kg.nodes) - len(log) > 0):
    #             q = Query(cast(List[CandidateEntity], eligible_entities)[entity_index].entity, QueryType.Singleton)
    #             features = [eligible_entities[entity_index].score] + self.topic_features(graph_topics_dist, q)
    #         elif len(self.get_eligible_pairs(log)) > 0:
    #             q = Query(eligible_entities[entity_index].pair, query_type)
    #             features = [eligible_entities[entity_index].score] + self.topic_features(graph_topics_dist, q)
    #         else:
    #             features = np.zeros(shape=(4,))
    #     else:
    #         features = np.zeros(shape=(4,))
    #
    #     if self._disable_topic_features:
    #         features = features[0:-2]
    #     if self._disable_query_features:
    #         features = features[2:]
    #
    #     return features
    #
    # def topic_features(self, graph_topics_dist: np.ndarray, query: Query) -> List[float]:
    #     """ Computes the entropy and divergence query features """
    #
    #     topics = self._topics_helper
    #
    #     # Identify the new documents that would be added
    #     new_docs, _ = self.fetch_docs(query)
    #
    #     # Count them
    #     num_new_docs = len(new_docs)
    #     # Compute the topic distribution of the new proposed documents
    #     new_docs_dist = topics.compute_topic_dist(frozenset(new_docs))
    #     # Compute the entropy of that distribution
    #     new_docs_entropy = stats.entropy(new_docs_dist)
    #     # Get the distribution and entropy of the latest documents
    #     last_query_dist = topics.compute_topic_dist(self.latest_docs)
    #     last_query_entropy = stats.entropy(last_query_dist)
    #     # Compute the delta entropy
    #     delta_entropy = new_docs_entropy - last_query_entropy
    #     # Compute the KL-Divergence of the graph distribution and of the latest documents
    #     differential_divergence = stats.entropy(graph_topics_dist, new_docs_dist)
    #
    #     # Put them together and return the results
    #     features = [num_new_docs, delta_entropy, differential_divergence]
    #     return features
    #
    #
    # def rl_reward(self) -> Tuple[bool, float]:
    #     """ Computes the reward for RL """
    #
    #     # Figure out if the environment contains a path
    #     succeeded = bool(self)
    #
    #     # This is the reward component based on the outcome of the search
    #     success_reward = 1000
    #
    #     num_papers = len(self.latest_docs)
    #
    #     # return the corresponding reward, either the outcome reward or the living reward
    #     reward = -num_papers * 3
    #     if succeeded:
    #         reward += success_reward
    #         finished = True
    #     else:
    #         if num_papers == 0:
    #             reward -= 100
    #
    #         if self.iterations == self.max_iterations:
    #             finished = True
    #         else:
    #             finished = False
    #
    #     return finished, reward
    #
    # def shaping_potential(self) -> float:
    #     """ Returns the shaping potential used for reward shaping """
    #     total_potential = 1000
    #     problem = self.problem
    #     gt_path = problem.gt_path
    #     edge_potential = total_potential / len(gt_path)
    #     kg = self.kg
    #     num_present_edges = 0
    #     for (a, b) in gt_path:
    #         if (a, b) in kg.edges:
    #             num_present_edges += 1
    #
    #     if num_present_edges == 0:
    #         return 0
    #     else:
    #         return edge_potential * num_present_edges
