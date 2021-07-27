import itertools as it
from collections import defaultdict
from functools import lru_cache
from typing import Set, Tuple, List, Optional, AbstractSet, FrozenSet, Collection, Dict, NamedTuple, \
    cast
from weakref import WeakValueDictionary

import networkx as nx
import numpy as np
from networkx import Graph, NetworkXNoPath
from scipy import stats
from spacy.language import Language

import utils
from actions import Query, QueryType
from parsing import QASCInstance, Pair

from machine_reading.ie import RedisWrapper
from machine_reading.ir.es import QASCIndexSearcher
from nlp import preprocess, load_stop_words
from tfidf import TfIdfHelper
from topic_modeling import TopicsHelper


class CandidateEntity(NamedTuple):
    entity: Optional[str]
    score: float


class CandidatePair(NamedTuple):
    pair: Tuple[str, str]
    score: float


# TODO Clean up this file

@lru_cache(maxsize=100)
def _get_eligible_pairs(kg: Graph, log: FrozenSet[FrozenSet[str]], singletons: FrozenSet[str]):
    """
    Generate the pairs to be evaluated for sampling
    :param log: If a pair is contained here, it is not included
    :return: List with the pairs eligible to be part of a query
    """

    eligible_pairs = list()
    seen = set()
    for a in kg.nodes:
        for b in kg.nodes:
            if a != b and a not in singletons and b not in singletons:
                pair = frozenset((a, b))
                if pair not in log:
                    if pair not in seen:
                        eligible_pairs.append(pair)
                        seen.add(pair)
    return eligible_pairs


@lru_cache(maxsize=128)
def _determine_outcome(start, end, kg) -> Tuple[bool, Optional[List[str]]]:
    """ Factored out this method to leverage the LRU decorator """
    # Find a the shortest path between the endpoints

    # shortest_path = nx.shortest_path(kg, start, end)  # TODO Change the outcome criteria here
    paths = list(nx.all_simple_paths(kg, start, end, 3))
    if len([p for p in paths if len(p) == 3]) >= 100:
        finished = True
    else:
        finished = False

    result = finished, paths

    return result


def get_terms(phrase:str) -> str:
    # Lower case and tokenize
    terms = phrase.lower().split()

    # filter stop wrods
    kept = {t for t in terms if t not in load_stop_words()}

    return " ".join(kept)


class QASCInstanceEnvironment:
    """ Represents the environment on which a problem is operated """

    # Store the assembled knowledge graphs here to avoid hitting the indices repeatedly
    _kg_cache = WeakValueDictionary()

    def __init__(self, problem: QASCInstance, max_iterations: int, use_embeddings: bool, num_top_entities: int,
                 seed: int, ir_index: QASCIndexSearcher, redis: RedisWrapper,
                 #vector_space: EmbeddingSpaceHelper  #, topics_helper: TopicsHelper,
                 # tfidf_helper: TfIdfHelper, nlp: Language) -> None:
                 ) -> None:
        self.problem = problem
        self.ir_index = ir_index
        self.redis = redis
        # self._vector_space = vector_space
        # self._topics_helper = topics_helper
        # self._tfidf_helper = tfidf_helper
        # self._nlp = nlp
        self._seed = seed
        self._use_embeddings = use_embeddings
        self.max_iterations = max_iterations
        self.num_top_entities = num_top_entities

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
        self.prev_kg: Graph = self.kg
        self._introductions: Dict[str, int] = dict()
        self._disable_topic_features = False
        self._disable_query_features = False
        self._disable_search_state_features = False

        # This is to do the ablation tests
        config = utils.read_config()

        if "ablation" in config:
            ablation = config['ablation']
            self._disable_topic_features = ablation['disable_topic'].lower() == 'true'
            self._disable_query_features = ablation['disable_query'].lower() == 'true'
            self._disable_search_state_features = ablation['disable_search'].lower() == 'true'

    @property
    def num_docs(self) -> int:
        """ Returns the size of the document set """
        return len(self.doc_set)

    @property
    def status(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Determines whether there's a path connecting the endpoints of the problem in the current document set

        Returns:
            True, and the shortest path if the path exists in the KG
            False, None, if there's no path in the KG
        """

        # First build a graph with the documents of the set
        kg = self.kg

        # Get the endpoints of the problem
        start = self.problem.question
        end = self.problem.answer

        finished, path = _determine_outcome(start, end, kg)

        # Time out when the max number of iterations is reached
        if not finished and self.iterations == self.max_iterations:
            finished = True

        return finished, path

    @property
    def path(self) -> Optional[List[str]]:
        """ Returns the path connecting the endpoints of the problem or None if it is non existent """
        _, path = self.status
        return path

    def add_docs(self, new_docs: Collection[str], query: Optional[Query] = None) -> None:
        """ Adds the documents to the knowledge graph.
         If query is specified, we keep track of the endpoints
         """
        # Keep track of the previous kg, for reward computation purposes
        self.prev_kg = self.kg
        # Add the new docs
        self.doc_set |= new_docs
        # Every time we add docs, we increment the iteration counter
        self.iterations += 1
        # Keep track of which where the latest doc added to the graph
        self._latest_docs = frozenset(new_docs)

        # TODO Deprecated, erase if it doesn't break at runtime
        # # Keep track of the iterations of introductions
        # for doc in new_docs:
        #     if doc not in self._introductions:
        #         self._introductions[doc] = self.iterations
        #         # Fetch the entities of those docs
        #         for e in self.index[doc]:
        #             if type(e) == str and e not in self._introductions:
        #                 self._introductions[e] = self.iterations

        # Log the query, if necessary
        endpoints = query.endpoints
        if query is not None:
            if query.type is QueryType.And:
                log = self._and_log
                self._and = None
            elif query.type is QueryType.Or:
                log = self._or_log
                if endpoints is not None:
                    self._singleton_log |= set(endpoints)
                    self._singleton = None
                    self._or = None
                    self._and_log.add(frozenset(endpoints))
                    self._and = None
            elif query.type is QueryType.Singleton:
                log = self._singleton_log
                self._singleton = None
            else:
                log = set()  # Should never fall into this case

            if endpoints is not None and type(endpoints) != str:
                endpoints = frozenset(endpoints)
            if endpoints is not None:
                log.add(endpoints)

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
        outcome, _ = self.status
        return outcome

    ###################

    @property
    def latest_docs(self) -> FrozenSet[str]:
        """ Return the latest batch of docs added to the knowledge graph """
        if self._latest_docs is None:
            return frozenset()
        else:
            return self._latest_docs

    @property
    def kg(self) -> Graph:
        """ Builds a knowledge graph out of the document set """

        # Get the endpoints of the problem
        start = self.problem.question
        end = self.problem.answer

        # The document set
        docs = self.doc_set

        cache = QASCInstanceEnvironment._kg_cache
        cache_key = (start, end, frozenset(docs))
        if cache_key in cache:
            return cache[cache_key]
        else:
            # Build the KG
            kg = Graph()

            kg.add_nodes_from((start, end))  # A bare bones KG contains the disconnected endpoints
            # Iterate over the documents to extract the edges that will be stitch together in the KG
            # This mimics the information extraction step on a more realistic scenario
            # Create an inverted index of extractions
            inverted_index = defaultdict(set)
            index = defaultdict(set)
            if len(docs) > 0:
                # Get all the elements of the documents
                for doc in (docs | {start, end}):
                    # Get the extractions from the cached redis client
                    extractions = self.redis.get_extractions(doc)
                    # Populate the index
                    index[doc] = extractions
                    # Populate the inverted index
                    for extraction in extractions:
                        inverted_index[extraction].add(doc)

                # Add the documents as nodes in the graph
                kg.add_nodes_from(docs)
                # Build the edges using frozensets to reduce density in the graph
                edges = set()
                for doc, extractions in index.items():
                    start = doc
                    for extraction in extractions:
                        for end in inverted_index[extraction]:
                            if start != end:
                                edges.add(frozenset((start, end)))
                # Add the built edges into the KG
                kg.add_edges_from(edges)

            # Store it in the cache
            cache[cache_key] = kg

            return kg

    def fetch_docs(self, query: Query) -> Tuple[AbstractSet[str], QueryType]:
        """ Get the documents returned by the query and just return the new ones based on the current state of this
        environment """

        # Use the inverted index to get the documents returned by the query
        qt = query.type

        ir_index = self.ir_index
        if query.endpoints is None or len(query.endpoints) == 0:
            return set(), qt
        else:
            if query.type is QueryType.Singleton:
                entity = query.endpoints
                query_str = entity
                max_hits = 10
            else:
                a, b = query.endpoints
                terms_a = get_terms(a)
                terms_b = get_terms(b)
                if query.type == QueryType.And:
                    query_str = f'{terms_a} {terms_b}'
                    max_hits = 10
                elif query.type == QueryType.Or:
                    query_str = f'({terms_a}) OR ({terms_b})'
                    max_hits = 20
                else:
                    # Shouldn't really fall into this case
                    raise Exception("Invalid query")

            try:
                result = ir_index.search(query_str, max_hits)
            except Exception as ex:
                print(ex)
                result = list()

            docs = {doc for doc, score in result}

            # Get the incremental documents
            new_docs = docs - self.doc_set

            return new_docs, qt

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
        self.prev_kg = self.kg

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
            num_edges = len(self.kg.edges)
            num_vertices = len(self.kg.nodes)
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
                components.append([num_edges, num_vertices])

            obs = np.concatenate(components).astype('float32')
            #
            # # If the embeddings where requested, use them
            # if self._use_embeddings:
            #     embeddings = vs[[self.problem.question, self.problem.answer]]
            #     obs = np.concatenate([embeddings[0], embeddings[1], obs])

            self._obs = obs

            return obs

    def query_features(self, query_type: QueryType, entity_index: int, graph_topics_dist: np.ndarray) -> np.ndarray:
        if query_type == QueryType.Or:
            log = self._or_log
            eligible_entities = self.or_entities
        elif query_type == QueryType.And:
            log = self._and_log
            eligible_entities = self.and_entities
        elif query_type == QueryType.Singleton:
            log = self._singleton_log
            eligible_entities = self.singleton_entity
        else:
            raise ValueError("Un supported query type")

        if entity_index < len(eligible_entities):
            if query_type == QueryType.Singleton and (len(self.kg.nodes) - len(log) > 0):
                q = Query(cast(List[CandidateEntity], eligible_entities)[entity_index].entity, QueryType.Singleton)
                features = [eligible_entities[entity_index].score] + self.topic_features(graph_topics_dist, q)
            elif len(self.get_eligible_pairs(log)) > 0:
                q = Query(eligible_entities[entity_index].pair, query_type)
                features = [eligible_entities[entity_index].score] + self.topic_features(graph_topics_dist, q)
            else:
                features = np.zeros(shape=(4,))
        else:
            features = np.zeros(shape=(4,))

        if self._disable_topic_features:
            features = features[0:-2]
        if self._disable_query_features:
            features = features[2:]

        return features

    def topic_features(self, graph_topics_dist: np.ndarray, query: Query) -> List[float]:
        """ Computes the entropy and divergence query features """

        topics = self._topics_helper

        # Identify the new documents that would be added
        new_docs, _ = self.fetch_docs(query)

        # Count them
        num_new_docs = len(new_docs)
        # Compute the topic distribution of the new proposed documents
        new_docs_dist = topics.compute_topic_dist(frozenset(new_docs))
        # Compute the entropy of that distribution
        new_docs_entropy = stats.entropy(new_docs_dist)
        # Get the distribution and entropy of the latest documents
        last_query_dist = topics.compute_topic_dist(self.latest_docs)
        last_query_entropy = stats.entropy(last_query_dist)
        # Compute the delta entropy
        delta_entropy = new_docs_entropy - last_query_entropy
        # Compute the KL-Divergence of the graph distribution and of the latest documents
        differential_divergence = stats.entropy(graph_topics_dist, new_docs_dist)

        # Put them together and return the results
        features = [num_new_docs, delta_entropy, differential_divergence]
        return features

    # def rl_reward(self) -> Tuple[bool, float]:
    #     """ Computes the reward for RL """
    #
    #     # Figure out if the environment contains a path
    #     succeeded = bool(self)
    #
    #     # This is the reward component based on the outcome of the search
    #     success_reward = 100
    #     failure_reward = -100
    #     living_reward = 5
    #
    #     # Information gain component, in terms of how many new connections where added by the latest step
    #     num_new_docs = len(self.latest_docs)
    #     if num_new_docs > 0:
    #         new_relations = len(self.kg.edges) - len(self.prev_kg.edges)
    #         information_ratio = new_relations / num_new_docs
    #     else:
    #         information_ratio = 0
    #
    #     sigmoid_factor = success_reward * 0.5
    #
    #     scaled_information_ratio = \
    #         (1 / (1 + np.exp(-(information_ratio - sigmoid_factor)))) * sigmoid_factor
    #
    #     # return the corresponding reward, either the outcome reward or the living reward
    #     if succeeded:
    #         reward = success_reward
    #         finished = True
    #     else:
    #         if self.iterations == self.max_iterations:
    #             reward = failure_reward
    #             finished = True
    #         else:
    #             reward = scaled_information_ratio - living_reward
    #             finished = False
    #
    #     return finished, reward

    def rl_reward(self) -> Tuple[bool, float]:
        """ Computes the reward for RL """

        # Figure out if the environment contains a path
        succeeded = bool(self)

        # This is the reward component based on the outcome of the search
        success_reward = 1000

        num_papers = len(self.latest_docs)

        # return the corresponding reward, either the outcome reward or the living reward
        reward = -num_papers * 3
        if succeeded:
            reward += success_reward
            finished = True
        else:
            if num_papers == 0:
                reward -= 100

            if self.iterations == self.max_iterations:
                finished = True
            else:
                finished = False

        return finished, reward

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
    #     reward = 0
    #     if succeeded:
    #         reward += success_reward
    #         finished = True
    #     else:
    #         if self.iterations == self.max_iterations:
    #             finished = True
    #         else:
    #             finished = False
    #
    #     return finished, reward

    @property
    def and_entities(self) -> List[CandidatePair]:
        if self._and is not None:
            return self._and
        else:
            vs = self._vector_space
            eligible_pairs = self.get_eligible_pairs(self._and_log)

            # Filter out eligible pairs that don't add anything new to the KG
            useful_pairs = list()
            for pair in eligible_pairs:
                docs, _ = self.fetch_docs(Query(pair, QueryType.And))
                if len(docs - self.doc_set) > 0:
                    useful_pairs.append(pair)

            eligible_pairs = useful_pairs

            if len(eligible_pairs) > 0:
                self._and = [CandidatePair(tuple(pair), 1) for pair in eligible_pairs]

            else:
                self._and = [CandidatePair(tuple(), 0.)]

        return self._and

    def get_eligible_pairs(self, log: Set[FrozenSet[str]]):
        return _get_eligible_pairs(self.kg, frozenset(log), frozenset(self._singleton_log))

    @property
    def or_entities(self) -> List[CandidatePair]:
        if self._or is not None:
            return self._or
        else:
            # tfidf = self._tfidf_helper
            vs = self._vector_space
            eligible_pairs = self.get_eligible_pairs(self._or_log)

            # Filter out eligible pairs that don't add anything new to the KG
            useful_pairs = list()
            for pair in eligible_pairs:
                docs, _ = self.fetch_docs(Query(pair, QueryType.Or))
                if len(docs - self.doc_set) > 0:
                    useful_pairs.append(pair)

            eligible_pairs = useful_pairs

            if len(eligible_pairs) > 0:

                # tokenized_pairs = [
                #     (list(preprocess(aa, self._nlp))[0], list(preprocess(bb, self._nlp))[0])
                #     for aa, bb in eligible_pairs
                # ]
                #
                # scores = np.asarray([
                #     safe_mean(np.asarray([safe_mean(tfidf[ta]), safe_mean(tfidf[tb])]))
                #     for ta, tb in tokenized_pairs])
                #
                # scores += 1e-6
                # probs = scores / scores.sum()
                #
                # ix = self._rng.choice(len(eligible_pairs), p=probs)
                #
                # self._or = [tuple(eligible_pairs[ix]), scores[ix]]

                # similarities = vs.similarity(*zip(*eligible_pairs))  # .reshape((1,))
                #
                # similarities += 1  # This is to shift all the cosine similarities up by one to account for the negatives
                #
                # # # probs = similarities / similarities.sum()
                #
                # top_ix = np.argsort(similarities)[::-1]
                #
                # ix = self._rng.choice(len(eligible_pairs), p=probs)

                self._or = [CandidatePair(tuple(pair), 1) for pair in eligible_pairs]
            else:
                self._or = [CandidatePair(tuple(), 0.)]

        return self._or

    @property
    def singleton_entity(self) -> List[CandidateEntity]:
        """ Samples an entity based on the average TFIDF score of its words """

        if self._singleton is not None:
            return self._singleton
        else:
            # tfidf = self._tfidf_helper

            eligible_entities = [e for e in self.kg.nodes if e not in self._singleton_log]

            # Filter out eligible pairs that don't add anything new to the KG
            useful_entities = list()
            for entity in eligible_entities:
                docs, _ = self.fetch_docs(Query(entity, QueryType.Singleton))
                if len(docs - self.doc_set) > 0:
                    useful_entities.append(entity)

            eligible_entities = useful_entities

            if len(eligible_entities) > 0:

                # tokenized_entities = list(preprocess(tuple(eligible_entities), self._nlp))
                # scores = np.asarray([safe_mean(tfidf[te]) for te in tokenized_entities])
                # scores += 1e-6
                # probs = scores / scores.sum()
                #
                # top_ix = np.argsort(probs)[::-1]

                # ix = self._rng.choice(len(eligible_entities), p=probs)

                self._singleton = [CandidateEntity(entity, 1) for entity in eligible_entities]
            else:
                self._singleton = [CandidateEntity(None, 0.)]

        return self._singleton

    def shaping_potential(self) -> float:
        """ Returns the shaping potential used for reward shaping """
        total_potential = 1000
        problem = self.problem
        gt_path = problem.gt_path
        edge_potential = total_potential / len(gt_path)
        kg = self.kg
        num_present_edges = 0
        for (a, b) in gt_path:
            if (a, b) in kg.edges:
                num_present_edges += 1

        if num_present_edges == 0:
            return 0
        else:
            return edge_potential * num_present_edges


def safe_mean(vals: np.ndarray) -> float:
    return cached_mean(frozenset(vals))


@lru_cache(maxsize=2048)
def cached_mean(vals: FrozenSet[float]) -> float:
    if len(vals) == 0:
        return 0.
    else:
        return np.asarray(list(vals)).mean()
