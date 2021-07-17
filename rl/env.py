from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, NamedTuple, List, Union, Mapping, Optional

import numpy as np
import spacy
from numpy.random.mtrand import RandomState
from rlpyt.envs.base import Env
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox
from spacy.language import Language

import parsing
import utils
from actions import Query, QueryType
from machine_reading.ie import RedisWrapper
from machine_reading.ir import QASCIndexSearcher
from parsing import QASCInstance
from environment import Environment
from nlp import EmbeddingSpaceHelper
from tfidf import TfIdfHelper
from topic_modeling import TopicsHelper

Observation = np.ndarray

EnvInfo = namedtuple("EnvInfo", "papers outcome query_type, query_entity")


@dataclass
class EnvironmentFactory:
    problems: List[QASCInstance]
    use_embeddings: bool
    num_top_entities: int
    # use_generational_ranking: bool
    rng: RandomState
    nlp: Language
    index: QASCIndexSearcher
    redis: RedisWrapper
    vector_space: EmbeddingSpaceHelper
    # topics: TopicsHelper
    # tfidf: TfIdfHelper
    # index: Mapping
    # inverted_index: Mapping

    def __post_init__(self) -> None:

        # Make a shuffled copy of the problems which is our sampling order computed a priori
        shuffled_problems = list(self.problems)
        self.rng.shuffle(shuffled_problems)
        self._shuffled_problems = shuffled_problems

    def __call__(self, *args, **kwargs):

        # Pop an element from the shuffled problems, but if they're empty, repeat the shuffling from the originals
        if len(self._shuffled_problems) == 0:
            self.__post_init__()

        sampled = self._shuffled_problems.pop()

        seed = self.rng.randint(0, int(1e6), size=1)

        env = Environment(sampled, 10, self.use_embeddings, self.num_top_entities, seed, self.index, self.redis, self.vector_space)

        return env

    @classmethod
    def from_json(cls, problems_path: Union[str, Path], use_embeddings: bool,
                   num_top_entities: int,
                   # indices_path: Union[str, Path], lda_path: Union[str, Path],
                   # corpus_path: Union[str, Path],
                   index_searcher:QASCIndexSearcher, redis_client:RedisWrapper,
                   seed: int):

        problems_path = Path(problems_path)

        problems = parsing.read_problems(problems_path)

        # index, inverted_index = utils.build_indices(indices_path)

        nlp = spacy.load("en_core_web_lg")

        vector_space = EmbeddingSpaceHelper(nlp)
        # topics_helper = TopicsHelper.from_shelf(lda_path)
        # tfidf_helper = TfIdfHelper.build_tfidf(corpus_path)
        rng = utils.build_rng(seed)

        factory = cls(problems, use_embeddings, num_top_entities, rng, nlp, index_searcher, redis_client, vector_space)

        return factory


class RlpytEnv(Env):
    """ Wraps our Environment class into an rlpyt's environment to be used for RL"""

    @property
    def horizon(self) -> int:
        return self._timeout

    def __init__(self, environment_factory: EnvironmentFactory, do_reward_shaping: bool) -> None:

        self._factory = environment_factory
        self._do_reward_shaping = do_reward_shaping

        env = environment_factory()
        # Store the number of iterations
        self._timeout = env.max_iterations
        # Store the environment
        self.env = env

        # Set up the action and observation spaces
        self._action_space = IntBox(0, high=((len(QueryType) - 1)*env.num_top_entities))
        self._observation_space = FloatBox(-1, 1, (5,))

    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, Optional[NamedTuple]]:
        """
        Executes the query requested by the caller
        :param action: Query to be executed by the inner environment
        :return: An element of this environment’s observation space corresponding to the next state.
                 reward (float): A scalar reward resulting from the state transition.
                 done (bool): Indicates whether the episode has ended.
                 info (namedtuple): Additional custom information.
        """

        env = self.env

        entity_ix = action % env.num_top_entities
        type_code = action // env.num_top_entities

        # action = action.reshape((1,))
        # Map the action from int to query
        # type_ = QueryType(action[0] + 1)
        type_ = QueryType(type_code + 1)
        # Select the entities form the environment
        if type_ == QueryType.And:
            if entity_ix < len(env.and_entities):
                endpoints = tuple(env.and_entities[entity_ix].pair)
            else:
                endpoints = tuple()
        elif type_ == QueryType.Or:
            if entity_ix < len(env.or_entities):
                endpoints = tuple(env.or_entities[entity_ix].pair)
            else:
                endpoints = tuple()
        elif type_ == QueryType.Singleton:
            if entity_ix < len(env.singleton_entity):
                endpoints = env.singleton_entity[entity_ix].entity
            else:
                endpoints = None
        else:
            raise RuntimeError("Unsupported query type")

        query = Query(endpoints, type_)

        # If shaping reward, observe the potential before mutating the environment
        if self._do_reward_shaping:
            prev_potential = self.env.shaping_potential()
        else:
            prev_potential = 0

        # Fetch the incremental documents from the query
        new_docs, realized_query_type = env.fetch_docs(query)
        # Add them to the environment
        realized_query = Query(query.endpoints, realized_query_type)
        env.add_docs(new_docs, query=realized_query)

        # Test whether if the trial is finished
        done, reward = self.env.rl_reward()

        # Shape the reward if requested
        if self._do_reward_shaping:
            potential = self.env.shaping_potential()
            potential_diff = potential - prev_potential
            reward += potential_diff

        # Generate the environment observation
        obs = self.observe()

        env_info = EnvInfo(papers=len(new_docs), outcome=bool(env), query_type=type_.value, query_entity=entity_ix)

        return obs, np.float(reward), done, env_info

    def reset(self) -> np.ndarray:
        """
        Resets this environment to its initial state
        :return: Environment observation
        """

        env = self._factory()
        self.env = env
        env.reset()

        # Return an observation
        return self.observe()

    def observe(self) -> np.ndarray:
        """
        Generates an observation of the current state of the environment
        :return: representation of the current environment
        """
        return self.env.observe

