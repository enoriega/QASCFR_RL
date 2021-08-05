from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, NamedTuple, List, Union, Optional, cast

import numpy as np
import spacy
from gensim.models import KeyedVectors
from numpy.random.mtrand import RandomState
from rlpyt.envs.base import Env
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox
from spacy.language import Language

import parsing
import utils
from actions import Query, QueryType
from machine_reading.ie import RedisWrapper
from machine_reading.ir.es import QASCIndexSearcher
from parsing import QASCItem
from environment import QASCItem, QASCInstanceEnvironment

# from nlp import EmbeddingSpaceHelper

Observation = np.ndarray

EnvInfo = namedtuple("EnvInfo", "papers outcome query_type, query_entity")


@dataclass
class QASCInstanceFactory:
    problems: List[QASCItem]
    use_embeddings: bool
    num_top_entities: int
    # use_generational_ranking: bool
    rng: RandomState
    nlp: Language
    index: QASCIndexSearcher
    redis: RedisWrapper
    vector_space: KeyedVectors

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

        # env = QASCItem(sampled, 10, self.use_embeddings, self.num_top_entities, seed, self.index, self.redis,
        #                self.vector_space)

        item = QASCItem(sampled.question, sampled.answer, sampled.gt_path)
        env = QASCInstanceEnvironment(item, 10, self.use_embeddings, self.num_top_entities, seed, self.index,
                                      self.redis, self.vector_space, self.nlp)

        return env

    @classmethod
    def from_json(cls, problems_path: Union[str, Path], use_embeddings: bool,
                  num_top_entities: int,
                  # indices_path: Union[str, Path], lda_path: Union[str, Path],
                  # corpus_path: Union[str, Path],
                  index_searcher: QASCIndexSearcher, redis_client: RedisWrapper,
                  seed: int):
        problems_path = Path(problems_path)

        problems = parsing.read_problems(problems_path)

        # index, inverted_index = utils.build_indices(indices_path)

        nlp = spacy.load("en_core_web_sm")

        # vector_space = EmbeddingSpaceHelper(nlp)
        vector_space = cast(KeyedVectors, KeyedVectors.load('data/glove.840B.300d.kv'))
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

    def __init__(self, environment_factory: QASCInstanceFactory, do_reward_shaping: bool) -> None:

        self._factory = environment_factory
        self._do_reward_shaping = do_reward_shaping

        env = environment_factory()
        # Store the number of iterations
        self._timeout = env.max_iterations
        # Store the environment
        self.instance = env

        # Set up the action and observation spaces
        self._action_space = IntBox(0, high=10)
        self._observation_space = FloatBox(-1, 1, (13,))
        pass

    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, Optional[NamedTuple]]:
        """
        Executes the query requested by the caller
        :param action: Query to be executed by the inner environment
        :return: An element of this environment’s observation space corresponding to the next state.
                 reward (float): A scalar reward resulting from the state transition.
                 done (bool): Indicates whether the episode has ended.
                 info (namedtuple): Additional custom information.
        """

        env = self.instance

        query = env.query

        # Run it through lucene
        docs = env.fetch_docs(query)

        # Reconcile the elements with the environment
        env.add_docs(docs)

        candidates = env.ranked_docs()

        chosen = candidates[action][0]

        env.add_explanation(chosen)

        # If shaping reward, observe the potential before mutating the environment
        if self._do_reward_shaping:
            prev_potential = env.shaping_potential()
        else:
            prev_potential = 0


        # Test whether if the trial is finished
        reward = env.rl_reward()
        # Update the previous score, to prepare it for the next
        env._prev_score = env.fr_score
        done = env.status

        # Shape the reward if requested
        if self._do_reward_shaping:
            potential = self.instance.shaping_potential()
            potential_diff = potential - prev_potential
            reward += potential_diff

        # Generate the environment observation
        obs = self.observe()

        env_info = EnvInfo(papers=len(env.doc_set), outcome=env.success, query_type=int(QueryType.And),
                           query_entity=int(action))

        return obs, np.float(reward), done, env_info

    def reset(self) -> np.ndarray:
        """
        Resets this environment to its initial state
        :return: Environment observation
        """

        instance = self._factory()
        self.instance = instance
        instance.reset()

        # Return an observation
        return self.observe()

    def observe(self) -> np.ndarray:
        """
        Generates an observation of the current state of the environment
        :return: representation of the current environment
        """
        return self.instance.observe
