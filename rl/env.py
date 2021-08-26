from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, NamedTuple, List, Union, Optional

import numpy as np
import spacy
from gensim.models import KeyedVectors
from numpy.random.mtrand import RandomState
from rlpyt.envs.base import Env
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox
from spacy.language import Language

import nlp
import parsing
import utils
from environment import QASCItem, QASCInstanceEnvironment
from machine_reading.ie import RedisWrapper
from machine_reading.ir.es import QASCIndexSearcher
from nlp import load_embeddings

# from nlp import EmbeddingSpaceHelper
from rl.aux import gt_match_type, RecallType

Observation = np.ndarray

EnvInfo = namedtuple("EnvInfo", "chosen_rank outcome partial_recall total_recall explanation_size")


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

        item = QASCItem(sampled.question, sampled.answer, sampled.gt_path, sampled.facts)
        env = QASCInstanceEnvironment(item, 10, self.use_embeddings, self.num_top_entities, seed, self.index,
                                      self.redis, self.vector_space, self.nlp, sampled.facts)

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
        vector_space = load_embeddings('data/glove.840B.300d.kv')
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

        remaining = env.remaining

        if len(env.explanation) == 0:
            coverage = 0.
        else:
            coverage = nlp.air_coverage(env.query,
                                        nlp.preprocess(env.explanation, env._language, stem=True))

            if len(remaining) <= 4:
                env.expand_query()
                query = env.query

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
        env._prev_score = env.fr_score(normalize=True)
        done = env.status
        outcome = env.fr_score()  # env.success
        path = env.explanation
        env.iterations += 1

        # Shape the reward if requested
        if self._do_reward_shaping:
            potential = self.instance.shaping_potential()
            potential_diff = potential - prev_potential
            reward += potential_diff

        # Generate the environment observation
        obs = self.observe()

        # Do some observation logging
        item = env.item

        partial_recall = False
        total_recall = False

        recall_type = gt_match_type(item, env.explanation)
        if recall_type == RecallType.Total:
            partial_recall = True
            total_recall = True
        elif recall_type == RecallType.Partial:
            partial_recall = True

        explanation_size = len(env.explanation)

        env_info = EnvInfo(outcome=env.success, chosen_rank=action,
                           partial_recall = partial_recall, total_recall = total_recall,
                           explanation_size = explanation_size)

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
