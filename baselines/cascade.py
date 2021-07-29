import abc
from typing import Optional, List, Tuple

from numpy.random import RandomState

import nlp
from actions import Query, QueryType
from environment import QASCInstanceEnvironment


class Agent(abc.ABC):
    """ Abstract base class for all the agent implementations """

    def __init__(self, rng: RandomState):
        """ Pass a seed RandomState for reproducibility """
        self._rng = rng


class CascadeAgent(Agent):

    # def __init__(self, rng:RandomState) -> None:
    #     self.rng = rng

    def run(self, env: QASCInstanceEnvironment) -> Tuple[List[str], bool]:
        """ Runs the cascade baseline over the specified environment.
        The environment mutates.
        Returns path when found, otherwise None"""

        rng = self._rng
        finished = False
        path = None

        prev_cov = 0.
        while not finished:

            query = env.query

            # Run it through lucene
            docs = env.fetch_docs(query)

            # Reconcile the elements with the environment
            env.add_docs(docs)

            ranked = env.ranked_docs()

            exploit_sentence = ranked[-1][0]
            explore_sentence = ranked[0][0]

            exploit_coverage = nlp.air_coverage(query,
                                                nlp.preprocess(env.explanation + [exploit_sentence], env._language))
            explore_coverage = nlp.air_coverage(query,
                                                nlp.preprocess(env.explanation + [explore_sentence], env._language))

            #if exploit_coverage > explore_coverage:
            if True:
                env.add_explanation(exploit_sentence)
                prev_cov = exploit_coverage
            else:
                env.add_explanation(explore_sentence)
                prev_cov = explore_coverage


            # Check the status of the search
            finished = env.status
            outcome = env.success
            path = env.explanation

        return path, outcome
