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


class AirAgent(Agent):

    # def __init__(self, rng:RandomState) -> None:
    #     self.rng = rng

    def run(self, env: QASCInstanceEnvironment) -> Tuple[List[str], bool]:
        """ Runs the cascade baseline over the specified environment.
        The environment mutates.
        Returns path when found, otherwise None"""

        rng = self._rng
        finished = False
        path = None

        # Run it through lucene
        docs = {h.text for h in env.doc_universe}  # env.fetch_docs(remaining)

        # Reconcile the elements with the environment
        env.add_docs(docs)

        prev_cov = 0.
        while not finished:

            remaining = env.remaining
            if len(env.explanation) == 0:
                coverage = 0.
            else:
                coverage = nlp.air_coverage(env.query,
                                                nlp.preprocess(env.explanation, env._language, stem=True))

                if len(remaining) <= 4:
                    env.expand_query()
                    query = env.query

            ranked = env.ranked_docs()

            if len(ranked) > 0:
                exploit_sentence = ranked[-1][0]
                explore_sentence = ranked[0][0]

                # exploit_coverage = nlp.air_coverage(remaining,
                #                                     nlp.preprocess(env.explanation + [exploit_sentence], env._language, stem=True))
                # explore_coverage = nlp.air_coverage(remaining,
                #                                     nlp.preprocess(env.explanation + [explore_sentence], env._language, stem=True))


                #if exploit_coverage > explore_coverage:
                if True:
                    env.add_explanation(exploit_sentence)
                else:
                    env.add_explanation(explore_sentence)


            # Check the status of the search
            finished = env.status
            outcome = env.fr_score #env.success
            path = env.explanation

            # Increment the iteration counter
            env.iterations += 1

            # env.remaining()

        return path, outcome
