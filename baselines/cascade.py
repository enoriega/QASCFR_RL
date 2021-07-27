import abc
from typing import Optional, List

from numpy.random import RandomState

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

    def run(self, env: QASCInstanceEnvironment) -> Optional[List[str]]:
        """ Runs the cascade baseline over the specified environment.
        The environment mutates.
        Returns path when found, otherwise None"""

        rng = self._rng
        finished = False
        path = None

        while not finished:
            and_pairs = env.get_eligible_pairs(env._and_log)
            or_pairs = env.get_eligible_pairs(env._or_log)
            # Try first with the "AND pairs". If none available, fall back to the "OR pairs"
            candidate_pairs = and_pairs if len(and_pairs) > 0 else or_pairs

            # If there are not any more candidate pairs, then exit the loop
            if len(candidate_pairs) == 0:
                finished = True
            else:
                # First execute an AND query
                ix = rng.randint(0, len(candidate_pairs))  # Randomly choose a pair of AND entities
                candidate_entities = candidate_pairs[ix]
                query = Query(candidate_entities, QueryType.And)

                # Run it through lucene
                docs, _ = env.fetch_docs(query)

                # If returned an empty set, then try OR
                if len(docs) == 0:
                    query = Query(candidate_entities, QueryType.Or)
                    docs, _ = env.fetch_docs(query)

                # Reconcile the elements with the environment
                env.add_docs(docs, query)

                # Check the status of the search
                finished, path = env.status

        return path
