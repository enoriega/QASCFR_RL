import itertools as it
import pickle
import json
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple, Set, FrozenSet, Union, List

from actions import Query


class Pair(NamedTuple):
    """ Represents the endpoints of an edge in a knowledge graph """
    src: str
    dst: str


class Edge(NamedTuple):
    """ Represents an edge on a knowledge graph and the documents of the corpus where the edge is found """
    pair: Pair
    docs: FrozenSet[str]


class QASCItem(NamedTuple):
    """ Represents a problem to be operated over in an environment by an agent """
    question: str
    answer: str
    gt_path: Sequence[str]

    @staticmethod
    def from_json_line(element: str):
        data = json.loads(element)
        choices = {c['label']: c['text'] for c in data['question']['choices']}
        facts = [data['fact1'], data['fact2']]
        question = data['question']['stem']
        answer = choices[data['answerKey']]

        return QASCItem(question, answer, tuple(facts))


class Results(NamedTuple):
    outcome: bool
    queries: Sequence[Query]
    differential_phrases: Sequence[Set[str]]
    connecting_paths: Union[Sequence[Sequence[str]], None]

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def phrases_read(self) -> int:
        return sum(len(d) for d in self.differential_phrases)


def read_problems(path: Union[Path, str]) -> List[QASCItem]:
    """
    Read the problems from the shelve file and return an iterable to operate over them
    This function might change to limit the scope of each run
    """

    # Convert it to a path for convenience
    if type(path) == str:
        path = Path(path)

    with path.open('r') as f:
        data = [QASCItem.from_json_line(line) for line in f]

    return data




if __name__ == "__main__":
    problems = read_problems("/home/enrique/Downloads/QASC_Dataset/dev.jsonl")
    print(problems)