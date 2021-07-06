import shelve
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union, Sequence, FrozenSet, Iterable

import numpy as np
import pandas as pd

TopicDist = List[Tuple[int, float]]
Topic = int


class TopicsHelper:
    """ Helper to fetch the topic models from the precomputed LDA frame """

    def __init__(self, lda, topic_frame: pd.DataFrame, lda_dict):
        self.lda = lda
        self.topic_frame = topic_frame.set_index('id')
        self.lda_dict = lda_dict
        self.num_topics = topic_frame.top_topic.max().astype(int) + 1

    @classmethod
    def from_shelf(cls, frame_path: Union[str, Path]):
        with shelve.open(str(frame_path)) as db:
            lda = db['lda_tfidf']
            topic_frame = db['frame']
            lda_dict = db['dictionary']

        return cls(lda, topic_frame, lda_dict)

    def get_topics_for_doc(self, doc: Union[str, Iterable[str]]) -> TopicDist:

        if type(doc) == str:
            doc = [doc]

        fr = self.topic_frame
        topics = fr.loc[doc, 'topics']
        return topics

    def get_top_topic_for_doc(self, doc: Union[str, Sequence[str]]) -> Topic:
        if type(doc) == str:
            doc = [doc]
        fr = self.topic_frame
        topic = fr.loc[doc, 'top_topic']
        return topic

    @lru_cache(maxsize=512)
    def compute_topic_dist(self, docs: FrozenSet[str]) -> np.ndarray:
        num_topics = self.num_topics

        dist = np.zeros(num_topics) + 1e-6  # Basic smoothing to be able to compute the KL divergence downstream

        topics = self.get_topics_for_doc(docs)
        for t in topics:
            for topic, score in t:
                dist[topic] += score

        # Normalize the distribution
        dist /= dist.sum()
        return dist
