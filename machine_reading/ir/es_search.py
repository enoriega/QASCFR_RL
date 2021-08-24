import logging
from math import ceil, floor
from time import time
from typing import Dict, List

from elasticsearch import Elasticsearch
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class EsHit:
    def __init__(self, score: float, position: int, text: str, type: str, index: str):
        """
        Basic information about an ElasticSearch Hit
        :param score: score returned by the query
        :param position: position in the retrieved results (before any filters are applied)
        :param text: retrieved sentence
        :param type: type of the hit in the index (by default, only documents of type "sentence"
        will be retrieved from the index)
        """
        self.score = score
        self.position = position
        self.text = text
        self.type = type
        self.index = index


    def __repr__(self):
        return f"{self.text} - {self.score}"


class EsSearch:
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "arc",
                 max_question_length: int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100,
                 split_qa: bool = False,
                 twostep_qa: bool = False,
                 get_chains: bool = False,
                 maxq_hits: int = 10,
                 dfs_query: bool = False,
                 use_diff: bool = False,
                 use_order: bool = False,
                 use_chain_order: bool = False,
                 output_all: bool = False,
                 must_match_diff_indices: bool = True,
                 must_match_qa: bool = False):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        :param max_question_length: Max number of characters used from the question for the
        query (for efficiency)
        :param max_hits_retrieved: Max number of hits requested from ElasticSearch
        :param max_hit_length: Max number of characters for accepted hits
        :param max_hits_per_choice: Max number of hits returned per answer choice
        :param split_qa Split into two queries (question and answer choice) and concatenate results
        """
        self._es = Elasticsearch([es_client], retries=3, timeout=180)
        tracer = logging.getLogger('elasticsearch')
        tracer.setLevel(logging.CRITICAL)
        self._indices = indices
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]
        if split_qa and twostep_qa:
            raise ValueError("Both split_qa and twostep_qa can't be true! Set one to false.")
        if get_chains and not twostep_qa:
            raise ValueError("get_chains is set to true but twostep_qa not set to true!"
                             "Set both to true (or false).")
        self._split_qa = split_qa
        self._twostep_qa = twostep_qa
        if dfs_query:
            self._search_type = "dfs_query_then_fetch"
        else:
            self._search_type = "query_then_fetch"
        if self._twostep_qa:
            self._stemmer = PorterStemmer()
            self._stop_words_set = set(stopwords.words('english'))
            self._maxq_hits = maxq_hits
            self._get_chains = get_chains
            self._use_diff = use_diff
            self._use_order = use_order
            self._use_chain_order = use_chain_order
            self._must_match_qa = must_match_qa
            self._must_match_diff_indices = must_match_diff_indices
            self._output_all = output_all

    def tokenize_str(self, input_str: str):
        tokens = [(self._stemmer.stem(str), str) for str in re.split("[\W_]+", input_str.lower())
                  if str not in self._stop_words_set and len(str) > 0]
        return tokens

    def has_overlap(self, f1: str, f2: str) -> bool:
        f1_tokens = self.tokenize_str(f1)
        f2_tokens = self.tokenize_str(f2)
        if len(f1_tokens) == 0:
            return True
        if len(f2_tokens) == 0:
            return False
        f1_stems = list(zip(*f1_tokens))[0]
        f2_stems = list(zip(*f2_tokens))[0]
        intersection = set(f1_stems).intersection(set(f2_stems))
        return len(intersection) > 0

    def fact_delta(self, fact: str, question: str) -> str:
        fact_tokens = self.tokenize_str(fact)
        question_tokens = self.tokenize_str(question)
        if len(fact_tokens) == 0:
            return ""
        fact_stems = list(zip(*fact_tokens))[0]
        if len(question_tokens) == 0:
            stem_diff = set(fact_stems)
        else:
            ques_stems = list(zip(*question_tokens))[0]
            stem_diff = set(fact_stems) - set(ques_stems)
        output_toks = []
        for (fstem, ftok) in fact_tokens:
            if fstem in stem_diff:
                output_toks.append(ftok)

        return " ".join(output_toks).strip()

    def get_hits_for_question(self, question: str, choices: List[str],
                              optional_terms: str = None,
                              input_hits: List[EsHit] = None) -> Dict[str, List[EsHit]]:
        """
        :param question: Question text
        :param choices: List of answer choices
        :return: Dictionary of hits per answer choice
        """
        choice_hits = dict()
        question_only_hits = None
        if self._twostep_qa and not self._use_diff and input_hits is None:
            if self._use_order:
                index = self._indices.split(",")[0]
            else:
                index = self._indices
            question_only_hits = self.filter_hits(self.get_hits_for_choice(
                question=question,
                choice=None,
                indices=index,
                optional_terms=optional_terms),
                max_hits=self._maxq_hits)
        for choice in choices:
            if self._split_qa:
                if optional_terms:
                    curr_optional_terms = optional_terms + " " + choice
                else:
                    curr_optional_terms = choice
                question_only_hits = self.filter_hits(self.get_hits_for_choice(question, None,
                                                                               optional_terms=curr_optional_terms))
                if optional_terms:
                    curr_optional_terms = optional_terms + " " + question
                else:
                    curr_optional_terms = question

                choice_only_hits = self.filter_hits(self.get_hits_for_choice(choice, None,
                                                                             optional_terms=curr_optional_terms))
                choice_hits[choice] = question_only_hits + choice_only_hits
            elif self._twostep_qa:
                question_choices_hits = []
                counter = 0
                main_query = question + " " + choice
                if self._use_diff and input_hits is None:
                    if optional_terms:
                        curr_optional_terms = main_query + " " + optional_terms
                    else:
                        curr_optional_terms = main_query
                        if self._use_order:
                            index = self._indices.split(",")[0]
                        else:
                            index = self._indices
                    # Get hits based on the question + answer choice
                    question_only_hits = self.filter_hits(
                        self.get_hits_for_choice(None, None, optional_terms=curr_optional_terms,
                                                 indices=index),
                        max_hits=self._maxq_hits)
                if input_hits is not None:
                    question_only_hits = input_hits
                elif question_only_hits is None:
                    raise ValueError("question_only_hits not set!")

                # For each hit retrieved in the first step
                for question_hit in question_only_hits:
                    # to get at least 2x the number of hits
                    if counter >= 2 * self._max_hits_per_choice:
                        break
                    if self._use_diff:
                        # Identify words from the HIT not found in the question+choice query
                        delta1 = self.fact_delta(question_hit.text, main_query)
                        # Identify words from the question+choice query not found in the HIT
                        delta2 = self.fact_delta(main_query, question_hit.text)
                        # Ignore HITs where few new terms are introduced
                        if len(delta1) < 2:
                            continue
                        curr_optional_terms = optional_terms
                    else:
                        delta1 = choice
                        delta2 = self.fact_delta(question_hit.text, question)
                        if len(delta2) < 2:
                            continue
                        if optional_terms:
                            curr_optional_terms = optional_terms + " " + question
                        else:
                            curr_optional_terms = question
                    
                    index_names = self._indices.split(",")
                    if len(index_names) > 1 and self._must_match_diff_indices:
                        if question_hit.index not in index_names:
                            print("Input hit from an index: {} not in the input list: {}".format(
                                question_hit.index, index_names
                            ))
                        else:
                            index_names.remove(question_hit.index)
                        new_indices = ",".join(index_names)
                    else:
                        new_indices = self._indices
                    # Get 4 hits per question hit to ensure diversity
                    max_choice_hits = floor(max(4,  2 * self._max_hits_per_choice/len(question_only_hits)))
                    # Require the retrieved HITS should match some part of the question and answer
                    if self._must_match_qa:
                        # first must match the two term deltas
                        must_match_list = [delta1, delta2]
                        # must match question words not in this HIT
                        if not self.has_overlap(question, question_hit.text):
                            must_match_list.append(question)
                        # must match answer words not in this HIT
                        if not self.has_overlap(choice, question_hit.text):
                            must_match_list.append(choice)
                        query = delta1 + " " + delta2
                        choice_only_hits = self.filter_hits(
                            self.get_hits_for_query(new_indices,
                                                    body=self.construct_query(query,
                                                                              must_match_list)),
                            max_hits=max_choice_hits)
                    else:
                        choice_only_hits = self.filter_hits(
                            self.get_hits_for_choice(delta1, delta2, indices=new_indices,
                                                     optional_terms=curr_optional_terms),
                            max_hits=max_choice_hits)
                    for choice_hit in choice_only_hits[:max_choice_hits]:
                        # Add upto max_choice_hits for each question+choice
                        question_choices_hits.append((question_hit, choice_hit))
                        counter += 1
                # sort HIT pairs by the sum of the IR scores
                sorted_hits = sorted(question_choices_hits,
                                     key=lambda x: -(x[0].score + x[1].score))
                output_hits = []
                added_hits = {}
                if self._get_chains:
                    if self._output_all:
                        choice_hits[choice] = sorted_hits
                    else:
                        choice_hits[choice] = sorted_hits[:self._max_hits_per_choice]
                else:
                    # Find unique hits from the HIT pairs
                    for question_hit, choice_hit in sorted_hits:
                        if question_hit.text not in added_hits:
                            added_hits[question_hit.text] = len(output_hits)
                            output_hits.append(question_hit)
                        if len(output_hits) >= self._max_hits_per_choice and not self._output_all:
                            break
                        if choice_hit.text not in added_hits:
                            added_hits[choice_hit.text] = len(output_hits)
                            output_hits.append(choice_hit)

                        if len(output_hits) >= self._max_hits_per_choice:
                            break
                    if not self._use_chain_order:
                        output_hits.sort(key=lambda x: -x.score)
                    if self._output_all:
                        choice_hits[choice] = output_hits
                    else:
                        # Select max_hits_per_choice sentences for each answer choice
                        choice_hits[choice] = output_hits[:self._max_hits_per_choice]
            else:
                choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice,
                                                                                optional_terms=optional_terms))
        return choice_hits

    # Constructs an ElasticSearch query from the input question and choice
    # Uses the last self._max_question_length characters from the question and requires that the
    # text matches the answer choice and the hit type is a "sentence"
    def construct_qa_query(self, question, choice, optional_terms=None):
        if question:
            question_substr = question[-self._max_question_length:]
        else:
            question_substr = ""

        if choice:
            query = question_substr + " " + choice
        else:
            query = question_substr

        if optional_terms:
            query = query + " " + optional_terms[-self._max_question_length:]
        req_terms = []
        if choice:
            req_terms.append(choice)
        if question:
            req_terms.append(question_substr)
        return self.construct_query(query, req_terms)

    def construct_query(self, query, required_terms):
        body = {
            "from": 0,
            "size": self._max_hits_retrieved,
            "query": {
                "bool": {
                    "must": [
                        {"match": {
                            "text": query
                        }}
                    ],
                    "filter": [
                        {"type": {"value": "sentence"}}
                    ]
                }
            }
        }
        for term in required_terms:
            body["query"]["bool"]["filter"].append({"match": {"text": term}})
        return body

    # Retrieve unfiltered hits for input question and answer choice
    def get_hits_for_choice(self, question, choice, indices=None, optional_terms=None):
        if indices is None:
            indices = self._indices
        hits = self.get_hits_for_query(indices,
                                       self.construct_qa_query(question, choice, optional_terms))
        if len(hits) == 0:
            print("No hits for query: {}, {}, {}, {}".format(question, choice, indices,
                                                             optional_terms))
        return hits

    def get_hits_for_query(self, indices, body):
        res = self._es.search(index=indices,
                              body=body,
                              search_type=self._search_type)
        if res["timed_out"]:
            raise ValueError("ElasticSearch timed out!!")
        # print(time() - start)
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit["_score"],
                           position=idx,
                           text=es_hit["_source"]["text"],
                           type=es_hit["_type"],
                           index=es_hit["_index"])
            hits.append(es_hit)

        return hits

    # Remove hits that contain negation, are too long, are duplicates, are noisy.
    def filter_hits(self, hits: List[EsHit], max_hits: int = -1) -> List[EsHit]:
        filtered_hits = []
        selected_hit_keys = set()
        if max_hits == -1:
            max_hits = self._max_hits_per_choice
        for hit in hits:
            hit_sentence = hit.text
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            if len(hit_sentence) > self._max_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:max_hits]

    # Check if the sentence is not noisy
    def is_clean_sentence(self, s):
        # must only contain expected characters, should be single-sentence and only uses hyphens
        # for hyphenated words
        return (re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) and
                not re.match(".*\D\. \D.*", s) and
                not re.match(".*\s\-\s.*", s))

    # Create a de-duplication key for a HIT
    def get_key(self, hit):
        # Ignore characters that do not effect semantics of a sentence and URLs
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '', re.sub('http[^ ]+', '', hit)).strip().rstrip(".")
