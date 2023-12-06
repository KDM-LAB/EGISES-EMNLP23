import itertools
import os
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.set_option('mode.chained_assignment', None) # disable warning "A value is trying to be set on a copy of a slice from a DataFrame."

from egises.distance_measure import Measure, JSD, Meteor
from egises.utils import write_scores_to_csv, divide_with_exception, calculate_proportion


class Summary:
    def __init__(self, origin_model: str, doc_id, uid, summary_text):
        self.origin_model = origin_model
        self.doc_id = doc_id
        self.uid = uid
        self.summary_text = summary_text


class Document:
    def __init__(self, doc_id, doc_text, doc_summ, user_summaries: Iterable[Summary],
                 model_summaries: Iterable[Summary]):
        self.doc_id = doc_id
        self.doc_text = doc_text
        self.doc_summ = doc_summ
        self.user_summaries = user_summaries
        self.model_summaries = model_summaries
        self.summary_doc_distances = {}
        self.summary_summary_distances = {}

    def populate_summary_doc_distances(self, measure: Measure):
        # check if summary_doc_distances in
        for user_summary in self.user_summaries:
            self.summary_doc_distances[(self.doc_id, user_summary.origin_model, user_summary.uid)] = measure.distance(
                user_summary.summary_text, self.doc_text)

        for model_summary in self.model_summaries:
            self.summary_doc_distances[(self.doc_id, model_summary.origin_model, model_summary.uid)] = measure.distance(
                model_summary.summary_text, self.doc_text)
        # print(f"self.summary_doc_distances: {self.summary_doc_distances}")

    def populate_summary_summary_distances(self, measure: Measure):
        # cache summary tokens
        summ_tokens = {}
        # calculate user_summary_summary_distances
        for user_summary1, user_summary2 in itertools.permutations(self.user_summaries, 2):
            if not (user_summary1.origin_model, user_summary1.uid) in summ_tokens:
                summ_tokens[(user_summary1.origin_model, user_summary1.uid)] = measure._tokenize(
                    user_summary1.summary_text)
            if not (user_summary1.origin_model, user_summary2.uid) in summ_tokens:
                summ_tokens[(user_summary1.origin_model, user_summary2.uid)] = measure._tokenize(
                    user_summary2.summary_text)

            self.summary_summary_distances[
                (self.doc_id, user_summary1.origin_model, user_summary1.uid, user_summary2.uid)] = measure.distance(
                summ_tokens[(user_summary1.origin_model, user_summary1.uid)],
                summ_tokens[(user_summary2.origin_model, user_summary2.uid)])

        # calculate model_summary_summary_distances
        for model_summary1, model_summary2 in itertools.permutations(self.model_summaries, 2):
            if not (model_summary1.origin_model, model_summary1.uid) in summ_tokens:
                summ_tokens[(model_summary1.origin_model, model_summary1.uid)] = measure._tokenize(
                    model_summary1.summary_text)
            if not (model_summary2.origin_model, model_summary2.uid) in summ_tokens:
                summ_tokens[(model_summary2.origin_model, model_summary2.uid)] = measure._tokenize(
                    model_summary2.summary_text)

            self.summary_summary_distances[
                (self.doc_id, model_summary1.origin_model, model_summary1.uid, model_summary2.uid)] = measure.distance(
                summ_tokens[(model_summary1.origin_model, model_summary1.uid)],
                summ_tokens[(model_summary2.origin_model, model_summary2.uid)])
        # print(f"self.summary_summary_distances: {self.summary_summary_distances}")


class Egises:
    def __init__(self, model_name, measure: Measure, documents: Iterable[Document], score_directory=""):
        self.model_name = model_name
        if not score_directory:
            self.score_directory = f"{measure.name}/{model_name}"
        else:
            self.score_directory = score_directory
        if not os.path.exists(f"{self.score_directory}"):
            # create directory
            os.makedirs(f"{self.score_directory}")

        self.summary_doc_score_path = f"{self.score_directory}/sum_doc_distances.csv"
        self.summ_summ_score_path = f"{self.score_directory}/sum_sum_doc_distances.csv"

        self.measure = measure
        self.documents = documents
        self.summary_doc_score_df = None
        self.summ_pair_score_df = None

    def populate_distances(self):
        if os.path.exists(self.summary_doc_score_path) and os.path.exists(self.summ_summ_score_path):
            pass
        else:
            summary_doc_tuples = []
            summ_pair_tuples = []
            for document in self.documents:
                document.populate_summary_doc_distances(self.measure)
                summary_doc_tuples.extend([(*k, v) for k, v in document.summary_doc_distances.items()])
                # print(f"self.summary_doc_tuples: {self.summary_doc_tuples}")
                document.populate_summary_summary_distances(self.measure)
                summ_pair_tuples.extend([(*k, v) for k, v in document.summary_summary_distances.items()])
                # print(f"self.summ_pair_tuples: {self.summ_pair_tuples}")

            write_scores_to_csv(summary_doc_tuples, fields=("doc_id", "origin_model", "uid", "score"),
                                filename=self.summary_doc_score_path)
            write_scores_to_csv(summ_pair_tuples, fields=("doc_id", "origin_model", "uid1", "uid2", "score"),
                                filename=self.summ_summ_score_path)

        self.summary_doc_score_df = pd.read_csv(self.summary_doc_score_path)
        self.summ_pair_score_df = pd.read_csv(self.summ_summ_score_path)
        # print(f"len(self.summary_doc_tuples): {len(self.summary_doc_tuples)}")
        # print(f"len(self.summ_pair_tuples): {len(self.summ_pair_tuples)}")

    def get_user_model_X_scores(self, model_name):
        usum_scores_df = self.summary_doc_score_df[self.summary_doc_score_df["origin_model"] == model_name]
        # TODO: rename to mpair_scores_df
        upair_scores_df = self.summ_pair_score_df[self.summ_pair_score_df["origin_model"] == model_name]

        usum_scores_df = usum_scores_df.set_index(["doc_id", "uid"])
        sum_doc_score_dict = {k: v["score"] for k, v in usum_scores_df.to_dict(orient="index").items()}

        # step2: get ratio of summary_summary_distance to summary_doc_distance
        # w(u_ij) = distance(ui,uj)/sum(distance(ui,doc))

        upair_scores_df["pair_score_weight"] = upair_scores_df.apply(
            lambda x: divide_with_exception(x["score"], sum_doc_score_dict[(x["doc_id"], x["uid1"])]), axis=1)

        # step 3: calculate softmax of pair_score_weight grouped by doc_id, uid1
        # softmax(w(u_ij)) = exp(w(u_ij))/sum(exp(w(u_il))) where l is all users who summarized doc i
        upair_scores_df["pair_score_weight_exp"] = upair_scores_df.apply(lambda x: np.exp(x["pair_score_weight"]),
                                                                         axis=1)
        upair_scores_df["pair_score_weight_exp_softmax"] = upair_scores_df.groupby(["doc_id", "uid1"])[
            "pair_score_weight_exp"].transform(lambda x: x / sum(x))

        upair_scores_df["final_score"] = upair_scores_df.apply(
            lambda x: round(x["pair_score_weight_exp_softmax"] * x["score"], 4), axis=1)
        # keep only doc_id, uid1, uid2, final_score
        final_df = upair_scores_df[["doc_id", "uid1", "uid2", "final_score"]]
        return final_df

    def get_egises_score(self, sample_percentage=100):
        # print(f"populating distances")
        self.populate_distances()
        user_X_df = self.get_user_model_X_scores(model_name="user")
        model_Y_df = self.get_user_model_X_scores(model_name=self.model_name)
        # sample sample_percentage% of model_Y_df
        model_Y_df = model_Y_df.sample(frac=sample_percentage / 100)
        user_X_df = user_X_df.set_index(["doc_id", "uid1", "uid2"])
        user_X_score_map = user_X_df.to_dict(orient="index")

        # calculate min/max on model_Y_df["final_score"] and user_X_score_map[(doc_id,uid1,uid2))]
        model_Y_df["proportion"] = model_Y_df.apply(lambda x: calculate_proportion(x.final_score, user_X_score_map[
            (x["doc_id"], x["uid1"], x["uid2"])]["final_score"]), axis=1)

        # calculate mean of proportion column
        model_Y_df["mean_proportion"] = model_Y_df.groupby(["doc_id", "uid1"])["proportion"].transform(
            lambda x: np.mean(x))
        # find mean of mean_proportion column
        return round(1 - model_Y_df['mean_proportion'].mean(), 4)
