import concurrent
import itertools
import os
import typing
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

pd.set_option('mode.chained_assignment',
              None)  # disable warning "A value is trying to be set on a copy of a slice from a DataFrame."

from egises.distance_measure import Measure, JSD, Meteor
from egises.utils import write_scores_to_csv, divide_with_exception, calculate_proportion


class Summary:
    def __init__(self, origin_model: str, doc_id, uid, summary_text):
        self.origin_model = origin_model
        self.doc_id = doc_id
        self.uid = uid
        self.summary_text = summary_text

    def __repr__(self):
        return f"Summary(origin_model='{self.origin_model}', doc_id='{self.doc_id}', uid='{self.uid}', summary_text='{self.summary_text}')"


class Document:
    def __init__(self, doc_id, doc_text, doc_summ, user_summaries: Iterable[Summary],
                 model_summaries: Iterable[Summary]):
        self.doc_id = doc_id
        self.doc_text = doc_text
        self.doc_summ = doc_summ
        self.user_summaries = user_summaries
        self.model_summaries = model_summaries
        self.summary_doc_distances = {}
        self.summary_summary_distances = {}  # deviation of user summaries
        self.summary_user_distances = {}  # accuracy of model summaries

    def __repr__(self):
        return f"Document(doc_id='{self.doc_id}', doc_summ='{self.doc_summ}', doc_text='{self.doc_text[:50]}...')"

    def populate_summary_doc_distances(self, measure: typing.Callable, max_workers=1):
        # check if summary_doc_distances in
        ukeys = [(self.doc_id, user_summary.origin_model, user_summary.uid) for user_summary in self.user_summaries]
        uargs = [(user_summary.summary_text, f"{self.doc_summ} {self.doc_text}") for user_summary in
                 self.user_summaries]
        mkeys = [(self.doc_id, model_summary.origin_model, model_summary.uid) for model_summary in
                 self.model_summaries]
        margs = [(model_summary.summary_text, f"{self.doc_summ} {self.doc_text}") for model_summary in
                 self.model_summaries]
        # print(f"uargs: {uargs}")
        if max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map the function to the data, distributing the workload among processes
                results = list(executor.map(measure, uargs + margs))
        else:
            results = list(map(measure, uargs + margs))
        self.summary_doc_distances = {k: v for k, v in zip(ukeys + mkeys, results)}

        # for user_summary in self.user_summaries:
        #     self.summary_doc_distances[(self.doc_id, user_summary.origin_model, user_summary.uid)] = measure((
        #         user_summary.summary_text, f"{self.doc_summ} {self.doc_text}"))
        #
        # for model_summary in self.model_summaries:
        #     self.summary_doc_distances[(self.doc_id, model_summary.origin_model, model_summary.uid)] = measure((
        #         model_summary.summary_text, f"{self.doc_summ} {self.doc_text}"))
        # # print(f"self.summary_doc_distances: {self.summary_doc_distances}")

    def populate_summary_summary_distances(self, measure: typing.Callable, max_workers=1):
        # calculate user_summary_summary_distances
        keys = [(self.doc_id, user_summary1.origin_model, user_summary1.uid, user_summary2.uid) for
                user_summary1, user_summary2 in itertools.permutations(self.user_summaries, 2)]
        m_keys = [(self.doc_id, model_summary1.origin_model, model_summary1.uid, model_summary2.uid) for
                  model_summary1, model_summary2 in itertools.permutations(self.model_summaries, 2)]
        su_keys = [(self.doc_id, model_summary.origin_model, model_summary.uid) for model_summary in
                   self.model_summaries]

        # get user generated summaries into a dictionary
        user_summary_dict = {(summary.doc_id, summary.uid): summary for summary in self.user_summaries}

        res_args = [(user_summary1.summary_text, user_summary2.summary_text) for user_summary1, user_summary2 in
                    itertools.permutations(self.user_summaries, 2)]
        m_res_args = [(model_summary1.summary_text, model_summary2.summary_text) for model_summary1, model_summary2 in
                      itertools.permutations(self.model_summaries, 2)]
        su_args = [(model_summary.summary_text, user_summary_dict[model_summary.doc_id, model_summary.uid].summary_text)
                   for model_summary in self.model_summaries]
        if max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map the function to the data, distributing the workload among processes
                results = list(executor.map(measure, res_args + m_res_args + su_args))
        else:
            results = list(map(measure, res_args + m_res_args + su_args))

        self.summary_summary_distances = {k: v for k, v in zip(keys + m_keys, results[:len(keys + m_keys)])}
        self.summary_user_distances = {k: v for k, v in zip(su_keys, results[len(keys + m_keys):])}


class Egises:
    def __init__(self, model_name, measure: typing.Callable, documents: Iterable[Document], score_directory="",
                 max_workers=1, debug_flag=False):
        self.model_name = model_name
        if not score_directory:
            self.score_directory = f"{measure.__name__}/{model_name}"
        else:
            self.score_directory = score_directory
        if not os.path.exists(f"{self.score_directory}"):
            # create directory
            os.makedirs(f"{self.score_directory}")
            print(f"created directory: {self.score_directory}")
        else:
            print(f"directory already exists: {self.score_directory}")

        self.max_workers = max_workers
        self.debug_flag = debug_flag
        self.summary_doc_score_path = f"{self.score_directory}/sum_doc_distances.csv"
        self.summ_summ_score_path = f"{self.score_directory}/sum_sum_doc_distances.csv"
        self.sum_user_score_path = f"{self.score_directory}/sum_user_distances.csv"

        self.measure = measure
        self.documents = documents
        self.summary_doc_score_df = None
        self.summ_pair_score_df = None

    def populate_distances(self):
        last_seen, last_doc_processed = False, None
        if os.path.exists(self.summary_doc_score_path) and os.path.exists(self.summ_summ_score_path):
            summary_doc_score_df = pd.read_csv(self.summary_doc_score_path)
            # find last doc_id in summary_doc_distances
            last_doc_processed = summary_doc_score_df.iloc[-1]["doc_id"]
        pbar = tqdm(total=3840, desc="Populating Distances")
        for document in self.documents:

            # find last doc_id in summary_doc_distances
            if not last_seen and last_doc_processed and document.doc_id != last_doc_processed:
                pbar.update(1)
                continue
            elif last_doc_processed and document.doc_id == last_doc_processed:
                pbar.update(1)
                last_seen = True
                continue
            summary_doc_tuples = []
            summ_pair_tuples = []
            summ_user_tuples = []
            document.populate_summary_doc_distances(self.measure, max_workers=self.max_workers)
            summary_doc_tuples.extend([(*k, v) for k, v in document.summary_doc_distances.items()])
            # print(f"self.summary_doc_tuples: {self.summary_doc_tuples}")
            document.populate_summary_summary_distances(self.measure, max_workers=self.max_workers)
            summ_pair_tuples.extend([(*k, v) for k, v in document.summary_summary_distances.items()])
            summ_user_tuples.extend([(*k, v) for k, v in document.summary_user_distances.items()])
            # print(f"self.summ_pair_tuples: {self.summ_pair_tuples}")
            # distance between summaries and documents
            write_scores_to_csv(summary_doc_tuples, fields=("doc_id", "origin_model", "uid", "score"),
                                filename=self.summary_doc_score_path)

            # distance between summaries
            write_scores_to_csv(summ_pair_tuples, fields=("doc_id", "origin_model", "uid1", "uid2", "score"),
                                filename=self.summ_summ_score_path)

            # distance between user/gold personalized summaries and model summaries
            write_scores_to_csv(summ_user_tuples, fields=("doc_id", "origin_model", "uid", "score"),
                                filename=self.sum_user_score_path)
            pbar.update(1)

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
        upair_scores_df["pair_score_weight_exp"] = upair_scores_df.apply(
            lambda x: np.exp(x["pair_score_weight"]),
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
        # model_Y_df = model_Y_df.sample(frac=sample_percentage / 100)
        accuracy_df = pd.read_csv(self.sum_user_score_path)

        accuracy_dict = {(k[0], k[1]): v["score"] for k, v in accuracy_df.set_index(["doc_id", "uid"]).to_dict(
            orient="index").items()}

        # print(model_Y_df.head(2))
        user_X_df = user_X_df.set_index(["doc_id", "uid1", "uid2"])
        user_X_score_map = user_X_df.to_dict(orient="index")

        # calculate proportion of all document,user1,user2 pairs
        # calculate mean of all scores for a given document,u_i

        # calculate min/max on model_Y_df["final_score"] and user_X_score_map[(doc_id,uid1,uid2))]
        model_Y_df["proportion"] = model_Y_df.apply(lambda x: calculate_proportion(x.final_score, user_X_score_map[
            (x["doc_id"], x["uid1"], x["uid2"])]["final_score"]), axis=1)

        # calculate mean of proportion column

        model_Y_df["doc_userwise_proportional_divergence"] = model_Y_df.groupby(["doc_id", "uid1"])[
            "proportion"].transform(
            lambda x: np.mean(x))
        # find mean of unique doc_id

        model_Y_df["docwise_mean_proportion"] = model_Y_df.groupby(["doc_id"])[
            "doc_userwise_proportional_divergence"].transform(
            lambda x: np.mean(x))

        if self.debug_flag:
            # save model_Y_df to csv
            model_Y_df.to_csv(f"{self.score_directory}/model_Y_df.csv", index=False)

        # temporary df to calculate docwise_mean_proportion
        final_df = model_Y_df[["doc_id", "docwise_mean_proportion"]].drop_duplicates()

        # sample percentage of final_df
        final_df = final_df.sample(frac=sample_percentage / 100)
        sampled_doc_ids = final_df["doc_id"].tolist()

        # calculate mean of accuracy of model-user pairs
        doc_pairs = list(model_Y_df.groupby(["doc_id", "uid1"]).groups.keys())
        doc_pairs.extend(model_Y_df.groupby(["doc_id", "uid2"]).groups.keys())
        doc_pairs = list(set(doc_pairs))
        # print(doc_pairs[:2])
        # print(accuracy_dict.values())
        msum_accuracies = [accuracy_dict[pair] for pair in doc_pairs if pair[0] in sampled_doc_ids]
        mean_msum_accuracy = np.mean(msum_accuracies)

        # find mean of mean_proportion column
        return round(1 - final_df['docwise_mean_proportion'].mean(), 4), round(mean_msum_accuracy, 4)
