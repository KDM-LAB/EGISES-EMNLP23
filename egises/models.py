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

from egises.utils import custom_sigmoid, write_scores_to_csv, divide_with_exception, calculate_minmax_proportion

from dataclasses import dataclass


@dataclass
class PersevalParams:
    ADP_alpha: float = 4.0
    ADP_beta: float = 1.0
    ACP_alpha: float = 4.0
    ACP_beta: float = 1.0
    EDP_alpha: float = 3.0
    EDP_beta: float = 1.0  # used for ablation ranging from 1.0 to 2.0
    epsilon: float = 0.0000001


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
                 max_workers=1, debug_flag=True, version="v2"):
        self.model_name = model_name
        self.version = version
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

    def populate_distances(self, simplified_flag=False):
        """
        :param simplified_flag: doesnt normalize scores based on doc distances
        :return:
        """
        last_seen, last_doc_processed = False, None
        processed_doc_ids = []
        if os.path.exists(self.summary_doc_score_path) and os.path.exists(self.summ_summ_score_path):
            summary_doc_score_df = pd.read_csv(self.summary_doc_score_path)
            # get unique doc_ids
            processed_doc_ids = list(summary_doc_score_df["doc_id"].unique())
        # populate document scores from where left off
        pbar = tqdm(total=3840, desc="Populating Distances")
        for document in self.documents:
            # find last doc_id in summary_doc_distances
            # avoid pouplating distances as hj distances already processed
            if self.measure.__name__ == "calculate_hj":
                break
            if document.doc_id in processed_doc_ids:
                pbar.update(1)
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
        self.accuracy_df = pd.read_csv(self.sum_user_score_path)

        # set Nan values with 1(maximum distance) as Nan values can lead to final scores as Nan
        self.accuracy_df = self.accuracy_df.fillna(1.0)
        # calculate X,Y scores for all document,u1,u2  pairs
        self.user_X_df = self.get_user_model_X_scores(model_name="user")
        self.model_Y_df = self.get_user_model_X_scores(model_name=self.model_name)

        # create map of user_X_df[(doc_id,uid1,uid2)] to user_X_df["final_score"]
        user_X_df = self.user_X_df.set_index(["doc_id", "uid1", "uid2"])
        user_X_score_map = user_X_df.to_dict(orient="index")

        # calculate min/max on model_Y_df["final_score"] and user_X_score_map[(doc_id,uid1,uid2))]

        # consider only those rows in self.model_Y_df["proportion"] where (doc_id,uid1,uid2)] is user_X_score_map.keys
        self.model_Y_df = self.model_Y_df[self.model_Y_df.apply(
            lambda x: (x["doc_id"], x["uid1"], x["uid2"]) in user_X_score_map.keys(), axis=1)]
        if not simplified_flag:
            self.model_Y_df["proportion"] = self.model_Y_df.apply(
                lambda x: calculate_minmax_proportion(x.final_score, user_X_score_map[
                    (x["doc_id"], x["uid1"], x["uid2"])]["final_score"], epsilon=0.00001), axis=1)
        else:  # simplified version where propotion is not weighted
            self.model_Y_df["proportion"] = self.model_Y_df.apply(
                lambda x: calculate_minmax_proportion(x.score, user_X_score_map[
                    (x["doc_id"], x["uid1"], x["uid2"])]["score"], epsilon=0.00001), axis=1)

    def get_user_model_X_scores(self, model_name):
        usum_scores_df = self.summary_doc_score_df[self.summary_doc_score_df["origin_model"] == model_name]
        # TODO: rename to mpair_scores_df
        upair_scores_df = self.summ_pair_score_df[self.summ_pair_score_df["origin_model"] == model_name]

        usum_scores_df = usum_scores_df.set_index(["doc_id", "uid"])
        sum_doc_score_dict = {k: v["score"] for k, v in usum_scores_df.to_dict(orient="index").items()}

        # step2: get ratio of summary_summary_distance to summary_doc_distance
        # w(u_ij) = distance(ui,uj)/sum(distance(ui,doc))
        # consider only those rows in upair_scores_df where (x["doc_id"], x["uid1"]) is in sum_doc_score_dict
        upair_scores_df = upair_scores_df[upair_scores_df.apply(
            lambda x: (x["doc_id"], x["uid1"]) in sum_doc_score_dict.keys(), axis=1
        )]
        # print(upair_scores_df.shape)
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
            lambda x: x["pair_score_weight_exp_softmax"] * x["score"], axis=1)
        # keep only doc_id, uid1, uid2, final_score
        final_df = upair_scores_df[["doc_id", "uid1", "uid2", "score", "final_score"]]
        return final_df

    def get_egises_score(self, sample_percentage=100):
        # sample doc_id,u1, u2 pairs from model_Y_df
        model_Y_df = self.model_Y_df.sample(frac=sample_percentage / 100)
        accuracy_dict = {(k[0], k[1]): v["score"] for k, v in self.accuracy_df.set_index(["doc_id", "uid"]).to_dict(
            orient="index").items()}

        # find mean of model_Y_df["final_score"] grouped by doc_id,uid1
        model_Y_df["doc_userwise_proportional_divergence"] = model_Y_df.groupby(["doc_id", "uid1"])[
            "proportion"].transform(
            lambda x: np.mean(x))

        # find mean of model_Y_df["doc_userwise_proportional_divergence"] grouped by doc_id
        model_Y_df["docwise_mean_proportion"] = model_Y_df.groupby(["doc_id"])[
            "doc_userwise_proportional_divergence"].transform(
            lambda x: np.mean(x))

        if self.debug_flag and sample_percentage == 100:
            # save model_Y_df to csv
            model_Y_df.to_csv(f"{self.score_directory}/model_Y_df_{self.version}.csv", index=False)

        # temporary df to calculate docwise_mean_proportion
        final_df = model_Y_df[["doc_id", "docwise_mean_proportion"]].drop_duplicates()

        # calculate mean of accuracy of model-user pairs
        doc_pairs = list(model_Y_df.groupby(["doc_id", "uid1"]).groups.keys())
        doc_pairs.extend(model_Y_df.groupby(["doc_id", "uid2"]).groups.keys())
        doc_pairs = list(set(doc_pairs))
        # print(doc_pairs[:2])
        # print(accuracy_dict.values())
        msum_accuracies = [accuracy_dict[pair] for pair in doc_pairs]
        mean_msum_accuracy = np.mean(msum_accuracies)

        # find mean of mean_proportion column
        return 1 - final_df['docwise_mean_proportion'].mean(), mean_msum_accuracy

    def calculate_edp(self, accuracy_df, perseval_params: PersevalParams) -> dict:
        # calculation of d_mean
        summ_user_mean_dict = accuracy_df.groupby(["doc_id", "origin_model"]).apply(
            lambda x: np.mean(x["score"])).to_dict()

        # calculation of d_min
        summ_user_min_dict = accuracy_df.groupby(["doc_id", "origin_model"]).apply(lambda x: min(x["score"])).to_dict()

        accuracy_df["d_min"] = accuracy_df.apply(lambda x: summ_user_min_dict[(x["doc_id"], x["origin_model"])],
                                                 axis=1)
        accuracy_df["d_mean"] = accuracy_df.apply(lambda x: summ_user_mean_dict[(x["doc_id"], x["origin_model"])],
                                                  axis=1)

        # calculate Accuracy Inconsistency Penalty(ACP)
        accuracy_df["pterm1"] = accuracy_df.apply(
            lambda x: ((x["score"] - x["d_min"]) / ((x["d_mean"] - x["d_min"]) + perseval_params.epsilon)), axis=1)
        # applied sigmoid to pterm1
        # # calculate max min values of pterm1
        # max_pterm1 = accuracy_df["pterm1"].max()
        # min_pterm1 = accuracy_df["pterm1"].min()
        # print(f"max_pterm1: {max_pterm1}, min_pterm1: {min_pterm1}, alpha={perseval_params.ACP_alpha}, beta={perseval_params.ACP_beta}")
        accuracy_df["ACP"] = accuracy_df.apply(
            lambda x: custom_sigmoid(x["pterm1"], alpha=perseval_params.ACP_alpha, beta=perseval_params.ACP_beta),
            axis=1)

        # calculate Accuracy Drop Penalty(ADP)
        accuracy_df["pterm2"] = accuracy_df.apply(
            lambda x: (x["d_min"] - 0) / (1 - x["d_min"] + perseval_params.epsilon), axis=1)
        # # calculate max min values of pterm2
        # max_pterm2 = accuracy_df["pterm2"].max()
        # min_pterm2 = accuracy_df["pterm2"].min()
        # print(f"max_pterm2: {max_pterm2}, min_pterm2: {min_pterm2}, alpha={perseval_params.ADP_alpha}, beta={perseval_params.ADP_beta}")
        accuracy_df["ADP"] = accuracy_df.apply(
            lambda x: custom_sigmoid(x["pterm2"], alpha=perseval_params.ADP_alpha, beta=perseval_params.ADP_beta),
            axis=1)

        # calculate Document Generalization Penalty(DGP)
        accuracy_df["DGP"] = accuracy_df.apply(lambda x: (x["ACP"] + x["ADP"]), axis=1)

        # # calculate max min values of DGP
        # max_DGP = accuracy_df["DGP"].max()
        # min_DGP = accuracy_df["DGP"].min()
        # print(f"max_DGP: {max_DGP}, min_DGP: {min_DGP}, alpha={perseval_params.EDP_alpha}, beta={perseval_params.EDP_beta}")
        accuracy_df["EDP"] = accuracy_df.apply(
            lambda x: (1 - custom_sigmoid(x["DGP"], alpha=perseval_params.EDP_alpha,
                                          beta=perseval_params.EDP_beta)) + perseval_params.epsilon,
            axis=1)

        doc_user_edp_dict = accuracy_df.groupby(["doc_id", "uid"]).apply(lambda x: np.mean(x["EDP"])).to_dict()
        return doc_user_edp_dict

    def get_perseval_score(self, sample_percentage=100, perseval_params: PersevalParams = None):
        if not perseval_params:
            perseval_params = PersevalParams()
        # calculate_degress
        model_Y_df = self.model_Y_df.sample(frac=sample_percentage / 100)

        # for debug purpose
        if sample_percentage == 100 and self.debug_flag:
            model_Y_df.to_csv(f"{self.score_directory}/model_Y_df_perseval_df_{self.version}.csv", index=False)

        # find mean of model_Y_df["final_score"] grouped by doc_id,uid1
        model_Y_df["doc_userwise_proportional_divergence"] = model_Y_df.groupby(["doc_id", "uid1"])[
            "proportion"].transform(
            lambda x: np.mean(x))

        doc_user_degress_df = model_Y_df[["doc_id", "uid1", "doc_userwise_proportional_divergence"]].drop_duplicates()
        # get doc_id, uid1 pairs from doc_user_degress_df
        degress_pairs = list(doc_user_degress_df.groupby(["doc_id", "uid1"]).groups.keys())

        # pick records from accuracy_df where doc_id, uid in doc_user_degress_df
        accuracy_df = self.accuracy_df[
            self.accuracy_df.apply(lambda x: (x["doc_id"], x["uid"]) in degress_pairs, axis=1)]
        # calculate_edp based on sampled model_Y_df
        doc_user_edp_dict = self.calculate_edp(accuracy_df, perseval_params)

        try:
            assert len(doc_user_edp_dict) == len(doc_user_degress_df)
        except AssertionError as err:
            print(f"len(doc_user_edp_dict): {len(doc_user_edp_dict)}")
            print(f"len(doc_user_degress_df): {len(doc_user_degress_df)}")
            raise Exception("length of doc_user_edp_dict and doc_user_degress_df should be equal")

        doc_user_degress_df["edp"] = doc_user_degress_df.apply(
            lambda x: doc_user_edp_dict[(x["doc_id"], x["uid1"])], axis=1)
        doc_user_degress_df["perseval"] = doc_user_degress_df.apply(
            lambda x: x["doc_userwise_proportional_divergence"] * x["edp"], axis=1)
        doc_user_degress_df["docwise_perseval_proportion"] = doc_user_degress_df.groupby(["doc_id"])[
            "perseval"].transform(
            lambda x: np.mean(x))

        # for debug purpose
        if sample_percentage == 100 and self.debug_flag:
            doc_user_degress_df.to_csv(f"{self.score_directory}/doc_degress_perseval_df_{self.version}.csv",
                                       index=False)

        final_doc_df = doc_user_degress_df[["doc_id", "docwise_perseval_proportion"]].drop_duplicates()

        return final_doc_df['docwise_perseval_proportion'].mean(), accuracy_df["score"].mean()
        # take docwise mean of perseval
