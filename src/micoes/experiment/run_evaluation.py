import numpy as np
import pandas as pd

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

from micoes.experiment.run_microcluster_explainer_for_stream import run_temporal_explainer_comparison


class FeatureScoresEvaluation(object):
    def __init__(self, ground_truth_path, feature_scores, precise=0.01):
        self.ground_truth_df = pd.read_pickle(ground_truth_path)
        self.feature_scores = feature_scores
        self.precise = precise

    def _get_outlier_indices_intersection(self):
        outlier_indices = np.array(list(self.feature_scores.keys()))
        outlier_indices_gt = np.array(self.ground_truth_df["outlier_indices"])
        intersection = outlier_indices[np.in1d(outlier_indices, outlier_indices_gt)]
        return list(intersection)

    def _check_feature_scores(self, ground_truth, o_idx):
        matched = False
        for attribute in ground_truth:
            if self.feature_scores[o_idx][attribute] > self.precise:
                matched = True
            else:
                matched = False
        return matched

    def _find_matched_non_zero_feature_scores(self, intersection):
        matches = list()
        for o_idx in intersection:
            ground_truth = self.ground_truth_df.loc[self.ground_truth_df["outlier_indices"] == o_idx, "outlying_attributes"].to_list()[0][0]
            matched = self._check_feature_scores(ground_truth, o_idx)
            if matched:
                matches.append(o_idx)
        return matches

    def compute_matched_outlying_attributes(self, intersection, matches):
        number_of_identified_outlier = len(intersection)
        number_of_matched_explanation = len(matches)
        return number_of_matched_explanation / number_of_identified_outlier

    def compare_groundtruth_and_feature_scores(self):
        intersection = self._get_outlier_indices_intersection()
        self.intersection = intersection
        matches = self._find_matched_non_zero_feature_scores(intersection)
        self.matches = matches
        return matches, intersection


def run_evaluation_on_count_based_window(filepath, groundtruth_path, window_size=60, init_percentage=10,
                                         detector_type='lof', alpha=3, precise=0.01, lamda=1,
                                         label='label', break_window=None):
    check = pd.read_csv(filepath)
    len_init = int(init_percentage * check.shape[0] / 100)

    # parameters for CluStream
    n_microclusters = int(init_percentage * window_size / 100)
    if check.shape[0] > window_size * init_percentage:
        n_microclusters = n_microclusters * 3
    alpha = 3
    tau = window_size * precise

    # parameters for DenStream
    mu = window_size
    lamda = 1
    beta = precise
    eta = np.std(check.head(len_init).drop(['label'], axis=1).values) * 2

    detection_results, cluexplainer_results, denexplainer_results = run_temporal_explainer_comparison(filepath=filepath,
                                                                                                      n_microclusters=n_microclusters,
                                                                                                      alpha=alpha, tau=tau,
                                                                                                      lamda=lamda, mu=mu, beta=beta, eta=eta,
                                                                                                      label=label,
                                                                                                      init_percentage=init_percentage,
                                                                                                      detector_type=detector_type,
                                                                                                      max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                                                                                      regularization='l1', regularization_param=1,
                                                                                                      intercept_scaling=1, simple_feature_contribution=True,
                                                                                                      arrival_rate='Fixed', time_unit='seconds', time_interval=1, time_window=None,
                                                                                                      window_size=window_size, sliding_size=60,
                                                                                                      delay_time=1, break_window=break_window)

    windows = list(cluexplainer_results.keys())
    den_matches = list()
    clu_matches = list()
    coin_matches = list()
    den_total_intersection = list()
    clu_total_intersection = list()
    coin_total_intersection = list()
    for window in windows:
        clu_feature_scores = cluexplainer_results[window]['explanation_result']
        clu_FSE = FeatureScoresEvaluation(groundtruth_path, clu_feature_scores, precise)
        clu_window_matches, clu_intersection = clu_FSE.compare_groundtruth_and_feature_scores()
        clu_matches = clu_matches + clu_window_matches
        clu_total_intersection = clu_total_intersection + clu_intersection

        den_feature_scores = denexplainer_results[window]['explanation_result']
        den_FSE = FeatureScoresEvaluation(groundtruth_path, den_feature_scores, precise)
        den_window_matches, den_intersection = den_FSE.compare_groundtruth_and_feature_scores()
        den_matches = den_matches + den_window_matches
        den_total_intersection = den_total_intersection + den_intersection

        coin_feature_scores = cluexplainer_results[window]['coin_result']['feature_scores']
        coin_FSE = FeatureScoresEvaluation(groundtruth_path, coin_feature_scores, precise)
        coin_window_matches, coin_intersection = coin_FSE.compare_groundtruth_and_feature_scores()
        coin_matches = coin_matches + coin_window_matches
        coin_total_intersection = coin_total_intersection + coin_intersection

    den_matched_percentage = 0
    clu_matched_percentage = 0
    coin_matched_percentage = 0

    den_matched_percentage = len(den_matches) / len(den_total_intersection)
    clu_matched_percentage = len(clu_matches) / len(clu_total_intersection)
    coin_matched_percentage = len(coin_matches) / len(coin_total_intersection)

    matched_result = {  # 'den_matches': den_matches,
        # 'clu_matches': clu_matches,
        # 'coin_matches': coin_matches,
        # 'den_total_intersection': den_total_intersection,
        # 'clu_total_intersection': clu_total_intersection,
        # 'coin_total_intersection': coin_total_intersection,
        'den_matched_percentage': den_matched_percentage,
        'clu_matched_percentage': clu_matched_percentage,
        'coin_matched_percentage': coin_matched_percentage}
    return detection_results, cluexplainer_results, denexplainer_results, matched_result
