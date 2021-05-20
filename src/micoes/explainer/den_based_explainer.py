import os
import copy
import numpy as np

from sklearn.cluster import KMeans
from sklearn import svm

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

from .common import run_svc
from .common import map_feature_scores
from .common import gaussian_synthetic_sampling
from .common import compute_feature_contribution
from .common import compute_simple_feature_contribution

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class DenstreamExplainer(object):
    def __init__(self, p_microclusters, c_microclusters,
                 max_distance, n_features, features=None,
                 regularization='l1', regularization_param=1,
                 intercept_scaling=1, simple_feature_contribution=True, o_microclusters=None):
        self.p_microclusters = p_microclusters
        self.c_microclusters = c_microclusters
        self.max_distance = max_distance
        self.n_features = n_features
        self.features = features
        self.intercept_scaling = intercept_scaling
        self.regularization = regularization
        self.regularization_param = regularization_param
        self.simple_feature_contribution = simple_feature_contribution
        self.o_microclusters = o_microclusters

    def explain_outlier(self, outlier, round_flag=0, prior_knowledge=1):
        """
        run this method for every outlier data points
        """
        microclusters = self.c_microclusters + self.p_microclusters

        if self.o_microclusters is not None and len(microclusters) == 0:
            microclusters = self.o_microclusters

        relevant_microcluster_centers, nums_c = self._find_relevant_microclusters(microclusters, outlier)

        relevant_microcluster_centers = np.array(relevant_microcluster_centers)

        outlier_class = gaussian_synthetic_sampling(relevant_microcluster_centers, outlier, round_flag)

        relevant_microcluster_centers = np.reshape(relevant_microcluster_centers, (-1, self.n_features))
        classifiers = []
        for i in range(relevant_microcluster_centers.shape[0]):
            center = relevant_microcluster_centers[i, :]
            clf = run_svc(outlier_class,
                          center.reshape((-1, self.n_features)),
                          self.regularization,
                          self.regularization_param,
                          self.intercept_scaling)
            classifiers.append(clf)

        if self.simple_feature_contribution:
            feature_scores = compute_simple_feature_contribution(self.n_features, classifiers)
        else:
            feature_scores = compute_feature_contribution(self.n_features, nums_c, classifiers)

        # if feature names are available, set feature_scores as dictionary
        if self.features is not None:
            feature_scores = map_feature_scores(self.features, feature_scores)

        return feature_scores

    def _find_relevant_microclusters(self, microclusters, outlier):
        relevant_microcluster_centers = list()
        nums_c = list()
        min_dist = float("inf")
        min_idx = None
        i = 0
        mc_outlier_dist = None
        for mc in microclusters:
            center = mc.get_center()
            mc_outlier_dist = np.linalg.norm(outlier - center)
            if mc_outlier_dist < self.max_distance:
                # relevant_microcluster.append(mc)
                relevant_microcluster_centers.append(center)
                nums_c.append(mc.get_number_of_points())
            if mc_outlier_dist < min_dist:
                min_dist = mc_outlier_dist
                min_idx = i
            i += 1
        if not relevant_microcluster_centers:
            mc = microclusters[min_idx]
            relevant_microcluster_centers.append(mc.get_center())
            nums_c.append(mc.get_number_of_points())

        return relevant_microcluster_centers, nums_c
