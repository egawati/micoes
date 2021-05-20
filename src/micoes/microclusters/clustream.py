import time
import math
import sys
import numpy as np

from sklearn.cluster import KMeans

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def update_clu_microcluster(clu, X, profiling=False):
    if profiling:
        start = time.perf_counter()

    for point in X:
        clu.online_update_microcluster(point, time.time())

    if profiling:
        duration = time.perf_counter() - start
        return duration


class CluStream(object):
    def __init__(self, n_microclusters, n_attributes, alpha, tau):
        """
        Parameters
        ----------
        n_microclusters
            int, number of microcluster
        n_attributes
            int, number of attributes
        alpha
            int
        tau
            period of time

        """
        self.n_microclusters = n_microclusters
        self.microclusters = list()
        self.n_attributes = n_attributes
        self.alpha = alpha
        self.tau = tau

    def initialize(self, points):
        """
        Initialize clusters
        Parameters
        ----------
        points
            n by d numpy array where d (the number of column) represent the number of attributes
        """
        clustering = KMeans(n_clusters=self.n_microclusters, random_state=0).fit(points)
        labels = np.unique(clustering.labels_)
        for label in labels:
            indices = list(np.where(clustering.labels_ == label)[0])
            objects = points[indices, :]
            n_points = objects.shape[0]
            linear_sum = np.sum(objects, axis=0)
            squared_sum = np.sum(objects**2, axis=0)
            if self.n_attributes == objects.shape[1]:
                mc = MicroCluster(self.n_attributes, self.alpha,
                                  n_points=n_points,
                                  linear_sum=linear_sum,
                                  squared_sum=squared_sum)
                mc.update_center_and_radius()
                self.microclusters.append(mc)
            else:
                logging.warning('the number of initial attributes is not the same with the number of the current object attributes')

    def online_update_microcluster(self, point, point_timestamp):
        smallest_distance = sys.float_info.max
        idx = None
        found = False
        for i, mc in enumerate(self.microclusters):
            dist = np.linalg.norm(mc.center - point)
            if smallest_distance > dist:
                smallest_distance = dist
                idx = i
        if smallest_distance <= self.microclusters[idx].radius:
            self.microclusters[idx].update_cluster_feature(point, point_timestamp)
        else:
            self._delete_the_least_recent_mc()
            new_mc = MicroCluster(self.n_attributes, self.alpha)
            new_mc.update_cluster_feature(point, point_timestamp)
            new_mc.update_center_and_radius()
            self.microclusters.append(new_mc)

    def _delete_the_least_recent_mc(self):
        t_diff = time.time() - self.tau
        least_recent_timestamp = time.time()
        idx = None
        for i, mc in enumerate(self.microclusters):
            if mc.latest_timestamp < t_diff and mc.latest_timestamp < least_recent_timestamp:
                least_recent_timestamp = mc.latest_timestamp
                idx = i
        if idx is not None:
            self.microclusters.pop(idx)
            # logging.info('Removing an old microcluster')
        else:
            self._merge_two_closest_mc()
            # logging.info('Merging two microclusters')

    def _merge_two_closest_mc(self):
        smallest_distance = sys.float_info.max
        idx1 = None
        idx2 = None
        for i, mc1 in enumerate(self.microclusters):
            for j, mc2 in enumerate(self.microclusters[i + 1:]):
                dist = np.linalg.norm(mc1.center - mc2.center)
                if dist < smallest_distance:
                    smallest_distance = dist
                    idx1 = i
                    idx2 = j
        mc1 = self.microclusters.pop(i)
        mc2 = self.microclusters.pop(j)
        mc = MicroCluster(self.n_attributes, self.alpha,
                          n_points=mc1.n_points + mc2.n_points,
                          linear_sum=mc1.linear_sum + mc2.linear_sum,
                          squared_sum=mc1.squared_sum + mc2.squared_sum,
                          t_linear_sum=mc1.t_linear_sum + mc2.t_linear_sum,
                          t_squared_sum=mc1.t_squared_sum + mc2.t_squared_sum)
        mc.latest_timestamp = mc1.latest_timestamp
        mc.update_center_and_radius()
        if mc.latest_timestamp < mc2.latest_timestamp:
            mc.latest_timestamp = mc2.latest_timestamp
        self.microclusters.append(mc)


class MicroCluster(object):
    def __init__(self, n_attributes, alpha,
                 n_points=0, linear_sum=0.0, squared_sum=0.0,
                 t_linear_sum=0.0, t_squared_sum=0):
        self.latest_timestamp = time.time()
        self.n_points = n_points  # the weights
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.t_linear_sum = t_linear_sum
        self.t_squared_sum = t_squared_sum
        self.n_attributes = n_attributes
        self.alpha = alpha

    def update_cluster_feature(self, point, point_timestamp):
        self.n_points += 1
        self.linear_sum += point
        self.squared_sum += point ** 2
        self.t_linear_sum += point_timestamp
        self.t_squared_sum += point_timestamp ** 2
        self.latest_timestamp = time.time()

    def update_center_and_radius(self):
        sigma = np.sqrt((self.squared_sum / self.n_points) - (self.linear_sum / self.n_points)**2)
        if len(sigma) == self.n_attributes:
            self.radius = (np.sum(sigma) / len(sigma)) * self.alpha
            self.center = self.linear_sum / self.n_points

    def get_number_of_points(self):
        return self.n_points

    def get_center(self):
        return self.center
