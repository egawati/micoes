import sys
import time
import math
import numpy as np

from sklearn.cluster import DBSCAN

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def update_den_microcluster(den, X, profiling=False):
    if profiling:
        start = time.perf_counter()

    for point in X:
        den.online_update_microcluster(point, time.time())

    if profiling:
        duration = time.perf_counter() - start
        return duration


class DenStream(object):
    def __init__(self, lamda, mu, beta, eta, n_attributes):
        """
        Parameters
        ----------
        lamda
            decaying factor
        mu
            core weight threshold
        beta
            tolerance factor
        eta
            the radius threshold
        """
        self.lamda = lamda
        self.mu = mu
        self.beta = beta
        self.eta = eta
        self.n_attributes = n_attributes
        self.o_microclusters = list()  # outlier microcluster
        self.p_microclusters = list()  # potential core microcluster
        self.c_microclusters = list()  # core microcluster

    def initialize(self, points):
        """
        Initialize clusters
        Parameters
        ----------
        points
            n by d numpy array where d (the number of column) represent the number of attributes
        """
        clustering = DBSCAN(eps=self.eta, min_samples=1).fit(points)
        labels = np.unique(clustering.labels_)
        for label in labels:
            indices = list(np.where(clustering.labels_ == label)[0])
            objects = points[indices, :]
            n_tpoints = objects.shape[0]
            tlinear_sum = np.sum(objects, axis=0)
            tsquared_sum = np.sum(objects**2, axis=0)
            if n_tpoints >= self.mu:
                microcluster_type = 'c-microcluster'
            else:
                microcluster_type = 'o-microcluster'
            microcluster = MicroCluster(self.n_attributes,
                                        self.lamda,
                                        self.mu,
                                        self.eta,
                                        self.beta,
                                        n_tpoints,
                                        tlinear_sum,
                                        tsquared_sum,
                                        microcluster_type
                                        )
            if microcluster_type == 'c-microcluster':
                self.c_microclusters.append(microcluster)
            else:
                self.o_microclusters.append(microcluster)

    def _find_closest_microcluster(self, microclusters, point, point_timestamp, token='p-microcluster'):
        """
        Find the closest microcluster for the incoming point and assign the point to it
        Parameters
        ----------
        microclusters
            list of microcluster
        point
            a data point
        point
            a data point timestamp
        """
        smallest_distance = sys.float_info.max
        idx = None
        found = False
        for i, mc in enumerate(microclusters):
            dist = np.linalg.norm(mc.center - point)
            if smallest_distance > dist:
                smallest_distance = dist
                idx = i
        if smallest_distance <= self.eta:
            microcluster_type = microclusters[idx].microcluster_type
            microclusters[idx].update_cluster_feature(point, point_timestamp)
            microclusters[idx].update_microcluster_type()
            updated_microcluster_type = microclusters[idx].microcluster_type
            if microcluster_type == 'p-microcluster' and updated_microcluster_type == 'c-microcluster':
                self.c_microclusters.append(microclusters[idx])
                self.p_microclusters.pop(idx)
                #logging.info('Updating PCMC to CMC')
            elif microcluster_type == 'o-microcluster' and updated_microcluster_type == 'p-microcluster':
                self.p_microclusters.append(microclusters[idx])
                self.o_microclusters.pop(idx)
                #logging.info('Updating OMC to PCMC')
            found = True
        if not found and token == 'o-microcluster':
            mc = MicroCluster(self.n_attributes,
                              self.lamda,
                              self.mu,
                              self.eta,
                              self.beta,
                              microcluster_type='o-microcluster')
            mc.update_cluster_feature(point, point_timestamp)
            self.o_microclusters.append(mc)
        return found

    def online_update_microcluster(self, point, point_timestamp):
        """
        insert a new point to a microcluster
        """
        found = False
        if self.p_microclusters:
            found = self._find_closest_microcluster(self.p_microclusters, point, point_timestamp, token='p-microcluster')
        if not found:
            self._find_closest_microcluster(self.o_microclusters, point, point_timestamp, token='o-microcluster')

    def _remove_microclusters(self, microclusters, indices):
        check = 0
        for idx in indices:
            microclusters.pop(idx - check)
            check += 1
        return microclusters

    def check_microclusters_status(self, period):
        """
        After a given number of T time steps,
        need to delete any c_microcluster that has npoints < self.mu
        and delete any o_microcluster thas has been around for a period of time
        """
        # Update core microclusters
        cmc_npoints = np.array([mc.n_tpoints for mc in self.c_microclusters])
        cmc_to_remove = list(np.where(cmc_npoints < self.mu)[0])
        if cmc_to_remove:
            self.c_microclusters = self._remove_microclusters(self.c_microclusters, cmc_to_remove)

        # Update outlier microclusters
        omc_lastest_timestamp = np.array([mc.latest_timestamp for mc in self.o_microclusters])
        current_ts = time.time()
        omc_to_remove = list(np.where(current_ts - omc_lastest_timestamp > period)[0])
        if omc_to_remove:
            self.o_microclusters = self._remove_microclusters(self.o_microclusters, omc_to_remove)


class MicroCluster(object):
    def __init__(self, n_attributes, lamda, mu, eta, beta,
                 n_tpoints=0, tlinear_sum=0.0, tsquared_sum=0.0,
                 microcluster_type='o-microcluster'):
        self.n_points = n_tpoints
        self.n_tpoints = n_tpoints  # the weights
        self.tlinear_sum = tlinear_sum
        self.tsquared_sum = tsquared_sum
        self.center = np.zeros(n_attributes)
        self.radius = 0
        self.lamda = lamda
        self.mu = mu
        self.eta = eta
        self.beta = beta
        self.latest_timestamp = time.time()
        self.microcluster_type = microcluster_type

    def _fading_function(self, t):
        f = 2 ** -(self.lamda * t)
        return f

    def update_cluster_feature(self, point, point_timestamp):
        ts = time.time()
        ts_diff = ts - point_timestamp
        self.n_tpoints = self.n_tpoints + self._fading_function(ts_diff)
        self.tlinear_sum = self.tlinear_sum + self._fading_function(ts_diff) * point
        self.tsquared_sum = self.tsquared_sum + self._fading_function(ts_diff) * (point**2)
        sigma = np.sqrt((self.tsquared_sum / self.n_tpoints) - (self.tlinear_sum / self.n_tpoints)**2)
        self.radius = np.sum(sigma) / len(sigma)
        self.center = self.tlinear_sum / self.n_tpoints
        self.latest_timestamp = ts
        self.n_points += 1

    def update_microcluster_type(self):
        if self.n_tpoints >= self.mu and self.radius <= self.eta:
            self.microcluster_type = 'c-microcluster'
        elif self.n_tpoints >= self.beta * self.mu and self.radius <= self.eta:
            self.microcluster_type = 'p-microcluster'

    def get_number_of_points(self):
        return self.n_points

    def get_center(self):
        return self.center
