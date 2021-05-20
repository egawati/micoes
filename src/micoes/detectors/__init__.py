import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
#from timeit import default_timer as timer

import time


def lof_detector(X):
    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(X)
    y_pred = np.array(0.5 * (1 - y_pred), dtype=int)
    return y_pred


def isoforest_detector(X):
    num_inst = X.shape[0]
    clf = IsolationForest(behaviour='new', max_samples=num_inst, random_state=0)
    clf.fit(X)
    y_pred = clf.predict(X)
    y_pred = np.array(0.5 * (1 - y_pred), dtype=int)
    return y_pred


def run_detection(X, detector_type='lof', profiling=False):
    if profiling:
        start = time.perf_counter()
    detector = None

    if detector_type == 'lof':
        detector = lof_detector(X)
    elif detector_type == 'isolationforest':
        detector = isoforest_detector(X)

    if profiling:
        duration = time.perf_counter() - start
        return (detector, duration)
    else:
        return detector
