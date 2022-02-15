import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



def gaussian_synthetic_sampling2(mc_centers, outlier, round_flag=False):
    """
    Parameters
    ----------
    mc_centers
        n x d numpy array of microclusters' centers, where n is the number of microcluster and d is the number of features
    outlier
        the outlier object
    round_flag
        whether to round each generated sampling
    """
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(mc_centers)
    index = neighbors.kneighbors([outlier])[1][0, 0]
    covariance = np.identity(mc_centers.shape[1]) * np.abs(mc_centers[index] - outlier) / (3**2)
    gaussian_data = np.random.multivariate_normal(outlier, covariance, mc_centers.shape[0])
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    outlier_class = np.vstack((gaussian_data, outlier))
    return outlier_class


def gaussian_synthetic_sampling(mc_centers, outlier, round_flag=False):
    """
    Parameters
    ----------
    mc_centers
        n x d numpy array of microclusters' centers, where n is the number of microcluster and d is the number of features
    outlier
        the outlier object
    round_flag
        whether to round each generated sampling
    """
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(mc_centers)
    neighbor_min_dist = neighbors.kneighbors([outlier])[0][0, 0] / mc_centers.shape[1]
    covariance = np.identity(mc_centers.shape[1]) * neighbor_min_dist / (3**2)
    n = mc_centers.shape[0]
    if n < mc_centers.shape[1]:
        n = mc_centers.shape[1]
    gaussian_data = np.random.multivariate_normal(outlier, covariance, n)
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    outlier_class = np.vstack((gaussian_data, outlier))
    return outlier_class


def compute_feature_contribution(n_features, npoints, classifiers):
    """
    Parameters
    ----------
    n_features
        number of features
    n_points
        list of number of inlier class used by each classifiers
    classifiers
        the classifier models
    """
    feature_scores = np.zeros(n_features)
    for npoint, clf in zip(npoints, classifiers):
        feature_scores += npoint * np.abs(clf.coef_[0])
    feature_scores /= float(np.sum(npoints))
    feature_scores /= np.sum(feature_scores)
    return feature_scores


def compute_simple_feature_contribution(n_features, classifiers):
    """
    Parameters
    ----------
    n_features
        number of features
    classifiers
        the classifier models
    """
    feature_scores = np.zeros(n_features)
    for clf in classifiers:
        feature_scores += np.abs(clf.coef_[0])
    feature_scores = feature_scores / len(classifiers)
    return feature_scores


def map_feature_scores(feature_names, feature_scores):
    result = {k: v for k, v in zip(feature_names, feature_scores)}
    return result


def run_svc(outlier_class, inlier_class, regularization, regularization_param, intercept_scaling):
    n_samples = outlier_class.shape[0] + inlier_class.shape[0]
    dual = False
    # if n_samples <= outlier_class.shape[1]:
    #     dual = True
    clf = LinearSVC(penalty=regularization,
                    C=regularization_param,
                    dual=dual,
                    intercept_scaling=intercept_scaling)
    X = np.vstack((outlier_class, inlier_class))
    y = np.hstack((np.ones(outlier_class.shape[0]), np.zeros(inlier_class.shape[0])))
    clf.fit(X, y)
    return clf
