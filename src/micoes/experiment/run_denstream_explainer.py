import numpy as np
import pandas as pd
import math
import time
#from timeit import default_timer as timer

from micoes.explainer import DenstreamExplainer
from micoes.microclusters.denstream import DenStream
from micoes.multiprocessing import run_detector_denstream

from .run_coin import run_coin_explainer

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_detector_explainer_denstream_online_wtiming(den, X, detector_type='lof', features=None,
                                                    max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                                    regularization='l1', regularization_param=1,
                                                    intercept_scaling=1, simple_feature_contribution=True):
  # run the outlier detection and den_stream in parallel
  # logging.info('Run outlier detection and online microcluster')
  detector_denstream_rs, multiprocessing_duration = run_detector_denstream(X, den, detector_type, profiling=True)
  # get the explaination_results
  results = get_detector_explainer_denstream_result(den, detector_denstream_rs, X,
                                                    max_rad_multiplier=max_rad_multiplier,
                                                    round_flag=round_flag,
                                                    prior_knowledge=1,
                                                    regularization=regularization,
                                                    regularization_param=regularization_param,
                                                    intercept_scaling=intercept_scaling,
                                                    simple_feature_contribution=simple_feature_contribution)

  if isinstance(detector_denstream_rs[1], float):  # thread 1 was for outlier detection
    detection_results = detector_denstream_rs[0][0]
  else:  # thread 1 was for microcluster generation
    detection_results = detector_denstream_rs[1][0]

  # run original coin explanation
  # logging.info('Run original coin explanation')
  interpreter_duration, coin_explanation = run_coin_explainer(X, detection_results, prior_knowledge)
  results['coin_duration'] = interpreter_duration
  results['coin_result'] = coin_explanation
  return results


def get_detector_explainer_denstream_result(den, detector_denstream_rs, X, features=None,
                                            max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                            regularization='l1', regularization_param=1,
                                            intercept_scaling=1, simple_feature_contribution=True):
  results = {}
  if isinstance(detector_denstream_rs[1], float):  # thread 1 was for outlier detection
    detection_results = detector_denstream_rs[0][0]
    results['detection_duration'] = detector_denstream_rs[0][1]
    results['microcluster_duration'] = detector_denstream_rs[1]
  else:  # thread 1 was for microcluster generation
    detection_results = detector_denstream_rs[1][0]
    results['detection_duration'] = detector_denstream_rs[1][1]
    results['microcluster_duration'] = detector_denstream_rs[0]

  # run the DenStream microcluster based explanation
  # logging.info('Run den-microcluster based xplanation')
  start = time.perf_counter()
  outlier_indices = np.where(detection_results == 1)[0]
  # rads = [mc.get_radius() for mc in den.microclusters]
  # max_distance = max(rads) * max_rad_multiplier

  max_distance = np.std(X) * max_rad_multiplier

  explainer = DenstreamExplainer(den.p_microclusters, den.c_microclusters,
                                 max_distance, den.n_attributes,
                                 features=features,
                                 regularization=regularization,
                                 regularization_param=regularization_param,
                                 intercept_scaling=intercept_scaling,
                                 simple_feature_contribution=simple_feature_contribution)
  explanations = {}
  for i in list(outlier_indices):
    outlier = X[i, :]
    explanations['outlier_at_index_' + str(i)] = explainer.explain_outlier(outlier, round_flag, prior_knowledge)
  duration = time.perf_counter() - start

  results['explanation_duration'] = duration
  results['explanation_result'] = explanations
  results['total_data'] = X.shape[0]
  results['total_outliers'] = len(outlier_indices)
  results['total_features'] = X.shape[1]
  return results


def run_denexplainer_experiment_on_each_file(filename, lamda, mu, beta, eta,
                                             label='label', init_percentage=10, detector_type='lof',
                                             max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                             regularization='l1', regularization_param=1,
                                             intercept_scaling=1, simple_feature_contribution=True):
  raw = pd.read_csv(filename)

  data = raw.drop(['label'], axis=1)
  features = tuple(data.columns)
  data = data.values

  init = math.ceil(data.shape[0] * init_percentage / 100)

  den = DenStream(lamda=lamda,
                  mu=mu,
                  beta=beta,
                  eta=eta,
                  n_attributes=data.shape[1])
  Xinit = data[:init, :]

  den.initialize(Xinit)
  X = data[init:, :]
  result = run_detector_explainer_denstream_online_wtiming(den, X,
                                                           detector_type=detector_type,
                                                           max_rad_multiplier=max_rad_multiplier,
                                                           round_flag=round_flag,
                                                           prior_knowledge=prior_knowledge,
                                                           regularization=regularization,
                                                           regularization_param=regularization_param,
                                                           intercept_scaling=intercept_scaling,
                                                           simple_feature_contribution=simple_feature_contribution)
  return result


def build_dataframe_denexplainer_experiments_basic(folder, filenames, lamda, mu, beta, eta,
                                                   label='label', init_percentage=20, detector_type='lof',
                                                   max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                                   regularization='l1', regularization_param=1,
                                                   intercept_scaling=1, simple_feature_contribution=True):
  stream_names = list()
  detection_duration = list()
  microcluster_duration = list()
  explanation_duration = list()
  explanation_result = list()
  total_data = list()
  total_features = list()
  total_outliers = list()
  coin_duration = list()
  coin_result = list()

  for filename in filenames:
    filepath = folder + filename
    stream_name = filename.replace('.csv', '')
    logging.info(f'Running on {stream_name} -----------------')
    stream_names.append(stream_name)
    result = run_denexplainer_experiment_on_each_file(filepath, lamda, mu, beta, eta,
                                                      label=label,
                                                      init_percentage=init_percentage,
                                                      detector_type=detector_type,
                                                      max_rad_multiplier=max_rad_multiplier,
                                                      round_flag=round_flag,
                                                      prior_knowledge=prior_knowledge,
                                                      regularization=regularization,
                                                      regularization_param=regularization_param,
                                                      intercept_scaling=intercept_scaling,
                                                      simple_feature_contribution=simple_feature_contribution)
    detection_duration.append(result['detection_duration'])
    microcluster_duration.append(result['microcluster_duration'])
    explanation_duration.append(result['explanation_duration'])
    explanation_result.append(result['explanation_result'])
    total_data.append(result['total_data'])
    total_outliers.append(result['total_outliers'])
    total_features.append(result['total_features'])
    coin_duration.append(result['coin_duration'])
    coin_result.append(result['coin_result'])

  experiments = {}
  experiments['stream_names'] = stream_names
  experiments['detection_duration'] = detection_duration
  experiments['microcluster_duration'] = microcluster_duration
  experiments['explanation_duration'] = explanation_duration
  experiments['total_data'] = total_data
  experiments['total_outliers'] = total_outliers
  experiments['total_features'] = total_features
  experiments['coin_duration'] = np.sum((coin_duration, detection_duration), axis=0).tolist()
  experiments['den_explainer_duration'] = np.sum((np.maximum(detection_duration, microcluster_duration), explanation_duration), axis=0).tolist()

  explanations = {}
  explanations['stream_names'] = stream_names
  explanations['explanation_result'] = explanation_result
  explanations['coin_result'] = coin_result

  return pd.DataFrame(experiments), explanations


def build_compact_df_den_explainer_experiments(df):
  new_df = df.copy(deep=True)
  new_df['ratio_microcluster_detection'] = df['microcluster_duration'] / df['detection_duration']
  new_df['ratio_den_explanation_detection'] = df['den_explainer_duration'] / df['detection_duration']
  new_df['ratio_coin_detection'] = df['coin_duration'] / df['detection_duration']
  return new_df


def simulate_denstream_explainer(den, detector_denstream_rs, X, window_size, nth_window,
                                 features=None,
                                 max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                 regularization='l1', regularization_param=1,
                                 intercept_scaling=1, simple_feature_contribution=True):
  results = {}
  if isinstance(detector_denstream_rs[1], float):  # thread 1 was for outlier detection
    detection_results = detector_denstream_rs[0][0]
    results['detection_duration'] = detector_denstream_rs[0][1]
    results['microcluster_duration'] = detector_denstream_rs[1]
  else:  # thread 1 was for microcluster generation
    detection_results = detector_denstream_rs[1][0]
    results['detection_duration'] = detector_denstream_rs[1][1]
    results['microcluster_duration'] = detector_denstream_rs[0]

  # run the DenStream microcluster based explanation
  # logging.info('Run den-microcluster based xplanation')
  start = time.perf_counter()
  outlier_indices = np.where(detection_results == 1)[0]
  # rads = [mc.get_radius() for mc in den.microclusters]
  # max_distance = max(rads) * max_rad_multiplier

  max_distance = np.std(X) * max_rad_multiplier

  explainer = DenstreamExplainer(den.p_microclusters, den.c_microclusters,
                                 max_distance, den.n_attributes,
                                 features=features,
                                 regularization=regularization,
                                 regularization_param=regularization_param,
                                 intercept_scaling=intercept_scaling,
                                 simple_feature_contribution=simple_feature_contribution,
                                 o_microclusters=den.o_microclusters)
  explanations = {}
  for i in list(outlier_indices):
    outlier = X[i, :]
    o_idx = window_size * nth_window + i
    explanations[o_idx] = explainer.explain_outlier(outlier, round_flag, prior_knowledge)
  duration = time.perf_counter() - start

  results['explanation_duration'] = duration
  results['explanation_result'] = explanations
  results['total_data'] = X.shape[0]
  results['total_outliers'] = len(outlier_indices)
  results['total_features'] = X.shape[1]
  return results
