import numpy as np
import pandas as pd
import math
import time
import random

from skmultiflow.data import TemporalDataStream

from micoes.microclusters.clustream import CluStream
from micoes.microclusters.denstream import DenStream

from micoes.utils import generate_timestamp

from micoes.multiprocessing.detection_microcluster import run_detector_microcluster_serial

from micoes.experiment.run_coin import run_coin_explainer

from micoes.experiment.run_clustream_explainer import simulate_clustream_explainer
from micoes.experiment.run_denstream_explainer import simulate_denstream_explainer

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_temporal_explainer_comparison(filepath,
                                      n_microclusters, alpha, tau,
                                      lamda, mu, beta, eta,
                                      label='label', init_percentage=10, detector_type='lof',
                                      max_rad_multiplier=2, round_flag=True, prior_knowledge=1,
                                      regularization='l1', regularization_param=1,
                                      intercept_scaling=1, simple_feature_contribution=True,
                                      arrival_rate='Fixed', time_unit='seconds', time_interval=1, time_window=None,
                                      window_size=60, sliding_size=60, delay_time=1, break_window=None):

  raw = pd.read_csv(filepath)
  dataX = None
  if label:
    dataX = raw.drop(['label'], axis=1)
    datay = raw[['label']].values
  else:
    dataX = raw
  features = tuple(dataX.columns)
  dataX = dataX.values
  n_features = dataX.shape[1]

  # use init_percentage of the data for cluster initialization
  init = math.ceil(dataX.shape[0] * init_percentage / 100)
  if init < n_microclusters:
    init = n_microclusters

  randomlist = []
  random.seed(1)
  for i in range(0, init):
    n = random.randint(0, (dataX.shape[0] - 1))
    randomlist.append(n)

  if not n_microclusters:
    n_microclusters = init

  Xinit = dataX[randomlist, :]

  clu = CluStream(n_microclusters=n_microclusters,
                  n_attributes=n_features,
                  alpha=alpha,
                  tau=tau)
  clu.initialize(Xinit)

  den = DenStream(lamda=lamda,
                  mu=mu,
                  beta=beta,
                  eta=eta,
                  n_attributes=n_features)
  den.initialize(Xinit)

  time = generate_timestamp(dataX.shape[0],
                            arrival_rate=arrival_rate,
                            time_unit=time_unit,
                            time_interval=time_interval)

  # Set a delay of 1 minute
  delay_time = np.timedelta64(delay_time, "s")

  # Set the stream source (producer)
  tstream = TemporalDataStream(dataX, datay, time, sample_delay=delay_time, ordered=True)

  # data points keep coming and are added to the window
  detection_results = {}
  cluexplainer_results = {}
  denexplainer_results = {}

  idx = 0
  logging.info("Running stream...")
  while tstream.has_more_samples():
    window_idx = 'window_' + str(idx)
    X_online, y_online, arrival_time_online, available_time_online, sample_weight_online = tstream.next_sample(window_size)
    results = run_detector_microcluster_serial(X_online, clu, den, detector_type, profiling=True)

    detection_results[window_idx] = results[0]

    # logging.info('Run original coin explanation')
    interpreter_duration, coin_explanation = run_coin_explainer(X_online, detection_results[window_idx][0],
                                                                sgnf_prior=1,
                                                                feature_names=features,
                                                                window_size=window_size,
                                                                nth_window=idx)

    detector_clu = results[0:2]
    cluexplainer_results[window_idx] = simulate_clustream_explainer(clu, detector_clu, X_online,
                                                                    window_size=window_size, nth_window=idx,
                                                                    features=features,
                                                                    max_rad_multiplier=max_rad_multiplier,
                                                                    round_flag=round_flag,
                                                                    prior_knowledge=prior_knowledge,
                                                                    regularization=regularization,
                                                                    regularization_param=regularization_param,
                                                                    intercept_scaling=intercept_scaling,
                                                                    simple_feature_contribution=simple_feature_contribution)
    cluexplainer_results[window_idx]['coin_duration'] = interpreter_duration
    cluexplainer_results[window_idx]['coin_result'] = coin_explanation

    results.pop(1)
    detector_den = results
    denexplainer_results[window_idx] = simulate_denstream_explainer(den, detector_den, X_online,
                                                                    window_size=window_size, nth_window=idx,
                                                                    features=features,
                                                                    max_rad_multiplier=max_rad_multiplier,
                                                                    round_flag=round_flag,
                                                                    prior_knowledge=prior_knowledge,
                                                                    regularization=regularization,
                                                                    regularization_param=regularization_param,
                                                                    intercept_scaling=intercept_scaling,
                                                                    simple_feature_contribution=simple_feature_contribution)
    denexplainer_results[window_idx]['coin_duration'] = interpreter_duration
    denexplainer_results[window_idx]['coin_result'] = coin_explanation
    den.check_microclusters_status(period=window_size)

    idx += 1
    if break_window is not None and idx > break_window:
      break
  detection_results['n_features'] = dataX.shape[1]
  return detection_results, cluexplainer_results, denexplainer_results
