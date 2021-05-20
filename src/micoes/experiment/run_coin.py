import numpy as np
import pandas as pd


from micoes.coin import OutlierInterpreter
from micoes.detectors import run_detection
from micoes.explainer.common import map_feature_scores

#from timeit import default_timer as timer

import time

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_coin_explainer(X, y_pred, sgnf_prior=1, feature_names=None, window_size=60, nth_window=0):
    start = time.perf_counter()
    interpreter = OutlierInterpreter(data=X,
                                     inds_otlr=y_pred,
                                     nbrs_ratio=0.1,
                                     AUG=10,
                                     MIN_CLUSTER_SIZE=5,
                                     MAX_NUM_CLUSTER=4,
                                     VAL_TIMES=10,
                                     C_SVM=1.,
                                     RESOLUTION=0.05,
                                     THRE_PS=0.85,
                                     DEFK=0)
    ids_target = np.where(y_pred == 1)[0]
    importance_attr, outlierness, o_dictionary = interpreter.interpret_outliers(ids_target, sgnf_prior, int_flag=1)
    feature_scores_dict = {}
    for o_idx, feature_scores in zip(outlierness.keys(), importance_attr):
        idx = window_size * nth_window + o_idx
        if feature_names is not None:
            feature_scores_dict[idx] = map_feature_scores(feature_names, feature_scores)
        else:
            feature_scores_dict[idx] = feature_scores
    explanation = {'outlierness': outlierness, 'feature_scores': feature_scores_dict}
    interpreter_duration = time.perf_counter() - start
    return (interpreter_duration, explanation)


def run_coin_lof(X, sgnf_prior=1):
    results = {}

    # run outlier detection
    y_pred, detection_duration = run_detection(X, detector_type='lof', profiling=True)
    results['detection_duration'] = detection_duration

    # run outlier explanation
    start = time.perf_counter()
    interpreter = OutlierInterpreter(data=X,
                                     inds_otlr=y_pred,
                                     nbrs_ratio=0.1,
                                     AUG=10,
                                     MIN_CLUSTER_SIZE=5,
                                     MAX_NUM_CLUSTER=4,
                                     VAL_TIMES=10,
                                     C_SVM=1.,
                                     RESOLUTION=0.05,
                                     THRE_PS=0.85,
                                     DEFK=0)
    ids_target = np.where(y_pred == 1)[0]
    importance_attr, outlierness, o_dictionary = interpreter.interpret_outliers(ids_target, sgnf_prior, int_flag=1)
    interpreter_duration = time.perf_counter() - start

    results['explanation_duration'] = interpreter_duration
    results['explanation_result'] = {'outlierness': outlierness, 'feature_scores': importance_attr}
    results['total_data'] = X.shape[0]
    results['total_outliers'] = len(ids_target)
    results['total_features'] = X.shape[1]
    return results


def run_coin_experiment_on_each_file(filename, detector_type='lof', sgnf_prior=1):
    raw = pd.read_csv(filename)
    data = raw.drop(['label'], axis=1).values
    X = data
    result = run_coin_lof(X, sgnf_prior)
    return result


def build_dataframe_coin_experiments_basic(folder, filenames, detector_type='lof', sgnf_prior=1):
    stream_names = list()
    detection_duration = list()
    explanation_duration = list()
    explanation_result = list()
    total_data = list()
    total_features = list()
    total_outliers = list()

    for filename in filenames:
        filepath = folder + filename
        stream_name = filename.replace('.csv', '')
        logging.info(f'Running on {stream_name} -----------------')
        stream_names.append(stream_name)
        result = run_coin_experiment_on_each_file(filepath, detector_type, sgnf_prior)
        detection_duration.append(result['detection_duration'])
        explanation_duration.append(result['explanation_duration'])
        explanation_result.append(result['explanation_result'])
        total_data.append(result['total_data'])
        total_outliers.append(result['total_outliers'])
        total_features.append(result['total_features'])

    experiments = {}
    experiments['stream_names'] = stream_names
    experiments['detection_duration'] = detection_duration
    experiments['explanation_duration'] = explanation_duration
    experiments['total_data'] = total_data
    experiments['total_outliers'] = total_outliers
    experiments['total_features'] = total_features

    explanations = {}
    explanations['stream_names'] = stream_names
    explanations['explanation_result'] = explanation_result

    return pd.DataFrame(experiments), explanations


def build_compact_df_coin_experiments(df):
    df['ratio_explanation_detection'] = df['explanation_duration'] / df['detection_duration']
    return df
