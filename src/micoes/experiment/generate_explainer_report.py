import numpy as np
import pandas as pd


def get_detection_result(detector_rs):
    """
    detector_rs is a list
    """
    results = {}
    if isinstance(detector_rs[1], float):  # when thread 1 was for outlier detection
        if isinstance(detector_rs[0], np.ndarray):
            results['n_data'] = detector_rs[0].shape[0]
            results['n_outlier'] = np.where(detector_rs[0] == 1)[0].shape[0]
        else:
            results['n_data'] = detector_rs[0][0].shape[0]
            results['n_outlier'] = np.where(detector_rs[0][0] == 1)[0].shape[0]
    else:  # when thread 1 was for microcluster generation
        results['n_data'] = detector_rs[1][0].shape[0]
        results['n_outlier'] = np.where(detector_rs[1][0] == 1)[0].shape[0]
    return results


def build_temporal_df_microcluster_explainer(detection_results, explanation_results, explainer_type):
    """
    detection_results is a dictionary
    explanation_results is a dictionary
    explainer_type could be 'clu' or 'den'
    """
    detection_duration = list()
    microcluster_duration = list()
    explanation_duration = list()
    total_explanation_duration = list()
    total_data = list()
    total_outliers = list()
    coin_duration = list()
    windows = list()

    for window_idx in explanation_results.keys():
        detection_duration.append(explanation_results[window_idx]['detection_duration'])
        microcluster_duration.append(explanation_results[window_idx]['microcluster_duration'])
        explanation_duration.append(explanation_results[window_idx]['explanation_duration'])
        total_explanation_duration.append(np.sum((np.maximum(explanation_results[window_idx]['detection_duration'],
                                                             explanation_results[window_idx]['microcluster_duration']),
                                                  explanation_results[window_idx]['explanation_duration']), axis=0))

        detector_rs = get_detection_result(detection_results[window_idx])
        total_data.append(detector_rs['n_data'])
        total_outliers.append(detector_rs['n_outlier'])
        windows.append(window_idx)
        coin_duration.append(np.sum((explanation_results[window_idx]['coin_duration'],
                                     explanation_results[window_idx]['detection_duration']), axis=0))

    experiments = {}
    experiments['windows'] = windows
    experiments['detection_duration'] = detection_duration
    experiments['microcluster_duration'] = microcluster_duration
    experiments['explanation_duration'] = explanation_duration
    if explainer_type == 'clu':
        experiments['clu_coin_explainer_duration'] = total_explanation_duration
    elif explainer_type == 'den':
        experiments['den_coin_explainer_duration'] = total_explanation_duration
    experiments['coin_duration'] = coin_duration
    experiments['n_data'] = total_data
    experiments['n_outlier'] = total_outliers

    return pd.DataFrame(experiments)


def build_summary_temporal_microcluster_explainer(experiments_df, explainer_type):
    total_data = np.sum(experiments_df['n_data'])
    total_outliers = np.sum(experiments_df['n_outlier'])
    total_detection_duration = np.sum(experiments_df['detection_duration'])
    total_microcluster_duration = np.sum(experiments_df['microcluster_duration'])
    total_explanation_duration = np.sum(experiments_df['explanation_duration'])

    total_microcluster_coin_explainer_duration = 0
    if explainer_type == 'clu':
        total_microcluster_coin_explainer_duration = np.sum(experiments_df['clu_coin_explainer_duration'])
    elif explainer_type == 'den':
        total_microcluster_coin_explainer_duration = np.sum(experiments_df['den_coin_explainer_duration'])

    total_coin_duration = np.sum(experiments_df['coin_duration'])

    ratio_microcluster_explainer_over_coin = total_microcluster_coin_explainer_duration / total_coin_duration

    return (total_data, total_outliers, total_detection_duration, total_microcluster_duration, total_explanation_duration, total_microcluster_coin_explainer_duration, total_coin_duration, ratio_microcluster_explainer_over_coin)


def build_df_summary_temporal_microcluster_explainer(stream_names, dfs, explainer_type, n_features):
    explainer = 'total_microcluster_based_explainer_duration'
    if explainer_type == 'clu':
        explainer = 'total_clu_coin_explainer_duration'
    elif explainer_type == 'den':
        explainer = 'total_den_coin_explainer_duration'

    columns = ['stream_name', 'n_features', 'total_data', 'total_outliers', 'total_detection_duration',
               'total_microcluster_duration', 'total_explanation_duration', explainer,
               'total_coin_duration', 'ratio_microcluster_explainer_over_coin']
    summary_df = pd.DataFrame(columns=columns)

    for i in range(len(stream_names)):
        info = build_summary_temporal_microcluster_explainer(dfs[i], explainer_type)
        summary_df.loc[i] = [stream_names[i], n_features[i]] + list(info)
    return summary_df
