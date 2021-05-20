import pandas as pd
import numpy as np


def merge_dataframe(clu_df, den_df, time_unit):
  if 'window size' in list(clu_df.columns):
    df = clu_df[['window size', 'n features', 'total data', 'total outliers',
                 f'detection execution time ({time_unit})', f'coin execution time ({time_unit})']]
  else:
    df = clu_df[['stream name', 'n features', 'total data', 'total outliers',
                 f'coin execution time ({time_unit})']]
  df[f'clu-micoes execution time ({time_unit})'] = clu_df[f'clu-micoes execution time ({time_unit})']
  df[f'den-micoes execution time ({time_unit})'] = den_df[f'den-micoes execution time ({time_unit})']

  df[f'clu-micoes over coin execution time (%)'] = clu_df[f'clu-micoes over coin execution time (%)']
  df[f'den-micoes over coin execution time (%)'] = den_df[f'den-micoes over coin execution time (%)']

  df[f'coin matched (%)'] = clu_df[f'coin matched (%)']
  df[f'clu-micoes matched (%)'] = clu_df[f'clu-micoes matched (%)']
  df[f'den-micoes matched (%)'] = den_df[f'den-micoes matched (%)']
  return df


def rename_experiment_compactdf_columns(df, time_unit, microcluster_type='clu', round=2, multiplier=1):
  microcluster_explainer = f'total_{microcluster_type}_coin_explainer_duration'
  new_microcluster_explainer = f'{microcluster_type}-micoes execution time ({time_unit})'
  microcluster_matched = f'{microcluster_type}_matched'
  new_microcluster_matched = f'{microcluster_type}-micoes matched (%)'

  df['ratio_microcluster_explainer_over_coin'] = df['ratio_microcluster_explainer_over_coin'] * 100

  df['total_detection_duration'] = df['total_detection_duration'] * multiplier
  df['total_microcluster_duration'] = df['total_microcluster_duration'] * multiplier
  df['total_explanation_duration'] = df['total_explanation_duration'] * multiplier
  df[microcluster_explainer] = df[microcluster_explainer] * multiplier
  df['total_coin_duration'] = df['total_coin_duration'] * multiplier
  df[microcluster_matched] = df[microcluster_matched] * 100
  df['coin_matched'] = df['coin_matched'] * 100

  #df = round_dataframe_floating_points(df, microcluster_type, round)

  columns = {'stream_name': 'stream name',
             'n_features': 'n features',
             'total_data': 'total data',
             'total_outliers': 'total outliers',
             'total_detection_duration': f'detection execution time ({time_unit})',
             'total_microcluster_duration': f'{microcluster_type} microcluster execution time ({time_unit})',
             'total_explanation_duration': f'feature contribution execution time ({time_unit})',
             microcluster_explainer: new_microcluster_explainer,
             'total_coin_duration': f'coin execution time ({time_unit})',
             'ratio_microcluster_explainer_over_coin': f'{microcluster_type}-micoes over coin execution time (%)',
             microcluster_matched: new_microcluster_matched,
             'coin_matched': 'coin matched (%)'}

  df = df.rename(columns=columns)

  return df


def round_dataframe_floating_points(df, microcluster_type, round=2):
  microcluster_explainer = f'total_{microcluster_type}_coin_explainer_duration'
  columns = {'total_detection_duration': round,
             'total_microcluster_duration': round,
             'total_explanation_duration': round,
             microcluster_explainer: round,
             'total_coin_duration': round,
             'ratio_microcluster_explainer_over_coin': round}

  df = df.round(columns)
  return df


def add_window_size_column(df, window_sizes):
  columns = list(df.columns.values)[1:]
  df['window size'] = window_sizes
  df = df[['stream name', 'window size'] + columns]
  return df
