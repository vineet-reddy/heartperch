# coding=utf-8
"""Aggregate window-level predictions to recording- and patient-level metrics.

Assumes validation/eval pipeline keeps metadata: 'recording_id', 'patient_id'.
Reads TensorBoard scalars are already handled by trainer; here we compute
explicit AUC/PR at higher aggregation levels and write CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_predictions(df: pd.DataFrame, prob_col: str = 'murmur_prob') -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate window → recording → patient level.
  
  Args:
    df: DataFrame with columns: recording_id, patient_id, {prob_col}, label
    prob_col: Name of probability column to aggregate
    
  Returns:
    (recording_df, patient_df) with aggregated predictions
  """
  # Recording level: mean and max over windows
  rec = df.groupby('recording_id').agg({
      prob_col: ['mean', 'max'],
      'label': 'max',
      'patient_id': 'first',
  }).reset_index()
  rec.columns = ['recording_id', 'prob_mean', 'prob_max', 'label', 'patient_id']
  
  # Patient level: max over sites (using mean across windows per site)
  site = df.groupby(['patient_id', 'recording_id']).agg({
      prob_col: 'mean',
      'label': 'max',
  }).reset_index()
  site.columns = ['patient_id', 'recording_id', 'prob_site_mean', 'label']
  
  patient = site.groupby('patient_id').agg({
      'prob_site_mean': 'max',
      'label': 'max',
  }).reset_index()
  patient.columns = ['patient_id', 'prob_mean', 'label']
  
  return rec, patient


def aggregate(pred_csv: Path, out_csv: Path) -> None:
  df = pd.read_csv(pred_csv)
  # Expected columns: recording_id, patient_id, segment_id, murmur_logit, label
  if 'murmur_prob' not in df.columns:
    if 'murmur_logit' in df.columns:
      df['murmur_prob'] = 1.0 / (1.0 + np.exp(-df['murmur_logit'].astype(float)))
    else:
      raise ValueError('Missing murmur_prob or murmur_logit in predictions CSV')

  rec, patient = aggregate_predictions(df, prob_col='murmur_prob')

  # Write outputs
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  rec.to_csv(out_csv.with_suffix('.recording.csv'), index=False)
  patient.to_csv(out_csv.with_suffix('.patient.csv'), index=False)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--preds_csv', required=True, type=Path)
  ap.add_argument('--out_csv', required=True, type=Path)
  args = ap.parse_args()
  aggregate(args.preds_csv, args.out_csv)


if __name__ == '__main__':
  main()


