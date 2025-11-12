# coding=utf-8
"""Evaluate linear probe on validation embeddings.

Usage:
  python -m pulse.examples.evaluate_linear_probe \
    --embedding_dir ./embeddings \
    --model_path ./models/linear_probe/model.joblib \
    --output_dir ./results
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pulse.examples.aggregate_heart_eval import aggregate_predictions
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def compute_confusion_metrics(y_true, y_prob, threshold=0.5):
  """Compute confusion matrix and derived metrics.
  
  Args:
    y_true: True binary labels
    y_prob: Predicted probabilities
    threshold: Decision threshold (default: 0.5)
    
  Returns:
    Dictionary with confusion matrix metrics
  """
  y_pred = (y_prob >= threshold).astype(int)
  cm = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = cm.ravel()
  
  return {
      'tn': int(tn),
      'fp': int(fp),
      'fn': int(fn),
      'tp': int(tp),
      'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
      'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
      'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
      'n_samples': int(tn + fp + fn + tp),
      'n_positive': int(fn + tp),
      'n_negative': int(tn + fp),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding_dir', required=True)
  parser.add_argument('--model_path', required=True)
  parser.add_argument('--output_dir', required=True)
  parser.add_argument('--split', default='valid')
  args = parser.parse_args()
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Load model
  print('=== Loading model ===')
  model = joblib.load(args.model_path)
  
  # Load embeddings
  print(f'\n=== Loading {args.split} embeddings ===')
  emb_path = Path(args.embedding_dir) / f'embeddings_{args.split}.npz'
  data = np.load(emb_path)
  
  X = data['embeddings']
  y = data['labels']
  recording_ids = data['recording_ids']
  patient_ids = data['patient_ids']
  segment_ids = data['segment_ids']
  
  print(f'{args.split.capitalize()}: {X.shape[0]} samples, {X.shape[1]} features')
  print(f'  Positive: {y.sum()}/{len(y)} ({100*y.mean():.1f}%)')
  
  # Generate predictions
  print(f'\n=== Generating predictions ===')
  y_prob = model.predict_proba(X)[:, 1]
  
  # Create window-level DataFrame
  window_df = pd.DataFrame({
      'recording_id': [r.decode('utf-8') if isinstance(r, bytes) else r for r in recording_ids],
      'patient_id': [p.decode('utf-8') if isinstance(p, bytes) else p for p in patient_ids],
      'segment_id': segment_ids,
      'murmur_prob': y_prob,
      'label': y,
  })
  
  # Save window-level
  window_df.to_csv(output_dir / 'predictions_window.csv', index=False)
  
  # Aggregate predictions
  print('\n=== Aggregating predictions ===')
  rec_df, patient_df = aggregate_predictions(window_df, prob_col='murmur_prob')
  
  rec_df.to_csv(output_dir / 'predictions_recording.csv', index=False)
  patient_df.to_csv(output_dir / 'predictions_patient.csv', index=False)
  
  # Compute metrics
  print('\n=== Evaluation Results ===')
  results = []
  
  for level, df, col in [
      ('patient', patient_df, 'prob_mean'),
      ('recording_mean', rec_df, 'prob_mean'),
      ('recording_max', rec_df, 'prob_max'),
      ('window', window_df, 'murmur_prob'),
  ]:
    auc_roc = roc_auc_score(df['label'], df[col])
    auc_pr = average_precision_score(df['label'], df[col])
    n_pos = int(df['label'].sum())
    n_total = len(df)
    
    results.append({
        'level': level,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'n_samples': n_total,
        'n_positive': n_pos,
    })
    
    print(f'{level:18s} | AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f} | '
          f'N={n_total:4d} (Pos={n_pos:3d})')
  
  # Compute confusion matrices
  print('\n=== Confusion Matrices (threshold=0.5) ===')
  confusion_results = []
  
  for level, df, col in [
      ('patient', patient_df, 'prob_mean'),
      ('recording_mean', rec_df, 'prob_mean'),
      ('recording_max', rec_df, 'prob_max'),
      ('window', window_df, 'murmur_prob'),
  ]:
    cm_metrics = compute_confusion_metrics(df['label'].values, df[col].values)
    cm_metrics['level'] = level
    confusion_results.append(cm_metrics)
    
    print(f'\n{level}:')
    print(f'  TN={cm_metrics["tn"]:4d}, FP={cm_metrics["fp"]:4d}, '
          f'FN={cm_metrics["fn"]:4d}, TP={cm_metrics["tp"]:4d}')
    print(f'  Sensitivity: {100*cm_metrics["sensitivity"]:5.1f}%, '
          f'Specificity: {100*cm_metrics["specificity"]:5.1f}%, '
          f'Precision: {100*cm_metrics["precision"]:5.1f}%')
  
  # Save metrics
  pd.DataFrame(results).to_csv(output_dir / 'metrics.csv', index=False)
  pd.DataFrame(confusion_results).to_csv(output_dir / 'confusion_matrices.csv', index=False)
  print(f'\nResults saved to {output_dir}/')
  print(f'  - metrics.csv (AUC-ROC, AUC-PR)')
  print(f'  - confusion_matrices.csv (TN, FP, FN, TP, sensitivity, specificity, precision)')


if __name__ == '__main__':
  main()

