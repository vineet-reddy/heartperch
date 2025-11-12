# coding=utf-8
"""Train a logistic regression classifier on Perch v2 embeddings.

This script loads pre-computed embeddings from the CirCor dataset,
trains a simple logistic regression model, and saves it for evaluation.

Usage:
  python -m pulse.train.linear_probe \
    --mode binary \
    --embedding_dir ./embeddings \
    --output_dir ./models/linear_probe
  
  python -m pulse.train.linear_probe \
    --mode 3class \
    --embedding_dir ./embeddings \
    --output_dir ./models/linear_probe
"""

import argparse
import os
from pathlib import Path
import tempfile

from google.cloud import storage
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def load_embeddings_from_gcs(gcs_path: str, split: str) -> dict:
  """Load embeddings from GCS.
  
  Args:
    gcs_path: GCS path like gs://bucket/path/to/embeddings/
    split: 'train' or 'valid'
    
  Returns:
    Dictionary with embeddings, labels, and metadata
  """
  # Parse GCS path
  bucket_name = gcs_path.split('/')[2]
  blob_path = '/'.join(gcs_path.split('/')[3:])
  if not blob_path.endswith('/'):
    blob_path += '/'
  blob_path += f'embeddings_{split}.npz'
  
  # Download from GCS
  print(f'Downloading from gs://{bucket_name}/{blob_path}...')
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_path)
  
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    blob.download_to_filename(temp_file.name)
    data = np.load(temp_file.name)
    result = {key: data[key] for key in data.files}
  
  os.unlink(temp_file.name)
  return result


def load_embeddings_local(local_path: str, split: str) -> dict:
  """Load embeddings from local disk.
  
  Args:
    local_path: Local directory path
    split: 'train' or 'valid'
    
  Returns:
    Dictionary with embeddings, labels, and metadata
  """
  file_path = Path(local_path) / f'embeddings_{split}.npz'
  print(f'Loading from {file_path}...')
  data = np.load(file_path)
  return {key: data[key] for key in data.files}


def load_embeddings(path: str, split: str) -> dict:
  """Load embeddings from GCS or local disk.
  
  Args:
    path: GCS path (gs://...) or local directory path
    split: 'train' or 'valid'
    
  Returns:
    Dictionary with embeddings, labels, and metadata
  """
  if path.startswith('gs://'):
    return load_embeddings_from_gcs(path, split)
  else:
    return load_embeddings_local(path, split)


def save_model_to_gcs(model, gcs_path: str):
  """Save model to GCS.
  
  Args:
    model: Trained sklearn model
    gcs_path: GCS path like gs://bucket/path/to/model.joblib
  """
  # Save to temp file
  with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as temp_file:
    joblib.dump(model, temp_file.name)
    
    # Upload to GCS
    bucket_name = gcs_path.split('/')[2]
    blob_path = '/'.join(gcs_path.split('/')[3:])
    
    print(f'Uploading model to gs://{bucket_name}/{blob_path}...')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(temp_file.name)
    
    os.unlink(temp_file.name)


def save_model_local(model, local_path: str):
  """Save model to local disk.
  
  Args:
    model: Trained sklearn model
    local_path: Local file path
  """
  Path(local_path).parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(model, local_path)


def save_model(model, path: str):
  """Save model to GCS or local disk.
  
  Args:
    model: Trained sklearn model
    path: GCS path (gs://...) or local file path
  """
  if path.startswith('gs://'):
    save_model_to_gcs(model, path)
  else:
    save_model_local(model, path)


def main():
  parser = argparse.ArgumentParser(
      description='Train logistic regression on Perch embeddings'
  )
  parser.add_argument(
      '--embedding_dir',
      type=str,
      required=True,
      help='Directory containing embeddings (local or gs://)',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='Output directory for trained model (local or gs://)',
  )
  parser.add_argument(
      '--max_iter',
      type=int,
      default=100000,
      help='Maximum iterations for logistic regression (high value ensures convergence)',
  )
  parser.add_argument(
      '--class_weight',
      type=str,
      default='balanced',
      help='Class weights for logistic regression',
  )
  parser.add_argument(
      '--C',
      type=float,
      default=1.0,
      help='Regularization strength (inverse)',
  )
  parser.add_argument(
      '--mode',
      type=str,
      required=True,
      choices=['binary', '3class'],
      help='Classification mode: binary or 3class',
  )
  
  args = parser.parse_args()
  
  # Load training embeddings
  print('=== Loading training data ===')
  train_data = load_embeddings(args.embedding_dir, 'train')
  
  X_train = train_data['embeddings']
  y_train = train_data['labels']
  
  print(f'Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features')
  print(f'  Positive class: {y_train.sum()} / {len(y_train)} '
        f'({100 * y_train.mean():.1f}%)')
  
  # Train logistic regression
  print('\n=== Training logistic regression ===')
  model = LogisticRegression(
      max_iter=args.max_iter,
      class_weight=args.class_weight if args.class_weight != 'None' else None,
      C=args.C,
      random_state=42,
      verbose=1,
  )
  
  model.fit(X_train, y_train)
  print('Training complete!')
  
  # Evaluate on training set
  print('\n=== Training set performance (window-level) ===')
  y_pred_train = model.predict(X_train)
  y_prob_train = model.predict_proba(X_train)
  train_acc = accuracy_score(y_train, y_pred_train)
  
  if args.mode == 'binary':
    train_auc = roc_auc_score(y_train, y_prob_train[:, 1])
    train_ap = average_precision_score(y_train, y_prob_train[:, 1])
    target_names = ['Absent', 'Present']
  else:  # 3class
    train_auc = roc_auc_score(y_train, y_prob_train, multi_class='ovr', average='macro')
    train_ap = average_precision_score(y_train, y_prob_train, average='macro')
    target_names = ['Absent', 'Present', 'Unknown']
  
  print(f'Accuracy: {train_acc:.4f}')
  print(f'AUC-ROC: {train_auc:.4f}')
  print(f'AUC-PR: {train_ap:.4f}')
  print('\nClassification Report:')
  print(classification_report(y_train, y_pred_train, target_names=target_names))
  print('\nNote: Use evaluate_linear_probe.py for recording/patient-level metrics')
  
  # Save model
  print('\n=== Saving model ===')
  if args.output_dir.startswith('gs://'):
    model_path = args.output_dir
    if not model_path.endswith('/'):
      model_path += '/'
    model_path += 'model.joblib'
  else:
    model_path = str(Path(args.output_dir) / 'model.joblib')
  
  save_model(model, model_path)
  print(f'Model saved to {model_path}')
  
  print('\n=== Training complete! ===')


if __name__ == '__main__':
  main()

