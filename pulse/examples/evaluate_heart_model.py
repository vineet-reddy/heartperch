# coding=utf-8
"""End-to-end heart murmur evaluation: predictions + aggregation + metrics."""

from pathlib import Path

from absl import app, flags, logging
from chirp import config_utils
from chirp.configs import config_globals
from chirp.data import utils as data_utils
from chirp.models import output
from chirp.train import classifier
import flax.jax_utils as flax_utils
from ml_collections.config_flags import config_flags
import numpy as np
import pandas as pd
from pulse.examples.aggregate_heart_eval import aggregate_predictions
from sklearn.metrics import roc_auc_score, average_precision_score
import tqdm

_CONFIG = config_flags.DEFINE_config_file('config')
_WORKDIR = flags.DEFINE_string('workdir', None, 'Model checkpoint directory')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory')
flags.mark_flags_as_required(['config', 'workdir', 'output_dir'])


def predict(model_bundle, train_state, dataset, output_head='murmur'):
  """Generate window-level predictions with metadata."""
  train_state = flax_utils.replicate(train_state)
  predictions = []
  
  for batch in tqdm.tqdm(dataset.as_numpy_iterator(), desc='Predicting'):
    variables = {"params": train_state.params[0], **train_state.model_state}
    model_outputs = model_bundle.model.apply(variables, batch["audio"], train=False)
    logits = output.output_head_logits(model_outputs, model_bundle.output_head_metadatas)
    
    batch_logits = logits[f'{output_head}_logits']
    batch_size = len(batch["label"])
    
    for i in range(batch_size):
      predictions.append({
          'recording_id': batch['recording_id'][i].decode('utf-8'),
          'patient_id': batch['patient_id'][i].decode('utf-8'),
          'segment_id': int(batch.get('segment_id', np.arange(batch_size))[i]),
          'logit': float(batch_logits[i, 1]),  # Index 1 = positive class for binary classification
          'label': int(batch['label'][i]),  # Labels are [B], not [B, 1]
      })
  
  return pd.DataFrame(predictions)


def aggregate(df):
  """Convert logits to probabilities and aggregate using shared logic."""
  df['prob'] = 1.0 / (1.0 + np.exp(-df['logit']))
  return aggregate_predictions(df, prob_col='prob')


def compute_metrics(df, pred_col):
  """Compute AUC-ROC and AUC-PR."""
  y_true = df['label'].values
  y_pred = df[pred_col].values
  return {
      'auc_roc': roc_auc_score(y_true, y_pred),
      'auc_pr': average_precision_score(y_true, y_pred),
      'n_samples': len(df),
      'n_positive': int(y_true.sum()),
  }


def main(argv):
  config = config_utils.parse_config(_CONFIG.value, config_globals.get_globals())
  out_dir = Path(_OUTPUT_DIR.value)
  out_dir.mkdir(parents=True, exist_ok=True)
  
  # Load model
  model_bundle, train_state = classifier.initialize_model(
      workdir=_WORKDIR.value, **config.init_config
  )
  train_state = model_bundle.ckpt.restore(train_state)
  
  # Load dataset
  dataset, _ = data_utils.get_dataset(**config.eval_dataset_config.valid)
  
  # Predict
  window_df = predict(model_bundle, train_state, dataset)
  window_df.to_csv(out_dir / 'predictions_window.csv', index=False)
  
  # Aggregate
  rec_df, pat_df = aggregate(window_df)
  rec_df.to_csv(out_dir / 'predictions_recording.csv', index=False)
  pat_df.to_csv(out_dir / 'predictions_patient.csv', index=False)
  
  # Metrics
  results = []
  results.append({'level': 'window', **compute_metrics(window_df, 'prob')})
  results.append({'level': 'recording_mean', **compute_metrics(rec_df, 'prob_mean')})
  results.append({'level': 'recording_max', **compute_metrics(rec_df, 'prob_max')})
  results.append({'level': 'patient', **compute_metrics(pat_df, 'prob_mean')})
  
  metrics_df = pd.DataFrame(results)
  metrics_df.to_csv(out_dir / 'metrics.csv', index=False)
  
  # Print summary
  logging.info('\nResults:')
  for _, row in metrics_df.iterrows():
    logging.info(f"{row['level']:18s} AUC-ROC: {row['auc_roc']:.4f}  AUC-PR: {row['auc_pr']:.4f}")
  logging.info(f'\nSaved to {out_dir}')


if __name__ == '__main__':
  app.run(main)
