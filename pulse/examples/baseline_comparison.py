# coding=utf-8
"""Compare Perch embeddings against baseline feature extractors.

Usage:
  python -m pulse.examples.baseline_comparison \
    --embedding_dir ./embeddings \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./results
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm

from pulse.data import circor
from pulse.preprocessing import heart_ops
from pulse.examples.aggregate_heart_eval import aggregate_predictions
from pulse.configs import heart_presets
from chirp.preprocessing import pipeline


def extract_random_features(n_samples, n_features=1280, seed=42):
  """Random Gaussian embeddings baseline."""
  np.random.seed(seed)
  return np.random.randn(n_samples, n_features).astype(np.float32)


def extract_mfcc_features(audio_windows, sr=32000, n_mfcc=40):
  """Extract MFCC features from audio windows."""
  features = []
  for audio in tqdm(audio_windows, desc='Extracting MFCCs', leave=False):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Get statistics across time: mean, std, min, max
    feat = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.min(axis=1),
        mfcc.max(axis=1)
    ])
    features.append(feat)
  return np.array(features)


def extract_spectral_features(audio_windows, sr=32000):
  """Extract basic spectral features."""
  features = []
  for audio in tqdm(audio_windows, desc='Extracting spectral features', leave=False):
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    feat = np.array([
        cent.mean(), cent.std(),
        rolloff.mean(), rolloff.std(),
        zcr.mean(), zcr.std(),
    ])
    features.append(feat)
  return np.array(features)


def extract_vggish_features(audio_windows, sr=32000):
  """Extract VGGish embeddings (general audio pre-training)."""
  model = hub.load('https://tfhub.dev/google/vggish/1')
  features = []
  for audio in tqdm(audio_windows, desc='Extracting VGGish', leave=False):
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    embeddings = model(audio_16k)
    features.append(np.mean(embeddings.numpy(), axis=0))
  return np.array(features)


def extract_beats_features(audio_windows, sr=32000):
  """Extract BEATs embeddings (SOTA audio event detection).
  
  BEATs (Bidirectional Encoder representation from Audio Transformers) is a
  self-supervised learning framework for audio representation pre-training.
  Paper: https://arxiv.org/abs/2212.09058
  Model: Microsoft BEATs from unilm repository
  """
  try:
    import torch
    from beats import BEATs, BEATsConfig
    
    print('  Loading Microsoft BEATs model...')
    # Official Microsoft checkpoint: BEATs_iter3_plus_AS2M (pretrained on AudioSet-2M)
    checkpoint_url = "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
    
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    features = []
    for audio in tqdm(audio_windows, desc='Extracting BEATs', leave=False):
      # Resample to 16kHz (BEATs requirement)
      audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
      
      # Convert to tensor and add batch dimension
      audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)
      
      # Extract features
      with torch.no_grad():
        embedding = model.extract_features(audio_tensor)[0]
        # Mean pool over time dimension
        embedding = embedding.mean(dim=1).squeeze().cpu().numpy()
      
      features.append(embedding)
    
    return np.array(features)
    
  except ImportError as e:
    print(f'  Warning: Could not import BEATs: {e}')
    print('  Ensure poetry install completed successfully')
    return None
  except Exception as e:
    print(f'  Warning: BEATs extraction failed: {e}')
    import traceback
    traceback.print_exc()
    return None


def load_audio_from_tfds(tfds_data_dir, split, train_fraction=0.8):
  """Load raw audio windows from TFDS using same preprocessing as embeddings."""
  # Get config to use same dataset version as embedding extraction
  config = heart_presets.get_heart_base_config()
  
  # Load dataset
  ds_builder = tfds.builder(config.dataset_directory, data_dir=tfds_data_dir)
  raw_dataset = ds_builder.as_dataset(split='train')
  
  # Apply same preprocessing as embedding extraction
  ops = [
      heart_ops.PatientHashSplit(split=split, train_fraction=train_fraction),
      pipeline.RepeatPadding(pad_size=config.window_size_s, sample_rate=config.sample_rate_hz, add_mask=False),
      heart_ops.MapMurmurToBinary(),
      pipeline.ExtractStridedWindows(
          window_length_sec=config.window_size_s,
          window_stride_sec=config.window_stride_s,
          sample_rate=config.sample_rate_hz,
      ),
  ]
  prep_pipeline = pipeline.Pipeline(ops=ops)
  processed_dataset = prep_pipeline(raw_dataset, ds_builder.info)
  
  # Extract audio, labels, and metadata
  audio_list = []
  label_list = []
  recording_id_list = []
  patient_id_list = []
  segment_id_list = []
  
  for batch in tqdm(processed_dataset.as_numpy_iterator(), desc=f'Loading {split} audio', leave=False):
    audio_list.append(batch['audio'])
    label_list.append(batch['label'])
    
    rec_id = batch['recording_id']
    recording_id_list.append(rec_id.decode('utf-8') if isinstance(rec_id, bytes) else str(rec_id))
    
    pat_id = batch['patient_id']
    patient_id_list.append(pat_id.decode('utf-8') if isinstance(pat_id, bytes) else str(pat_id))
    
    segment_id_list.append(batch['segment_id'])
  
  return {
      'audio': np.array(audio_list),
      'labels': np.array(label_list),
      'recording_ids': np.array(recording_id_list),
      'patient_ids': np.array(patient_id_list),
      'segment_ids': np.array(segment_id_list),
  }


def align_datasets(emb_data, tfds_data):
  """Align embedding and TFDS data by sorting on (recording_id, segment_id).
  
  TensorFlow datasets don't guarantee deterministic iteration order, so we need
  to sort both datasets by a unique key to ensure proper alignment.
  """
  # Create sort keys
  emb_keys = [f"{rec}_{seg}" for rec, seg in 
              zip(emb_data['recording_ids'], emb_data['segment_ids'])]
  tfds_keys = [f"{rec}_{seg}" for rec, seg in 
               zip(tfds_data['recording_ids'], tfds_data['segment_ids'])]
  
  # Get sort indices
  emb_indices = np.argsort(emb_keys)
  tfds_indices = np.argsort(tfds_keys)
  
  # Sort all arrays
  emb_sorted = {key: val[emb_indices] for key, val in emb_data.items()}
  tfds_sorted = {key: val[tfds_indices] for key, val in tfds_data.items()}
  
  return emb_sorted, tfds_sorted


def train_and_evaluate(X_train, y_train, X_val, y_val, val_metadata, name, scale=True):
  """Train logistic regression and return metrics at recording and patient levels."""
  suffix = ' (scaled)' if scale else ' (unscaled)'
  print(f'  {name}{suffix}: {X_train.shape[0]} samples, {X_train.shape[1]} features', end=' ... ')
  
  # Optionally standardize features
  if scale:
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc = scaler.transform(X_val)
  else:
    X_train_proc = X_train
    X_val_proc = X_val
  
  model = LogisticRegression(
      max_iter=100000,  # High value to ensure convergence for all feature types
      class_weight='balanced',
      random_state=42,
      verbose=0,
  )
  model.fit(X_train_proc, y_train)
  print('done')
  
  # Evaluate at window level
  y_prob_train = model.predict_proba(X_train_proc)[:, 1]
  y_prob_val = model.predict_proba(X_val_proc)[:, 1]
  
  # Create DataFrame for aggregation (like evaluate_linear_probe.py does)
  window_df = pd.DataFrame({
      'recording_id': val_metadata['recording_ids'],
      'patient_id': val_metadata['patient_ids'],
      'segment_id': val_metadata['segment_ids'],
      'murmur_prob': y_prob_val,
      'label': y_val,
  })
  
  # Aggregate to recording and patient levels
  rec_df, patient_df = aggregate_predictions(window_df, prob_col='murmur_prob')
  
  return {
      'baseline': name + suffix,
      'n_features': X_train.shape[1],
      # PRIMARY METRICS - Recording level (whole WAV files)
      'recording_auc_roc_mean': roc_auc_score(rec_df['label'], rec_df['prob_mean']),
      'recording_auc_pr_mean': average_precision_score(rec_df['label'], rec_df['prob_mean']),
      'recording_auc_roc_max': roc_auc_score(rec_df['label'], rec_df['prob_max']),
      'recording_auc_pr_max': average_precision_score(rec_df['label'], rec_df['prob_max']),
      # FINAL METRIC - Patient level (clinical decision)
      'patient_auc_roc': roc_auc_score(patient_df['label'], patient_df['prob_mean']),
      'patient_auc_pr': average_precision_score(patient_df['label'], patient_df['prob_mean']),
      # Supplementary metrics
      'window_auc_roc': roc_auc_score(y_val, y_prob_val),
      'window_auc_pr': average_precision_score(y_val, y_prob_val),
      'train_auc': roc_auc_score(y_train, y_prob_train),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding_dir', default='./embeddings')
  parser.add_argument('--tfds_data_dir', default='~/tensorflow_datasets')
  parser.add_argument('--output_dir', default='./results')
  args = parser.parse_args()
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  tfds_data_dir = str(Path(args.tfds_data_dir).expanduser())
  
  print('=== Baseline Comparison ===\n')
  
  # Load Perch embeddings
  print('Loading Perch embeddings...')
  perch_train = np.load(Path(args.embedding_dir) / 'embeddings_train.npz')
  perch_val = np.load(Path(args.embedding_dir) / 'embeddings_valid.npz')
  
  emb_train = {
      'embeddings': perch_train['embeddings'],
      'labels': perch_train['labels'],
      'recording_ids': perch_train['recording_ids'],
      'patient_ids': perch_train['patient_ids'],
      'segment_ids': perch_train['segment_ids'],
  }
  emb_val = {
      'embeddings': perch_val['embeddings'],
      'labels': perch_val['labels'],
      'recording_ids': perch_val['recording_ids'],
      'patient_ids': perch_val['patient_ids'],
      'segment_ids': perch_val['segment_ids'],
  }
  
  # Get model name from saved embeddings (or fall back to dimension-based detection)
  if 'model_name' in perch_train:
    model_name = str(perch_train['model_name'])
    perch_model_name = f'Perch ({model_name})'
    print(f'  Model: {model_name}')
  else:
    # Fallback: Detect from embedding dimension (for legacy embedding files)
    emb_dim = emb_train['embeddings'].shape[1]
    perch_model_name = 'Perch'
    if emb_dim == 1280:
      perch_model_name = 'Perch (perch_8/surfperch, 1280-dim)'
    elif emb_dim == 1530:
      perch_model_name = 'Perch (perch_v2, 1530-dim)'
    else:
      perch_model_name = f'Perch ({emb_dim}-dim)'
    print(f'  Detected from dimension: {perch_model_name}')
  
  print(f'  Train: {len(emb_train["labels"])} samples ({emb_train["labels"].sum()} positive)')
  print(f'  Valid: {len(emb_val["labels"])} samples ({emb_val["labels"].sum()} positive)\n')
  
  # Load raw audio
  print('Loading raw audio from TFDS...')
  tfds_train = load_audio_from_tfds(tfds_data_dir, 'train')
  tfds_val = load_audio_from_tfds(tfds_data_dir, 'valid')
  print(f'  Loaded {len(tfds_train["audio"])} train, {len(tfds_val["audio"])} valid windows\n')
  
  # Align datasets (sort by recording_id + segment_id for deterministic ordering)
  print('Aligning datasets...')
  emb_train, tfds_train = align_datasets(emb_train, tfds_train)
  emb_val, tfds_val = align_datasets(emb_val, tfds_val)
  
  # Verify alignment
  if not np.array_equal(emb_train['labels'], tfds_train['labels']):
    raise ValueError('Train labels mismatch after alignment!')
  if not np.array_equal(emb_val['labels'], tfds_val['labels']):
    raise ValueError('Valid labels mismatch after alignment!')
  print('  Labels verified\n')
  
  # Extract references to aligned data
  X_perch_train, y_train = emb_train['embeddings'], emb_train['labels']
  X_perch_val, y_val = emb_val['embeddings'], emb_val['labels']
  audio_train, audio_val = tfds_train['audio'], tfds_val['audio']
  
  # Run comparisons
  # Extract all features first
  print('Extracting features...')
  X_random_train = extract_random_features(len(y_train), n_features=1280)
  X_random_val = extract_random_features(len(y_val), n_features=1280, seed=43)
  
  X_mfcc_train = extract_mfcc_features(audio_train)
  X_mfcc_val = extract_mfcc_features(audio_val)
  
  X_spec_train = extract_spectral_features(audio_train)
  X_spec_val = extract_spectral_features(audio_val)
  X_combined_train = np.concatenate([X_mfcc_train, X_spec_train], axis=1)
  X_combined_val = np.concatenate([X_mfcc_val, X_spec_val], axis=1)
  
  X_vggish_train = extract_vggish_features(audio_train)
  X_vggish_val = extract_vggish_features(audio_val)
  
  # Extract BEATs features (optional - may fail if transformers not installed)
  X_beats_train = extract_beats_features(audio_train)
  X_beats_val = extract_beats_features(audio_val)
  
  # Test scaling impact - run best scenario only
  results = []
  
  # Prepare validation metadata for patient-level aggregation
  val_metadata = {
      'recording_ids': emb_val['recording_ids'],
      'patient_ids': emb_val['patient_ids'],
      'segment_ids': emb_val['segment_ids'],
  }
  
  print('\nEvaluating baselines (optimal scaling):')
  # Perch unscaled (performs best without scaling)
  results.append(train_and_evaluate(X_perch_train, y_train, X_perch_val, y_val, val_metadata, perch_model_name, scale=False))
  # General audio pre-training baselines
  results.append(train_and_evaluate(X_vggish_train, y_train, X_vggish_val, y_val, val_metadata, 'VGGish', scale=False))
  if X_beats_train is not None:
    results.append(train_and_evaluate(X_beats_train, y_train, X_beats_val, y_val, val_metadata, 'BEATs', scale=False))
  # Traditional features scaled (helps convergence and performance)
  results.append(train_and_evaluate(X_random_train, y_train, X_random_val, y_val, val_metadata, 'Random', scale=True))
  results.append(train_and_evaluate(X_mfcc_train, y_train, X_mfcc_val, y_val, val_metadata, 'MFCC', scale=True))
  results.append(train_and_evaluate(X_combined_train, y_train, X_combined_val, y_val, val_metadata, 'MFCC+Spectral', scale=True))
  
  # Create comparison table (sort by recording-level AUC mean)
  df = pd.DataFrame(results).sort_values('recording_auc_roc_mean', ascending=False)
  
  print('\n' + '='*90)
  print('BASELINE COMPARISON RESULTS')
  print('='*90)
  print('\nPatient-level:')
  print(df[['baseline', 'patient_auc_roc', 'patient_auc_pr']].to_string(index=False))
  print('\nRecording-level:')
  print(df[['baseline', 'recording_auc_roc_mean', 'recording_auc_pr_mean', 'recording_auc_roc_max']].to_string(index=False))
  print('\nWindow-level:')
  print(df[['baseline', 'window_auc_roc', 'window_auc_pr']].to_string(index=False))
  print('='*90)
  
  # Save results
  output_file = output_dir / 'baseline_comparison.csv'
  df.to_csv(output_file, index=False)
  print(f'\nResults saved to {output_file}')
  
  # Summary
  print('\n' + '='*90)
  print('SUMMARY')
  print('='*90)
  
  perch_row = df[df['baseline'].str.contains('Perch')].iloc[0]
  best_baseline = df[~df['baseline'].str.contains('Perch')].iloc[0]
  
  print(f"\nPerch:")
  print(f"  Patient:    AUC-ROC={perch_row['patient_auc_roc']:.4f}, AUC-PR={perch_row['patient_auc_pr']:.4f}")
  print(f"  Recording:  AUC-ROC={perch_row['recording_auc_roc_mean']:.4f}, AUC-PR={perch_row['recording_auc_pr_mean']:.4f} (max={perch_row['recording_auc_roc_max']:.4f})")
  print(f"  Window:     AUC-ROC={perch_row['window_auc_roc']:.4f}, AUC-PR={perch_row['window_auc_pr']:.4f}")
  print(f"  Train:      AUC-ROC={perch_row['train_auc']:.4f}")
  
  print(f"\nBest Baseline ({best_baseline['baseline']}):")
  print(f"  Patient:    AUC-ROC={best_baseline['patient_auc_roc']:.4f}, AUC-PR={best_baseline['patient_auc_pr']:.4f}")
  print(f"  Recording:  AUC-ROC={best_baseline['recording_auc_roc_mean']:.4f}, AUC-PR={best_baseline['recording_auc_pr_mean']:.4f}")
  print(f"  Window:     AUC-ROC={best_baseline['window_auc_roc']:.4f}, AUC-PR={best_baseline['window_auc_pr']:.4f}")
  
  print(f"\nPerformance ranking (by recording-level AUC-ROC):")
  for i, (_, row) in enumerate(df.iterrows(), 1):
    print(f"  {i}. {row['baseline']:20s} Patient={row['patient_auc_roc']:.4f}, Recording={row['recording_auc_roc_mean']:.4f}, Window={row['window_auc_roc']:.4f}")
  
  print(f"\nDifferences (Perch vs Best Baseline):")
  print(f"  Patient:    {perch_row['patient_auc_roc'] - best_baseline['patient_auc_roc']:+.4f} AUC-ROC, {perch_row['patient_auc_pr'] - best_baseline['patient_auc_pr']:+.4f} AUC-PR")
  print(f"  Recording:  {perch_row['recording_auc_roc_mean'] - best_baseline['recording_auc_roc_mean']:+.4f} AUC-ROC, {perch_row['recording_auc_pr_mean'] - best_baseline['recording_auc_pr_mean']:+.4f} AUC-PR")
  print(f"  Window:     {perch_row['window_auc_roc'] - best_baseline['window_auc_roc']:+.4f} AUC-ROC, {perch_row['window_auc_pr'] - best_baseline['window_auc_pr']:+.4f} AUC-PR")
  
  # VGGish comparison
  vggish_row = df[df['baseline'].str.contains('VGGish')]
  if len(vggish_row) > 0:
    vggish_row = vggish_row.iloc[0]
    print(f"\nPerch vs VGGish:")
    print(f"  Patient:    {perch_row['patient_auc_roc'] - vggish_row['patient_auc_roc']:+.4f} AUC-ROC")
    print(f"  Recording:  {perch_row['recording_auc_roc_mean'] - vggish_row['recording_auc_roc_mean']:+.4f} AUC-ROC")
    print(f"  Window:     {perch_row['window_auc_roc'] - vggish_row['window_auc_roc']:+.4f} AUC-ROC")
  
  # Train-val gap
  train_val_gap = perch_row['train_auc'] - perch_row['window_auc_roc']
  print(f"\nTrain-val gap (window-level): {train_val_gap:.4f}")
  
  print('='*90)


if __name__ == '__main__':
  main()
