# coding=utf-8
"""Extract Perch 8 embeddings from the CirCor heart sound dataset.

This script loads the CirCor dataset at 32 kHz, extracts 5-second windows,
runs them through the Perch 8 model to get 1280-dim embeddings, and saves
the results locally.

Usage (3-class labels - RECOMMENDED for competition comparison):
  poetry run python -m pulse.inference.embed_heart_dataset \
    --use_3class \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./embeddings \
    --batch_size 32

Usage (binary labels - legacy):
  poetry run python -m pulse.inference.embed_heart_dataset \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./embeddings \
    --batch_size 32
"""

import argparse
import os

from chirp.preprocessing import pipeline
import numpy as np
from pulse.configs import heart_presets
from pulse.data import circor  # Import to register the dataset
from pulse.inference.perch_embedder import PerchEmbedder
from pulse.preprocessing import heart_ops
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def create_preprocessing_pipeline(sample_rate_hz: int, window_size_s: float,
                                   window_stride_s: float, split: str,
                                   train_fraction: float, use_3class: bool = False):
  """Create preprocessing pipeline for embedding extraction."""
  # Choose label mapping based on flag
  label_op = heart_ops.MapMurmurTo3Class() if use_3class else heart_ops.MapMurmurToBinary()
  
  ops = [
      heart_ops.PatientHashSplit(
          split=split,
          train_fraction=train_fraction,
      ),
      # Don't use OnlyJaxTypes() - it filters out patient_id/recording_id strings
      pipeline.RepeatPadding(
          pad_size=window_size_s,
          sample_rate=sample_rate_hz,
          add_mask=False,  # Mask not needed for inference
      ),
      label_op,  # Binary (Present vs Absent/Unknown) or 3-class (Present/Unknown/Absent)
      pipeline.ExtractStridedWindows(
          window_length_sec=window_size_s,
          window_stride_sec=window_stride_s,
          sample_rate=sample_rate_hz,
      ),
  ]
  return pipeline.Pipeline(ops=ops)


def extract_and_save_embeddings(
    dataset: tf.data.Dataset,
    embedder: PerchEmbedder,
    output_dir: str,
    split: str,
    batch_size: int = 32,
    use_3class: bool = False,
):
  """Extract embeddings and save to disk."""
  dataset = dataset.batch(batch_size, drop_remainder=False)
  
  all_embeddings = []
  all_labels = []
  all_recording_ids = []
  all_patient_ids = []
  all_segment_ids = []
  
  for batch in tqdm(dataset.as_numpy_iterator(), desc=f'{split}'):
    embeddings = embedder.embed_batch(batch['audio'])
    all_embeddings.append(embeddings)
    all_labels.append(batch['label'])
    all_recording_ids.extend([r.decode('utf-8') for r in batch['recording_id']])
    all_patient_ids.extend([p.decode('utf-8') for p in batch['patient_id']])
    all_segment_ids.append(batch['segment_id'])
  
  embeddings_array = np.concatenate(all_embeddings, axis=0)
  labels_array = np.concatenate(all_labels, axis=0)
  segment_ids_array = np.concatenate(all_segment_ids, axis=0)
  
  print(f'  {len(embeddings_array)} windows from {len(set(all_patient_ids))} patients')
  
  # Print label distribution based on mode
  if use_3class:
    unique_labels = np.unique(labels_array)
    print(f'  Label distribution (3-class):')
    for label_val in [0, 1, 2]:
      if label_val in unique_labels:
        count = (labels_array == label_val).sum()
        label_name = ['Absent', 'Present', 'Unknown'][label_val]
        print(f'    {label_name} ({label_val}): {count}/{len(labels_array)} ({100*count/len(labels_array):.1f}%)')
  else:
    print(f'  Positive: {labels_array.sum()}/{len(labels_array)} ({100*labels_array.sum()/len(labels_array):.1f}%)')
  
  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, f'embeddings_{split}.npz')
  np.savez_compressed(
      output_file,
      embeddings=embeddings_array,
      labels=labels_array,
      recording_ids=np.array(all_recording_ids),
      patient_ids=np.array(all_patient_ids),
      segment_ids=segment_ids_array,
  )
  
  print(f'  Saved: {output_file} ({os.path.getsize(output_file) / 1024**2:.1f} MB)')


def main():
  parser = argparse.ArgumentParser(description='Extract Perch 8 embeddings from CirCor')
  parser.add_argument('--tfds_data_dir', default='~/tensorflow_datasets')
  parser.add_argument('--output_dir', default='./embeddings')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--splits', nargs='+', default=['train', 'valid'])
  parser.add_argument('--use_3class', action='store_true',
                      help='Use 3-class labels (Present/Unknown/Absent) instead of binary')
  args = parser.parse_args()
  
  config = heart_presets.get_heart_base_config()
  print(f'Data: {args.tfds_data_dir}')
  print(f'Output: {args.output_dir}')
  print(f'Labels: {"3-class (Present/Unknown/Absent)" if args.use_3class else "Binary (Present vs Absent/Unknown)"}')
  
  # Initialize embedder
  embedder = PerchEmbedder(model_name=config.perch_model_name)
  
  # Process each split
  for split in args.splits:
    print(f'\nProcessing {split} split...')
    
    # Load dataset
    ds_builder = tfds.builder(config.dataset_directory, data_dir=args.tfds_data_dir)
    raw_dataset = ds_builder.as_dataset(split='train')
    
    # Apply preprocessing
    preprocessing_pipeline = create_preprocessing_pipeline(
        sample_rate_hz=config.sample_rate_hz,
        window_size_s=config.window_size_s,
        window_stride_s=config.window_stride_s,
        split=split,
        train_fraction=config.train_fraction,
        use_3class=args.use_3class,
    )
    processed_dataset = preprocessing_pipeline(raw_dataset, ds_builder.info).prefetch(tf.data.AUTOTUNE)
    
    # Extract embeddings
    extract_and_save_embeddings(
        dataset=processed_dataset,
        embedder=embedder,
        output_dir=args.output_dir,
        split=split,
        batch_size=args.batch_size,
        use_3class=args.use_3class,
    )
  
  print('\nDone!')


if __name__ == '__main__':
  main()

