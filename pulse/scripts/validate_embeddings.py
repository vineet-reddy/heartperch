"""Validate embedding splits for data leakage.

Usage:
  python -m pulse.scripts.validate_embeddings --embedding_dir ./embeddings
"""

import argparse
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding_dir', default='./embeddings')
  args = parser.parse_args()
  
  embedding_path = Path(args.embedding_dir)
  train = np.load(embedding_path / 'embeddings_train.npz', allow_pickle=True)
  valid = np.load(embedding_path / 'embeddings_valid.npz', allow_pickle=True)
  
  train_recordings = set(train['recording_ids'])
  valid_recordings = set(valid['recording_ids'])
  all_recordings = train_recordings | valid_recordings
  
  train_patients = set(train['patient_ids'])
  valid_patients = set(valid['patient_ids'])
  all_patients = train_patients | valid_patients
  
  print('=== Dataset Statistics ===')
  print(f'Train:')
  print(f'  Windows: {len(train["embeddings"])}')
  print(f'  Unique recordings: {len(train_recordings)}')
  print(f'  Unique patients: {len(train_patients)}')
  print(f'  Positive: {train["labels"].sum()}/{len(train["labels"])} ({100*train["labels"].sum()/len(train["labels"]):.1f}%)')
  print()
  print(f'Valid:')
  print(f'  Windows: {len(valid["embeddings"])}')
  print(f'  Unique recordings: {len(valid_recordings)}')
  print(f'  Unique patients: {len(valid_patients)}')
  print(f'  Positive: {valid["labels"].sum()}/{len(valid["labels"])} ({100*valid["labels"].sum()/len(valid["labels"]):.1f}%)')
  print()
  print(f'Total:')
  print(f'  Windows: {len(train["embeddings"]) + len(valid["embeddings"])}')
  print(f'  Unique recordings: {len(all_recordings)}')
  print(f'  Unique patients: {len(all_patients)}')
  print()
  print(f'=== CRITICAL CHECKS ===')
  print(f'  Patient overlap: {len(train_patients & valid_patients)} (MUST be 0)')
  print(f'  Recording overlap: {len(train_recordings & valid_recordings)} (MUST be 0)')


if __name__ == '__main__':
  main()

