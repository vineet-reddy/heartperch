"""Diagnostic script to analyze CirCor audio file parameters.

Scans audio files and reports their statistics to understand the data
before fixing encoding issues.

Usage:
  poetry run python -m pulse.scripts.diagnose_circor_audio
"""

import os
from chirp import audio_utils
from etils import epath
import pandas as pd
import numpy as np
from collections import defaultdict


def analyze_audio_file(wav_path, target_sample_rate=32000, resampling_method='polyphase'):
  """Load and analyze a single audio file."""
  try:
    audio = audio_utils.load_audio_file(
        str(wav_path),
        target_sample_rate=target_sample_rate,
        resampling_type=resampling_method,
    )
    
    if audio.shape[0] == 0:
      return None, "Empty audio"
    
    stats = {
        'min': float(audio.min()),
        'max': float(audio.max()),
        'mean': float(audio.mean()),
        'std': float(audio.std()),
        'shape': audio.shape,
        'dtype': str(audio.dtype),
        'abs_max': float(np.abs(audio).max()),
        'in_range': (audio.min() >= -1.0) and (audio.max() < 1.0),
    }
    return stats, None
  except Exception as e:
    return None, str(e)


def main():
  """Analyze all CirCor audio files."""
  data_path = epath.Path('gs://seizurepredict-ds005873/data')
  metadata = pd.read_csv(data_path / 'training_data.csv')
  audio_dir = data_path / 'training_data'
  
  print("=" * 80)
  print("CirCor Audio Files Diagnostic Report")
  print("=" * 80)
  
  target_sample_rate = 32000
  resampling_method = 'polyphase'
  
  print(f"Configuration:")
  print(f"Target sample rate: {target_sample_rate} Hz")
  print(f"Resampling method: {resampling_method}")
  
  all_stats = []
  errors = []
  out_of_range_files = []
  total_files = 0
  skipped_files = 0
  
  print("Scanning files...")
  print("-" * 80)
  
  for idx, row in metadata.iterrows():
    patient_id = str(row['Patient ID'])
    locations_str = str(row['Recording locations:'])
    murmur = str(row['Murmur'])
    outcome = str(row['Outcome'])
    
    if pd.isna(locations_str) or locations_str == 'nan':
      continue
    if murmur not in ['Present', 'Absent', 'Unknown']:
      continue
    if outcome not in ['Normal', 'Abnormal']:
      continue
    
    for location in locations_str.split('+'):
      location = location.strip()
      if location not in ['AV', 'MV', 'PV', 'TV']:
        continue
      
      wav_file = audio_dir / f"{patient_id}_{location}.wav"
      if not wav_file.exists():
        skipped_files += 1
        continue
      
      total_files += 1
      recording_id = f"{patient_id}_{location}"
      
      stats, error = analyze_audio_file(wav_file, target_sample_rate, resampling_method)
      
      if error:
        errors.append({'recording_id': recording_id, 'error': error})
        print(f"[ERROR] {recording_id}: {error}")
        continue
      
      if stats:
        stats['recording_id'] = recording_id
        stats['murmur'] = murmur
        stats['outcome'] = outcome
        all_stats.append(stats)
        
        if not stats['in_range']:
          out_of_range_files.append(stats)
          print(f"[OUT OF RANGE] {recording_id}: min={stats['min']:.6f}, max={stats['max']:.6f}")
  
  print("-" * 80)
  print()
  
  # Summary statistics
  print("=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print()
  print(f"Total files scanned: {total_files}")
  print(f"Successfully analyzed: {len(all_stats)}")
  print(f"Skipped (not found): {skipped_files}")
  print(f"Errors: {len(errors)}")
  print(f"Files with values outside [-1, 1): {len(out_of_range_files)}")
  print()
  
  if all_stats:
    # Aggregate statistics
    all_mins = [s['min'] for s in all_stats]
    all_maxs = [s['max'] for s in all_stats]
    all_abs_maxs = [s['abs_max'] for s in all_stats]
    
    print("AGGREGATE STATISTICS ACROSS ALL FILES:")
    print("-" * 80)
    print(f"Minimum value across all files: {min(all_mins):.8f}")
    print(f"Maximum value across all files: {max(all_maxs):.8f}")
    print(f"Largest absolute value: {max(all_abs_maxs):.8f}")
    print()
    print(f"Mean of minimums: {np.mean(all_mins):.8f} ± {np.std(all_mins):.8f}")
    print(f"Mean of maximums: {np.mean(all_maxs):.8f} ± {np.std(all_maxs):.8f}")
    print()
    
    # Distribution of min/max values
    print("DISTRIBUTION OF MIN VALUES:")
    print(f"< -1.0: {sum(1 for m in all_mins if m < -1.0)}")
    print(f"[-1.0, -0.5): {sum(1 for m in all_mins if -1.0 <= m < -0.5)}")
    print(f"[-0.5, -0.1): {sum(1 for m in all_mins if -0.5 <= m < -0.1)}")
    print(f"[-0.1, 0): {sum(1 for m in all_mins if -0.1 <= m < 0)}")
    print()
    
    print("DISTRIBUTION OF MAX VALUES:")
    print(f">= 1.0: {sum(1 for m in all_maxs if m >= 1.0)}")
    print(f"[0.5, 1.0): {sum(1 for m in all_maxs if 0.5 <= m < 1.0)}")
    print(f"[0.1, 0.5): {sum(1 for m in all_maxs if 0.1 <= m < 0.5)}")
    print(f"[0, 0.1): {sum(1 for m in all_maxs if 0 <= m < 0.1)}")
    print()
  
  # Show details of out-of-range files
  if out_of_range_files:
    print("=" * 80)
    print(f"DETAILS OF FILES WITH VALUES OUTSIDE [-1, 1) ({len(out_of_range_files)} files)")
    print("=" * 80)
    print()
    
    for i, stats in enumerate(out_of_range_files[:20], 1):  # Show first 20
      print(f"{i}. {stats['recording_id']}")
      print(f"min: {stats['min']:.8f}, max: {stats['max']:.8f}")
      print(f"mean: {stats['mean']:.8f}, std: {stats['std']:.8f}")
      print(f"shape: {stats['shape']}, dtype: {stats['dtype']}")
      print(f"murmur: {stats['murmur']}, outcome: {stats['outcome']}")
      print()
    
    if len(out_of_range_files) > 20:
      print(f"...and {len(out_of_range_files) - 20} more files")
      print()
  
  # Show errors if any
  if errors:
    print("=" * 80)
    print(f"ERRORS ({len(errors)} files)")
    print("=" * 80)
    print()
    for error_info in errors[:10]:  # Show first 10
      print(f"{error_info['recording_id']}: {error_info['error']}")
    if len(errors) > 10:
      print(f"...and {len(errors) - 10} more errors")
    print()
  
  print("=" * 80)
  print("END OF REPORT")
  print("=" * 80)


if __name__ == '__main__':
  main()

