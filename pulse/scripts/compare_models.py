# coding=utf-8
"""Compare results from different Perch models.

After running run_full_pipeline.sh with different models, use this to compare.

Usage:
  poetry run python -m pulse.scripts.compare_models --results_dir ./results_all_models
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
  parser = argparse.ArgumentParser(description='Compare Perch model results')
  parser.add_argument('--results_dir', default='./results_all_models',
                      help='Directory containing results for each model')
  args = parser.parse_args()
  
  results_dir = Path(args.results_dir)
  
  if not results_dir.exists():
    print(f"No results found at {results_dir}")
    print("Run the pipeline first:")
    print("  ./run_full_pipeline.sh")
    print("  MODEL_NAME=perch_v2 ./run_full_pipeline.sh")
    print("  MODEL_NAME=surfperch ./run_full_pipeline.sh")
    return
  
  # Load metrics for each model
  all_metrics = []
  for model_dir in sorted(results_dir.iterdir()):
    if not model_dir.is_dir():
      continue
    
    metrics_file = model_dir / 'metrics.csv'
    if not metrics_file.exists():
      continue
    
    df = pd.read_csv(metrics_file)
    df['model'] = model_dir.name
    all_metrics.append(df)
  
  if not all_metrics:
    print(f"No metrics.csv files found in {results_dir}")
    return
  
  # Combine all results
  combined = pd.concat(all_metrics, ignore_index=True)
  
  # Pivot to get models as rows
  pivot = combined.pivot(index='model', columns='level', values='auc_roc')
  
  print("="*80)
  print("PERCH MODEL COMPARISON - AUC-ROC")
  print("="*80)
  print(pivot.to_string())
  print("")
  
  # Print patient-level comparison (most important)
  patient_metrics = combined[combined['level'] == 'patient'][['model', 'auc_roc', 'auc_pr']].sort_values('auc_roc', ascending=False)
  print("Patient-Level Performance (Clinical Decision):")
  print(patient_metrics.to_string(index=False))
  print("")
  
  # Show baseline comparison for best model
  best_model = patient_metrics.iloc[0]['model']
  print(f"Baseline comparison for best model ({best_model}):")
  baseline_file = results_dir / best_model / 'baseline_comparison.csv'
  if baseline_file.exists():
    baseline_df = pd.read_csv(baseline_file)
    print(baseline_df[['baseline', 'patient_auc_roc', 'recording_auc_roc_mean']].to_string(index=False))
  
  print("="*80)
  
  # Save combined results
  output_file = results_dir / 'comparison.csv'
  combined.to_csv(output_file, index=False)
  print(f"\nFull comparison saved to: {output_file}")


if __name__ == '__main__':
  main()

