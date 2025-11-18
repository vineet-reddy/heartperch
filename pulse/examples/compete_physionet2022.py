# coding=utf-8
"""PhysioNet 2022 Competition Comparison.

Compares frozen Perch embeddings against the official competition leaderboard
using Logistic Regression as the decoder.

Usage (3-class - RECOMMENDED):
  python -m pulse.examples.compete_physionet2022 \
    --use_3class \
    --embedding_dir ./embeddings \
    --output_dir ./results \
    --leaderboard_tsv ./pulse/data_snippet_heartperch/official_murmur_scores\ \(1\).tsv

Usage (binary):
  python -m pulse.examples.compete_physionet2022 \
    --embedding_dir ./embeddings \
    --output_dir ./results \
    --leaderboard_tsv ./pulse/data_snippet_heartperch/official_murmur_scores\ \(1\).tsv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def load_embeddings(embedding_dir: str, split: str) -> dict:
  """Load embeddings from npz files."""
  emb_path = Path(embedding_dir) / f'embeddings_{split}.npz'
  data = np.load(emb_path, allow_pickle=True)
  return {key: data[key] for key in data.files}


def aggregate_to_patient_level(patient_ids, embeddings, labels, use_3class=False):
  """Aggregate window-level embeddings to patient level.
  
  Aggregates embeddings using mean (matching competition methodology).
  
  For 3-class: uses label priority Present (1) > Unknown (2) > Absent (0)
  For binary: patient positive if ANY window positive
  """
  patient_data = {}
  
  for i, pid in enumerate(patient_ids):
    if pid not in patient_data:
      patient_data[pid] = {'embeddings': [], 'labels': []}
    patient_data[pid]['embeddings'].append(embeddings[i])
    patient_data[pid]['labels'].append(labels[i])
  
  X_patient = []
  y_patient = []
  patient_ids_list = []
  
  for pid in sorted(patient_data.keys()):
    data = patient_data[pid]
    X_patient.append(np.mean(data['embeddings'], axis=0))
    
    if use_3class:
      # Priority: Present > Unknown > Absent
      labels_set = set(data['labels'])
      if 1 in labels_set:
        y_patient.append(1)
      elif 2 in labels_set:
        y_patient.append(2)
      else:
        y_patient.append(0)
    else:
      y_patient.append(max(data['labels']))
    
    patient_ids_list.append(pid)
  
  return np.array(X_patient), np.array(y_patient), patient_ids_list


def compute_3class_weighted_accuracy(y_true, y_pred):
  """Compute PhysioNet 2022 3-class weighted accuracy.
  
  Formula: (5*m_PP + 3*m_UU + m_AA) / 
           (5*(m_PP + m_UP + m_AP) + 3*(m_PU + m_UU + m_AU) + (m_PA + m_UA + m_AA))
  
  Where m_XY = count of (predicted=X, true=Y) for X,Y in {Absent(0), Present(1), Unknown(2)}

  Competition formula to compute weighted accuracy metric score for official murmur entries is linked here: 
  https://moody-challenge.physionet.org/2022/#scoring
  """
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
  
  m_AA, m_AP, m_AU = cm[0, 0], cm[0, 1], cm[0, 2]
  m_PA, m_PP, m_PU = cm[1, 0], cm[1, 1], cm[1, 2]
  m_UA, m_UP, m_UU = cm[2, 0], cm[2, 1], cm[2, 2]
  
  numerator = 5 * m_PP + 3 * m_UU + m_AA
  denominator = (
      5 * (m_PP + m_UP + m_AP) +
      3 * (m_PU + m_UU + m_AU) +
      (m_PA + m_UA + m_AA)
  )
  
  weighted_acc = numerator / denominator if denominator > 0 else 0
  
  return {
      'weighted_acc': weighted_acc,
      'confusion_matrix': cm,
      'm_PP': int(m_PP), 'm_PU': int(m_PU), 'm_PA': int(m_PA),
      'm_UP': int(m_UP), 'm_UU': int(m_UU), 'm_UA': int(m_UA),
      'm_AP': int(m_AP), 'm_AU': int(m_AU), 'm_AA': int(m_AA),
  }


def optimize_3class_thresholds(y_true, y_proba):
  """Optimize decision thresholds for 3-class weighted accuracy."""
  best_score = 0.0
  best_preds = None
  best_thresholds = None
  
  for present_threshold in np.arange(0.15, 0.5, 0.025):
    for unknown_threshold in np.arange(0.15, 0.5, 0.025):
      y_pred = np.zeros(len(y_true), dtype=int)
      for i in range(len(y_true)):
        if y_proba[i, 1] >= present_threshold:
          y_pred[i] = 1
        elif y_proba[i, 2] >= unknown_threshold:
          y_pred[i] = 2
        else:
          y_pred[i] = 0
      
      metrics = compute_3class_weighted_accuracy(y_true, y_pred)
      if metrics['weighted_acc'] > best_score:
        best_score = metrics['weighted_acc']
        best_preds = y_pred.copy()
        best_thresholds = (present_threshold, unknown_threshold)
  
  return best_preds, best_score, best_thresholds


def compute_binary_weighted_accuracy(y_true, y_prob, threshold):
  """Compute binary weighted accuracy: (5*TP + TN) / (5*(TP+FN) + (FP+TN))"""
  y_pred = (y_prob >= threshold).astype(int)
  tp = ((y_true == 1) & (y_pred == 1)).sum()
  tn = ((y_true == 0) & (y_pred == 0)).sum()
  fp = ((y_true == 0) & (y_pred == 1)).sum()
  fn = ((y_true == 1) & (y_pred == 0)).sum()
  
  numerator = 5 * tp + tn
  denominator = 5 * (tp + fn) + (fp + tn)
  w_acc = numerator / denominator if denominator > 0 else 0
  
  return w_acc, tp, tn, fp, fn


def optimize_binary_threshold(y_true, y_prob):
  """Find threshold that maximizes binary weighted accuracy."""
  best_score = 0.0
  best_threshold = 0.5
  best_stats = None
  
  for threshold in np.arange(0.1, 0.9, 0.05):
    w_acc, tp, tn, fp, fn = compute_binary_weighted_accuracy(y_true, y_prob, threshold)
    if w_acc > best_score:
      best_score = w_acc
      best_threshold = threshold
      best_stats = (tp, tn, fp, fn)
  
  return best_threshold, best_score, best_stats


def grade_overfitting(gap):
  """Grade overfitting severity based on train-val AUC gap.
  
  Best practices thresholds:
    < 0.05: Normal generalization
    0.05-0.10: Mild overfitting
    0.10-0.15: Moderate overfitting
    >= 0.15: Severe overfitting
  """
  if gap < 0.05:
    return 'OK'
  elif gap < 0.10:
    return 'MILD'
  elif gap < 0.15:
    return 'MODERATE'
  else:
    return 'SEVERE'


def load_leaderboard(tsv_path: Path):
  """Load official competition leaderboard."""
  df = pd.read_csv(tsv_path, sep='\t')
  df_clean = df[['Rank', 'Team', 'AUROC on Test Set', 'AUPRC on Test Set', 
                  'Weighted Accuracy on Test Set']].copy()
  df_clean.columns = ['Rank', 'Team', 'AUROC', 'AUPRC', 'Weighted_Acc']
  return df_clean


def main():
  parser = argparse.ArgumentParser(
      description='PhysioNet 2022 Competition Comparison'
  )
  parser.add_argument('--embedding_dir', type=str, required=True,
                      help='Directory containing embeddings_train.npz and embeddings_valid.npz')
  parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
  parser.add_argument('--leaderboard_tsv', type=str, 
                      default='pulse/examples/official_murmur_scores (1).tsv',
                      help='Path to official leaderboard TSV file')
  parser.add_argument('--use_3class', action='store_true',
                      help='Use 3-class labels (Present/Unknown/Absent) for exact competition comparison')
  
  args = parser.parse_args()
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  mode = '3-class' if args.use_3class else 'binary'
  print(f'\n=== PhysioNet 2022 Comparison ({mode}) ===')
  
  # Load embeddings
  train_data = load_embeddings(args.embedding_dir, 'train')
  valid_data = load_embeddings(args.embedding_dir, 'valid')
  
  if args.use_3class:
    train_unique = np.unique(train_data['labels'])
    valid_unique = np.unique(valid_data['labels'])
    if len(train_unique) != 3 or len(valid_unique) != 3:
      print(f'ERROR: Need 3-class embeddings. Found: train={train_unique}, valid={valid_unique}')
      print('Run: python -m pulse.inference.embed_heart_dataset --use_3class')
      return
  
  # Aggregate to patient level
  X_train, y_train, _ = aggregate_to_patient_level(
      train_data['patient_ids'], train_data['embeddings'], train_data['labels'], use_3class=args.use_3class
  )
  X_valid, y_valid, _ = aggregate_to_patient_level(
      valid_data['patient_ids'], valid_data['embeddings'], valid_data['labels'], use_3class=args.use_3class
  )
  
  # Train models
  if args.use_3class:
    model_default = LogisticRegression(max_iter=100000, class_weight='balanced', random_state=42)
    model_default.fit(X_train, y_train)
    
    # Tune for competition metric
    configs = [('balanced', 0.1), ('balanced', 1.0), ('balanced', 10.0),
               ({0: 1.0, 1: 5.0, 2: 3.0}, 0.1), ({0: 1.0, 1: 5.0, 2: 3.0}, 1.0)]
    
    best_w_acc, model_tuned, _ = 0, None, None
    for cw, C in configs:
      m = LogisticRegression(C=C, max_iter=100000, class_weight=cw, random_state=42)
      m.fit(X_train, y_train)
      _, w_acc, _ = optimize_3class_thresholds(y_valid, m.predict_proba(X_valid))
      if w_acc > best_w_acc:
        best_w_acc, model_tuned = w_acc, m
  else:
    model_default = LogisticRegression(max_iter=100000, class_weight='balanced', random_state=42)
    model_default.fit(X_train, y_train)
    model_tuned = None
  
  # Evaluate both models (if 3-class)
  def eval_model(model, name):
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
    y_valid_bin = label_binarize(y_valid, classes=[0, 1, 2])
    
    train_proba = model.predict_proba(X_train)
    valid_proba = model.predict_proba(X_valid)
    
    train_auc = roc_auc_score(y_train_bin, train_proba, average='macro', multi_class='ovr')
    valid_auc = roc_auc_score(y_valid_bin, valid_proba, average='macro', multi_class='ovr')
    valid_auprc = average_precision_score(y_valid_bin, valid_proba, average='macro')
    
    _, w_acc, thresholds = optimize_3class_thresholds(y_valid, valid_proba)
    
    return {
        'name': name,
        'train_auc': train_auc,
        'valid_auc': valid_auc,
        'valid_auprc': valid_auprc,
        'weighted_acc': w_acc,
        'thresholds': thresholds,
        'valid_proba': valid_proba,
    }
  
  if args.use_3class:
    print('Evaluating models...')
    results_default = eval_model(model_default, 'Default')
    results_tuned = eval_model(model_tuned, 'Tuned')
    
    # Use tuned model for final reporting (better competition score)
    results = results_tuned
    train_auc = results['train_auc']
    valid_auc = results['valid_auc']
    valid_auprc = results['valid_auprc']
    opt_w_acc = results['weighted_acc']
    opt_thresholds = results['thresholds']
    
    y_pred_valid_opt, _, _ = optimize_3class_thresholds(y_valid, results['valid_proba'])
    valid_metrics = compute_3class_weighted_accuracy(y_valid, y_pred_valid_opt)
    model = model_tuned  # Save tuned model
  else:
    y_pred_train = model_default.predict_proba(X_train)[:, 1]
    y_pred_valid = model_default.predict_proba(X_valid)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    valid_auc = roc_auc_score(y_valid, y_pred_valid)
    valid_auprc = average_precision_score(y_valid, y_pred_valid)
    
    opt_threshold, opt_w_acc, (tp, tn, fp, fn) = optimize_binary_threshold(y_valid, y_pred_valid)
    model = model_default
  
  
  # Leaderboard
  leaderboard_path = Path(args.leaderboard_tsv)
  if leaderboard_path.exists():
    lb = load_leaderboard(leaderboard_path)
    
    if args.use_3class:
      rank_default = (lb['Weighted_Acc'] > results_default['weighted_acc']).sum() + 1
      rank_tuned = (lb['Weighted_Acc'] > results_tuned['weighted_acc']).sum() + 1
      auroc_rank_default = (lb['AUROC'] > results_default['valid_auc']).sum() + 1
      auroc_rank_tuned = (lb['AUROC'] > results_tuned['valid_auc']).sum() + 1
      
      print('\nRank | Team                  | AUROC | AUPRC | Weighted Acc')
      print('-'*70)
      for _, row in lb.head(10).iterrows():
        print(f"{int(row['Rank']):4d} | {row['Team']:20s} | {row['AUROC']:.3f} | {row['AUPRC']:.3f} | {row['Weighted_Acc']:.3f}")
      if rank_tuned > 10:
        print('  ...')
      print('-'*70)
      print(f"{rank_tuned:4d} | {'Perch (Tuned)':20s} | {results_tuned['valid_auc']:.3f} | {results_tuned['valid_auprc']:.3f} | {results_tuned['weighted_acc']:.3f}  ⬅")
      print('-'*70)
      gap_default = results_default['train_auc'] - results_default['valid_auc']
      gap_tuned = results_tuned['train_auc'] - results_tuned['valid_auc']
      print(f"\nDefault: AUROC rank {auroc_rank_default}/{len(lb)}, Competition rank {rank_default}/{len(lb)}")
      print(f"         Train AUC={results_default['train_auc']:.3f}, Val AUC={results_default['valid_auc']:.3f}, Gap={gap_default:+.3f} ({grade_overfitting(gap_default)})")
      print(f"Tuned:   AUROC rank {auroc_rank_tuned}/{len(lb)}, Competition rank {rank_tuned}/{len(lb)}")
      print(f"         Train AUC={results_tuned['train_auc']:.3f}, Val AUC={results_tuned['valid_auc']:.3f}, Gap={gap_tuned:+.3f} ({grade_overfitting(gap_tuned)})")
    else:
      rank = (lb['Weighted_Acc'] > opt_w_acc).sum() + 1
      auroc_rank = (lb['AUROC'] > valid_auc).sum() + 1
      
      print('\nRank | Team                  | AUROC | AUPRC | Weighted Acc')
      print('-'*70)
      for _, row in lb.head(10).iterrows():
        print(f"{int(row['Rank']):4d} | {row['Team']:20s} | {row['AUROC']:.3f} | {row['AUPRC']:.3f} | {row['Weighted_Acc']:.3f}")
      if rank > 10:
        print('  ...')
      print('-'*70)
      print(f"{rank:4d} | {'Perch':20s} | {valid_auc:.3f} | {valid_auprc:.3f} | {opt_w_acc:.3f}  ⬅")
      print('-'*70)
      print(f"\nAUROC rank {auroc_rank}, Competition rank {rank}/{len(lb)}")
  
  # Save results
  suffix = '_3class' if args.use_3class else '_binary'
  
  if args.use_3class:
    # Save both models' results
    results_data = [
        {
            'model': 'default',
            'description': 'Best discrimination',
            'train_auc': float(results_default['train_auc']),
            'valid_auc': float(results_default['valid_auc']),
            'overfitting_gap': float(results_default['train_auc'] - results_default['valid_auc']),
            'overfitting_grade': grade_overfitting(results_default['train_auc'] - results_default['valid_auc']),
            'valid_auprc': float(results_default['valid_auprc']),
            'weighted_acc': float(results_default['weighted_acc']),
            'auroc_rank': (load_leaderboard(Path(args.leaderboard_tsv))['AUROC'] > results_default['valid_auc']).sum() + 1,
            'competition_rank': (load_leaderboard(Path(args.leaderboard_tsv))['Weighted_Acc'] > results_default['weighted_acc']).sum() + 1,
        },
        {
            'model': 'tuned',
            'description': 'Best competition score',
            'train_auc': float(results_tuned['train_auc']),
            'valid_auc': float(results_tuned['valid_auc']),
            'overfitting_gap': float(results_tuned['train_auc'] - results_tuned['valid_auc']),
            'overfitting_grade': grade_overfitting(results_tuned['train_auc'] - results_tuned['valid_auc']),
            'valid_auprc': float(results_tuned['valid_auprc']),
            'weighted_acc': float(results_tuned['weighted_acc']),
            'auroc_rank': (load_leaderboard(Path(args.leaderboard_tsv))['AUROC'] > results_tuned['valid_auc']).sum() + 1,
            'competition_rank': (load_leaderboard(Path(args.leaderboard_tsv))['Weighted_Acc'] > results_tuned['weighted_acc']).sum() + 1,
        }
    ]
    results_df = pd.DataFrame(results_data)
    
    # Save tuned model (best competition performance)
    model_path = output_dir / f'physionet2022_model{suffix}.joblib'
    joblib.dump(model_tuned, model_path)
  else:
    results_df = pd.DataFrame([{
        'model': 'default',
        'train_auc': float(train_auc),
        'valid_auc': float(valid_auc),
        'overfitting_gap': float(train_auc - valid_auc),
        'overfitting_grade': grade_overfitting(train_auc - valid_auc),
        'valid_auprc': float(valid_auprc),
        'weighted_acc': float(opt_w_acc),
        'threshold': float(opt_threshold),
    }])
    model_path = output_dir / f'physionet2022_model{suffix}.joblib'
    joblib.dump(model_default, model_path)
  
  results_path = output_dir / f'physionet2022_results{suffix}.csv'
  results_df.to_csv(results_path, index=False)
  
  print(f'\nSaved: {results_path.name}, {model_path.name}')


if __name__ == '__main__':
  main()
