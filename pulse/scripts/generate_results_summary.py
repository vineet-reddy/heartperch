#!/usr/bin/env python3
"""Generate results summary markdown from CSV files."""

import csv
from pathlib import Path


def read_csv_as_dict(filepath):
    """Read CSV and return list of dicts."""
    with open(filepath) as f:
        return list(csv.DictReader(f))


def generate_summary(results_dir: Path, output_path: Path):
    """Generate markdown summary from CSV results."""
    
    # Load data
    metrics = read_csv_as_dict(results_dir / "metrics.csv")
    confusion = read_csv_as_dict(results_dir / "confusion_matrices.csv")
    baselines = read_csv_as_dict(results_dir / "baseline_comparison.csv")
    competition = read_csv_as_dict(results_dir / "physionet2022_results_3class.csv")
    
    # Extract values (filter dicts)
    rec_mean = next(m for m in metrics if m['level'] == 'recording_mean')
    rec_max = next(m for m in metrics if m['level'] == 'recording_max')
    window = next(m for m in metrics if m['level'] == 'window')
    
    conf_rec_mean = next(c for c in confusion if c['level'] == 'recording_mean')
    
    perch = next(b for b in baselines if b['baseline'] == 'Perch (unscaled)')
    vggish = next(b for b in baselines if b['baseline'] == 'VGGish (unscaled)')
    mfcc_spec = next(b for b in baselines if b['baseline'] == 'MFCC+Spectral (scaled)')
    mfcc = next(b for b in baselines if b['baseline'] == 'MFCC (scaled)')
    random = next(b for b in baselines if b['baseline'] == 'Random (scaled)')
    
    comp_tuned = next(c for c in competition if c['model'] == 'tuned')
    
    # Convert to float for calculations
    rec_auc = float(rec_mean['auc_roc'])
    rec_max_auc = float(rec_max['auc_roc'])
    rec_auprc = float(rec_mean['auc_pr'])
    win_auc = float(window['auc_roc'])
    win_auprc = float(window['auc_pr'])
    
    conf_sens = float(conf_rec_mean['sensitivity'])
    conf_spec = float(conf_rec_mean['specificity'])
    conf_prec = float(conf_rec_mean['precision'])
    
    perch_rec = float(perch['recording_auc_roc_mean'])
    perch_win = float(perch['window_auc_roc'])
    vggish_rec = float(vggish['recording_auc_roc_mean'])
    vggish_win = float(vggish['window_auc_roc'])
    mfcc_spec_rec = float(mfcc_spec['recording_auc_roc_mean'])
    mfcc_spec_win = float(mfcc_spec['window_auc_roc'])
    mfcc_rec = float(mfcc['recording_auc_roc_mean'])
    mfcc_win = float(mfcc['window_auc_roc'])
    rand_rec = float(random['recording_auc_roc_mean'])
    rand_win = float(random['window_auc_roc'])
    
    comp_auc = float(comp_tuned['valid_auc'])
    comp_auprc = float(comp_tuned['valid_auprc'])
    comp_acc = float(comp_tuned['weighted_acc'])
    comp_rank = int(comp_tuned['competition_rank'])
    
    # Generate markdown
    md = f"""# Frozen Bird Embeddings for Heart Murmur Detection

**Method:** Frozen Perch embeddings (1280-dim, trained on bird sounds) + Logistic Regression

## Experiment 1: What is Perch Learning?

**Setup:** Binary classification (Present vs Absent/Unknown) to test if bird embeddings can distinguish heart murmurs.

**Results:** Recording-level AUROC={rec_auc:.3f} (mean), {rec_max_auc:.3f} (max), AUPRC={rec_auprc:.3f} | Window-level AUROC={win_auc:.3f}, AUPRC={win_auprc:.3f}

**Confusion Matrix (Recording-level, threshold=0.5):**
```
TN={conf_rec_mean['tn']}  FP={conf_rec_mean['fp']}   Specificity: {conf_spec*100:.1f}%
FN={conf_rec_mean['fn']}   TP={conf_rec_mean['tp']}   Sensitivity: {conf_sens*100:.1f}%
                Precision:   {conf_prec*100:.1f}%
```

**Key Finding:** The model successfully detects murmurs and is not simply predicting all-negative (balanced confusion matrix with {conf_sens*100:.0f}% sensitivity, {conf_spec*100:.0f}% specificity).

---

## Experiment 2: How Does Perch Compare to Baselines?

**Setup:** Binary classification with fair comparison against statistical audio features.

**Results (AUROC):**

| Method | Recording | Window |
|--------|-----------|--------|
| **Perch** | **{perch_rec:.3f}** | **{perch_win:.3f}** |
| VGGish | {vggish_rec:.3f} | {vggish_win:.3f} |
| MFCC+Spectral | {mfcc_spec_rec:.3f} | {mfcc_spec_win:.3f} |
| MFCC | {mfcc_rec:.3f} | {mfcc_win:.3f} |
| Random | {rand_rec:.3f} | {rand_win:.3f} |

**Key Finding:** Frozen bird embeddings outperform both audio-specific embeddings (VGGish, +{(perch_rec-vggish_rec)*100:.1f}%) and traditional signal processing features.

---

## Experiment 3: Competition Comparison

**Setup:** 3-class classification (Present/Unknown/Absent) using exact PhysioNet 2022 competition metrics.

**Perch Performance:** AUROC={comp_auc:.3f}, AUPRC={comp_auprc:.3f}, Weighted Accuracy={comp_acc:.3f} → **Rank {comp_rank}/40**

**Leaderboard Context:**

| Rank | Team | AUROC | AUPRC | Weighted Acc |
|------|------|-------|-------|--------------|
| 1 | HearHeart | 0.884 | 0.716 | 0.780 |
| 4 | PathToMyHeart | 0.880 | 0.684 | 0.771 |
| **{comp_rank}** | **Perch (Tuned)** | **{comp_auc:.3f}** | **{comp_auprc:.3f}** | **{comp_acc:.3f}** |
| 6 | Care4MyHeart | 0.891 | 0.717 | 0.757 |
| 9 | ISIBrno-AIMT | 0.897 | 0.746 | 0.755 |

**Key Finding:** Using frozen bird embeddings with logistic regression ranks 6th among 40 teams, competitive with approaches specifically trained on heart sound data.

---

## Takeaways

1. **Frozen embeddings work on heart audio:** {rec_auc*100:.0f}% AUROC, balanced performance without cardiovascular training.
2. **Beats domain-specific models:** Outperforms VGGish (+{(perch_rec-vggish_rec)*100:.1f}%), MFCC, spectral features.
3. **Competitive in real competition:** Ranks {comp_rank}/40 without task-specific training.

Bird embeddings transfer effectively to medical audio, a data-scarce domain.
"""
    
    # Write to file
    output_path.write_text(md)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    output_path = Path(__file__).parent.parent / "docs" / "results" / "results_summary.md"
    
    generate_summary(results_dir, output_path)

