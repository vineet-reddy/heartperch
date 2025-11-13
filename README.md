# HeartPerch

**Testing whether bird vocalization models transfer to medical audio.**

This is a fork of [Google's Perch](https://github.com/google-research/perch) that evaluates frozen bird vocalization embeddings on heart murmur detection from the [George B. Moody PhysioNet Challenge 2022](https://moody-challenge.physionet.org/2022/).

**Result:** Without training on any heart data, Perch ranked **6th out of 40 teams** in the competition, demonstrating strong zero-shot transfer learning to data-scarce medical domains. ([full results](pulse/docs/results/results_summary.pdf))

All heart-specific code is in [`pulse/`](pulse/). The original [`chirp/`](chirp/) codebase remains largely untouched.

---

## Why HeartPerch?

Perch's architecture (trained to disentangle overlapping bird vocalizations in noisy environments) has theoretically learned general representations of structured time-frequency patterns. These representations transfer surprisingly well to cardiac signals, capturing subtle rhythm irregularities and spectrotemporal patterns which general audio models often miss.

HeartPerch is a minimal wrapper around Perch that reuses its preprocessing pipeline and adds heart-specific functionality for murmur detection.

---

## Project Structure

```
pulse/
├── configs/          # Configuration presets (32kHz, 5s windows, 2.5s stride)
├── data/             # TFDS builder for CirCor dataset (v2.0.0)
├── preprocessing/    # Patient-level splits and label mapping
├── inference/        # Perch 8 embedding extraction (1280-dim)
├── train/            # Linear probe training (sklearn LogisticRegression)
├── examples/         # Evaluation and baseline comparison scripts
├── scripts/          # Dataset building and validation utilities
└── results/          # Evaluation metrics and predictions
```

---

## Pipeline Overview

The pipeline uses **Perch 8** (Google's bird vocalization model) as a frozen feature extractor:

1. **Build dataset**: CirCor PhysioNet 2022 → TFDS format (v2.0.0)
2. **Extract embeddings**: Audio @ 32kHz → 5s windows (2.5s stride) → Perch 8 → 1280-dim embeddings
3. **Train classifier**: Embeddings → sklearn LogisticRegression (binary or 3-class)
4. **Evaluate**: Window → Recording → Patient-level aggregation and metrics

---

## Quick Start

### Full Pipeline (Automated)

Run the complete pipeline with a single command:

```bash
./pulse/run_full_pipeline.sh
```

This script:
1. Builds the CirCor TFDS dataset (v2.0.0)
2. Extracts embeddings for binary and 3-class classification
3. Validates embeddings (checks for data leakage)
4. Trains linear probes
5. Evaluates against baselines and competition metrics
6. Backs up results to GCS

### Manual Steps

#### 1. Install Dependencies

```bash
cd ~/heartperch
poetry install --with jaxtrain
```

#### 2. Build Dataset

```bash
poetry run python -m pulse.scripts.build_circor_dataset
```

Creates TFDS dataset at `~/tensorflow_datasets/circor/2.0.0/`

#### 3. Extract Embeddings

```bash
poetry run python -m pulse.inference.embed_heart_dataset \
  --tfds_data_dir ~/tensorflow_datasets \
  --output_dir ./embeddings \
  --batch_size 32
```

#### 4. Train Linear Probe

```bash
poetry run python -m pulse.train.linear_probe \
  --embedding_dir ./embeddings \
  --output_dir ./models/linear_probe
```

#### 5. Evaluate

```bash
poetry run python -m pulse.examples.evaluate_linear_probe \
  --embedding_dir ./embeddings \
  --model_path ./models/linear_probe/model.joblib \
  --output_dir ./results
```

---

## Key Components

- **[`heart_presets.py`](pulse/configs/heart_presets.py)**: Config with Perch 8 requirements (32kHz, 5s windows, 2.5s stride, 1280-dim embeddings)
- **[`circor.py`](pulse/data/circor.py)**: TFDS builder for CirCor dataset (v2.0.0) with recording-level labels
- **[`heart_ops.py`](pulse/preprocessing/heart_ops.py)**: Binary label mapping and patient-level train/valid splits
- **[`perch_embedder.py`](pulse/inference/perch_embedder.py)**: Wrapper around Perch 8 for batch embedding extraction
- **[`compete_physionet2022.py`](pulse/examples/compete_physionet2022.py)**: 3-class evaluation using official competition metrics

---

## Design Decisions

- **Frozen embeddings**: Pre-compute embeddings once, iterate on classifiers without expensive inference
- **Patient-level splits**: Hash-based deterministic split prevents data leakage
- **sklearn classifiers**: Simple logistic regression (no TPU/GPU needed for training)
- **50% window overlap**: 2.5s stride captures more context for short cardiac events
- **Multi-level evaluation**: Window/recording/patient metrics for different aggregation strategies

---

## Results

📄 **[View Results Summary (PDF)](pulse/docs/results/results_summary.pdf)**

See [`pulse/docs/results/results_summary.md`](pulse/docs/results/results_summary.md) for full evaluation results.

**Highlights:**
- **Binary classification**: 86.3% AUROC (recording-level)
- **vs. Baselines**: +4.5% over VGGish, +9.4% over MFCC
- **Competition ranking**: 6th/40 teams (3-class, frozen embeddings only)

---

## Updating Dataset Version

To update to a new dataset version:

1. Update `pulse/data/circor.py` line 23: `VERSION = tfds.core.Version('X.X.X')`
2. Update `pulse/configs/heart_presets.py` line 30: `'circor/full_length:X.X.X'`
3. Rebuild dataset: `poetry run python -m pulse.scripts.build_circor_dataset`

All scripts automatically use the version specified in `heart_presets.py`.

---

## Generate Results Summary

```bash
# Generate markdown from CSV results
python3 pulse/scripts/generate_results_summary.py

# Convert to PDF
pandoc pulse/docs/results/results_summary.md \
  -o pulse/docs/results/results_summary.pdf \
  -V geometry:"margin=0.5in,top=0.6in,bottom=0.6in" \
  -V fontsize=10pt -V papersize=letter -V linestretch=0.9 \
  -V pagestyle=empty --pdf-engine=pdflatex

# Convert to LaTeX
pandoc pulse/docs/results/results_summary.md \
  -s -o pulse/docs/results/results_summary.tex \
  -V geometry:"margin=0.5in,top=0.6in,bottom=0.6in" \
  -V fontsize=10pt -V papersize=letter -V linestretch=0.9 \
  -V pagestyle=empty
```

---

## Troubleshooting

### Poetry not found (in tmux/new shells)

```bash
source ~/.bashrc
```

### Reset all generated data

```bash
rm -rf ~/tensorflow_datasets/circor ./embeddings* ./models ./results
```

