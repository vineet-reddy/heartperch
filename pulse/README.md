## Reason for Creating HeartPerch

Perch's architecture, trained to disentangle overlapping bird vocalizations in noisy environments, has learned a general representation of structured time-frequency patterns. By transferring it to cardiac signals, we can utilize its ability to detect subtle rhythm irregularities, morphological drift, and hidden spectrotemporal motifs that traditional ECG models or human review often miss. As such, Perch could perhaps uncover latent biomarkers of heart instability directly from the signal itself, without handcrafted features or heavy human supervision.

Here, the idea was to create a light wrapper around [`chirp`](../chirp) where we could reuse as many chirp components as possible to create [`pulse`](.), which holds all heart-specific perch modifications to create heartperch.

## Structure

```
pulse/
├── configs/          # Model configs for heart sound tasks (linear probe, fine-tuning)
├── data/             # Dataset builders (CirCor dataset)
├── preprocessing/    # Heart-specific audio preprocessing pipelines
├── examples/         # Aggregation and evaluation scripts
└── tests/            # Unit tests
```

## Key Components

**Configs**: Start with [`heart_presets.py`](configs/heart_presets.py) for base parameters (4kHz sample rate, 5s windows, 30-1800Hz frequency range). Model configs follow chirp's pattern—use [`efficientnetv2_linear_probe_murmur.py`](configs/efficientnetv2_linear_probe_murmur.py) for linear probing or [`efficientnetv2_partial_ft_murmur.py`](configs/efficientnetv2_partial_ft_murmur.py) for fine-tuning.

**Data**: The [`circor.py`](data/circor.py) dataset builder handles the CirCor PhysioNet 2022 Challenge dataset (murmur detection from PCG recordings at 4 auscultation sites per patient).

**Preprocessing**: [`heart_pipeline.py`](preprocessing/heart_pipeline.py) adapts chirp's pipeline for heart sounds and applies mel spectrograms with heart-appropriate parameters.

## Current State

**Plan**: Freeze Perch EfficientNetV2 backbone and train a classification head for murmur detection (linear probe + optional fine-tuning) on CirCor dataset.

**Blocker**: No public Perch EfficientNetV2 JAX/FLAX checkpoint available. Need access to pretrained weights to proceed with transfer learning.

Potential training command once checkpoints are available:

```bash
python -m chirp.train.classifier \
  --config_path pulse.configs.efficientnetv2_linear_probe_murmur \
  --workdir /path/to/workdir
```
