#!/bin/bash
set -e

echo "Cleaning old data..."
rm -rf ~/tensorflow_datasets/circor ./embeddings_binary ./embeddings_3class ./models ./results

echo "Building CirCor dataset..."
poetry run python -m pulse.scripts.build_circor_dataset

echo "Extracting binary embeddings..."
poetry run python -m pulse.inference.embed_heart_dataset \
  --tfds_data_dir ~/tensorflow_datasets \
  --output_dir ./embeddings_binary \
  --batch_size 32

echo "Extracting 3-class embeddings..."
poetry run python -m pulse.inference.embed_heart_dataset \
  --use_3class \
  --tfds_data_dir ~/tensorflow_datasets \
  --output_dir ./embeddings_3class \
  --batch_size 32

echo "Validating embeddings..."
poetry run python -m pulse.scripts.validate_embeddings \
  --embedding_dir ./embeddings_binary

# TODO: Add validation for 3-class embeddings

echo "Experiment 1: Training binary classifier..."
poetry run python -m pulse.train.linear_probe \
  --mode binary \
  --embedding_dir ./embeddings_binary \
  --output_dir ./models/linear_probe

poetry run python -m pulse.examples.evaluate_linear_probe \
  --embedding_dir ./embeddings_binary \
  --model_path ./models/linear_probe/model.joblib \
  --output_dir ./results

echo "Experiment 2: Comparing against baselines..."
poetry run python -m pulse.examples.baseline_comparison \
  --embedding_dir ./embeddings_binary \
  --tfds_data_dir ~/tensorflow_datasets \
  --output_dir ./results

echo "Experiment 3: PhysioNet 2022 competition comparison..."
poetry run python -m pulse.examples.compete_physionet2022 \
  --use_3class \
  --embedding_dir ./embeddings_3class \
  --output_dir ./results

echo "Backing up to GCS..."
gsutil -m cp ./embeddings_binary/*.npz gs://seizurepredict-ds005873/data/circor/embeddings_binary/
gsutil -m cp ./embeddings_3class/*.npz gs://seizurepredict-ds005873/data/circor/embeddings_3class/

echo "Pipeline complete!"
