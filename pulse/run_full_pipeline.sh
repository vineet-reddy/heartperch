#!/bin/bash
set -e

# Usage: ./pulse/run_full_pipeline.sh
# Flags:
#   MODEL_NAME=<model>    - run specific model (perch_8, perch_v2, surfperch)
#   REBUILD_DATASET=1     - erase and rebuild dataset from scratch
# Examples: 
#   MODEL_NAME=perch_8 ./pulse/run_full_pipeline.sh
#   REBUILD_DATASET=1 ./pulse/run_full_pipeline.sh
#   REBUILD_DATASET=1 MODEL_NAME=perch_8 ./pulse/run_full_pipeline.sh

# If MODEL_NAME is set, use it. Otherwise run all models.
if [ -z "$MODEL_NAME" ]; then
  MODELS_TO_RUN="perch_8 perch_v2 surfperch"
else
  MODELS_TO_RUN="$MODEL_NAME"
fi

# Build dataset once (if needed)
# If REBUILD_DATASET is set, force rebuild by removing existing dataset first
if [ -n "$REBUILD_DATASET" ]; then
  echo "REBUILD_DATASET flag detected. Removing existing dataset..."
  rm -rf ~/tensorflow_datasets/circor
fi

if [ ! -d ~/tensorflow_datasets/circor ]; then
  echo "Building CirCor dataset..."
  poetry run python -m pulse.scripts.build_circor_dataset
fi

# Clean all old data before starting
echo "Cleaning all old data..."
rm -rf ./embeddings_binary ./embeddings_3class ./models ./results ./results_all_models

# Loop through models
for MODEL_NAME in $MODELS_TO_RUN; do
  echo "Running pipeline for ${MODEL_NAME}..."
  
  echo "Extracting binary embeddings with ${MODEL_NAME}..."
  poetry run python -m pulse.inference.embed_heart_dataset \
    --model_name ${MODEL_NAME} \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./embeddings_binary/${MODEL_NAME} \
    --batch_size 32
  
  echo "Extracting 3-class embeddings with ${MODEL_NAME}..."
  poetry run python -m pulse.inference.embed_heart_dataset \
    --model_name ${MODEL_NAME} \
    --use_3class \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./embeddings_3class/${MODEL_NAME} \
    --batch_size 32
  
  echo "Validating embeddings..."
  poetry run python -m pulse.scripts.validate_embeddings \
    --embedding_dir ./embeddings_binary/${MODEL_NAME}
  
  echo "Experiment 1: Training binary classifier..."
  poetry run python -m pulse.train.linear_probe \
    --mode binary \
    --embedding_dir ./embeddings_binary/${MODEL_NAME} \
    --output_dir ./models/${MODEL_NAME}/linear_probe
  
  poetry run python -m pulse.examples.evaluate_linear_probe \
    --embedding_dir ./embeddings_binary/${MODEL_NAME} \
    --model_path ./models/${MODEL_NAME}/linear_probe/model.joblib \
    --output_dir ./results/${MODEL_NAME}
  
  echo "Experiment 2: Comparing against baselines..."
  poetry run python -m pulse.examples.baseline_comparison \
    --embedding_dir ./embeddings_binary/${MODEL_NAME} \
    --tfds_data_dir ~/tensorflow_datasets \
    --output_dir ./results/${MODEL_NAME}
  
  echo "Experiment 3: PhysioNet 2022 competition comparison..."
  poetry run python -m pulse.examples.compete_physionet2022 \
    --use_3class \
    --embedding_dir ./embeddings_3class/${MODEL_NAME} \
    --output_dir ./results/${MODEL_NAME}
  
  # Save results with model name
  mkdir -p ./results_all_models/${MODEL_NAME}
  cp ./results/${MODEL_NAME}/metrics.csv ./results_all_models/${MODEL_NAME}/metrics.csv
  cp ./results/${MODEL_NAME}/baseline_comparison.csv ./results_all_models/${MODEL_NAME}/baseline_comparison.csv
  
  echo "Pipeline complete for ${MODEL_NAME}!"
done

# Compare all models if we ran multiple
if [ $(echo $MODELS_TO_RUN | wc -w) -gt 1 ]; then
  echo "Comparing all models..."
  poetry run python -m pulse.scripts.compare_models --results_dir ./results_all_models
fi

echo "Pipeline complete!"
