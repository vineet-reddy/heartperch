# coding=utf-8
"""Heart sound configuration presets for Perch linear probe pipeline.

Simplified config for embedding extraction and logistic regression training.
No mel spectrograms - embeddings are extracted from raw audio via Perch v2.
"""
from ml_collections import config_dict


def get_heart_base_config(**kwargs):
  """Creates the base config object for heart sound embedding tasks.
  
  Essential parameters only for Perch v2 embedding extraction pipeline.
  
  Args:
    **kwargs: Values to add or override in the base config.
    
  Returns:
    Config dict containing heart sound default values.
  """
  config = config_dict.ConfigDict()
  
  # Audio parameters (Perch v2 requirements)
  config.sample_rate_hz = 32000  # Perch v2 expects 32kHz
  config.window_size_s = 5.0      # Perch v2 processes 5-second windows
  config.window_stride_s = 2.5    # 50% overlap for better coverage
  
  # Dataset parameters
  config.tfds_data_dir = ''
  config.dataset_directory = 'circor/full_length:2.0.0'
  config.train_fraction = 0.8  # Patient-level train/valid split
  
  # Embedding extraction parameters
  config.batch_size = 32
  config.perch_model_name = 'perch_8'
  config.embedding_dim = 1280  # Perch 8 output dimension
  
  config.update(kwargs)
  return config

