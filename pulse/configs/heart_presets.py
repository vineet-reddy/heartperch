# coding=utf-8
"""Heart sound preprocessing presets and base configurations.

This module provides base configuration builders for heart sound classification,
following the same pattern as chirp.configs.presets.
"""
from chirp import config_utils
from chirp.models import frontend
from ml_collections import config_dict

_c = config_utils.callable_config
_o = config_utils.object_config


def get_heart_base_config(**kwargs):
  """Creates the base config object for heart sound tasks.
  
  Contains common values and FieldReferences for heart sound classification.
  Individual configs can override any of these by passing kwargs or updating
  the returned config.
  
  Args:
    **kwargs: Values to add or override in the base config.
    
  Returns:
    Config dict containing common heart sound default values.
  """
  config = config_dict.ConfigDict()
  
  # Audio parameters
  config.sample_rate_hz = 4000
  config.train_window_size_s = 5.0
  config.eval_window_size_s = 5.0
  config.window_stride_s = 2.5
  
  # Mel spectrogram parameters
  config.win_length_s = 0.02  # STFT window: 20ms
  config.hop_length_s = 0.01  # STFT hop: 10ms
  config.n_mels = 128
  config.n_fft = 256
  config.freq_range = (30.0, 1800.0)  # Heart sounds frequency range
  
  # Training parameters
  config.batch_size = 32
  config.shuffle_buffer_size = 1000
  config.mixin_prob = 0.2
  config.num_train_steps = 50_000
  
  # Normalization
  config.min_gain = 0.25
  config.max_gain = 1.0
  
  # Scaling config for mel spectrograms
  config.scaling_config = frontend.LogScalingConfig(
      floor=1e-5, offset=0.0, scalar=0.1
  )
  
  # Loss function: softmax_cross_entropy for binary classification with 2 classes.
  # Model outputs [B, 2] logits, labels are [B] class indices (0 or 1).
  config.loss_fn = _o('optax.softmax_cross_entropy')
  
  # Dataset directories
  config.tfds_data_dir = ''
  config.dataset_directory = 'circor/full_length:1.0.0'
  
  config.update(kwargs)
  return config


def get_heart_input_shape(config: config_dict.ConfigDict):
  """Computes the mel spectrogram input shape for heart sound configs.
  
  Args:
    config: Config dict containing sample_rate_hz, train_window_size_s,
      hop_length_s, win_length_s, and n_mels.
      
  Returns:
    Tuple of (num_frames, num_mels) representing the expected input shape.
  """
  num_samples = config.get_ref('train_window_size_s') * config.get_ref('sample_rate_hz')
  stride = config.get_ref('hop_length_s') * config.get_ref('sample_rate_hz')
  kernel_size = config.get_ref('win_length_s') * config.get_ref('sample_rate_hz')
  return config_utils.compute_melspec_shape(
      num_samples=num_samples,
      stride=stride,
      kernel_size=kernel_size,
      num_mels=config.get_ref('n_mels')
  )

