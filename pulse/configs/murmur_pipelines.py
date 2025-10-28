# coding=utf-8
"""Shared data pipelines for murmur binary classification."""
from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config


def _base_ops(config, window_size_ref):
  """Common preprocessing ops for murmur classification."""
  return [
      _c('pipeline.OnlyJaxTypes'),
      _c('pipeline.RepeatPadding', 
         pad_size=config.get_ref(window_size_ref), 
         sample_rate=config.get_ref('sample_rate_hz')),
      _c('heart_ops.MapMurmurToBinary'),
      _c('pipeline.ExtractStridedWindows',
         window_length_sec=config.get_ref(window_size_ref),
         window_stride_sec=config.get_ref('window_stride_s'),
         sample_rate=config.get_ref('sample_rate_hz')),
      _c('pipeline.MelSpectrogram',
         features=config.get_ref('n_mels'),
         kernel_size=config.get_ref('win_length_s') * config.get_ref('sample_rate_hz'),
         stride=config.get_ref('hop_length_s') * config.get_ref('sample_rate_hz'),
         sample_rate=config.get_ref('sample_rate_hz'),
         freq_range=config.get_ref('freq_range'),
         name='audio',
         power=1.0,
         scaling_config=config.get_ref('scaling_config'),
         nfft=config.get_ref('n_fft')),
  ]


def get_train_pipeline(config: config_dict.ConfigDict):
  """Training pipeline: patient split + preprocessing + shuffle + batch."""
  return _c('pipeline.Pipeline', ops=[
      _c('heart_ops.PatientHashSplit', split='train', train_fraction=0.8),
      *_base_ops(config, 'train_window_size_s'),
      _c('pipeline.Shuffle', shuffle_buffer_size=config.get_ref('shuffle_buffer_size')),
      # MixAudio disabled: mixing patients corrupts binary labels for murmur detection.
      # For medical signals, mixing dilutes SNR and creates invalid training examples.
      # _c('pipeline.MixAudio', mixin_prob=config.get_ref('mixin_prob')),
      _c('pipeline.OnlyKeep', names=['audio', 'label']),
      _c('pipeline.Batch', batch_size=config.get_ref('batch_size'), drop_remainder=True),
  ])


def get_eval_pipeline(config: config_dict.ConfigDict):
  """Eval pipeline: patient split + preprocessing + batch (deterministic, keeps metadata)."""
  return _c('pipeline.Pipeline', ops=[
      _c('heart_ops.PatientHashSplit', split='valid', train_fraction=0.8),
      *_base_ops(config, 'eval_window_size_s'),
      _c('pipeline.OnlyKeep', names=['audio', 'label', 'patient_id', 'recording_id', 'segment_id']),
      _c('pipeline.Batch', batch_size=64, drop_remainder=False),
  ])

