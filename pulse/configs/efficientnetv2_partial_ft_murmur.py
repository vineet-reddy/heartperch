# coding=utf-8
"""Partial fine-tune: EfficientNetV2-S with small LR on encoder, larger on head.

Uses the same pipelines as the linear probe config.
"""
from chirp import config_utils
from ml_collections import config_dict
from pulse.configs import heart_presets
from pulse.configs import murmur_pipelines

_c = config_utils.callable_config
_o = config_utils.object_config


def get_config() -> config_dict.ConfigDict:
  # Start with heart sound base config (like chirp's get_base_config)
  config = heart_presets.get_heart_base_config()
  
  # Override dataset directories if needed
  # NOTE: Set these to your prepared TFDS location and dataset name.
  # Example: tfds_data_dir='/tmp/tfds', dataset_directory='circor/full_length:1.0.0'

  # Init: model, optimizer (with partial fine-tuning), heads metadata
  config.init_config = config_dict.ConfigDict()
  config.init_config.rng_seed = 0
  config.init_config.input_shape = heart_presets.get_heart_input_shape(config)

  # Output head metadata: single binary head named 'murmur'.
  # Use SimpleClassList from pulse for binary classification without perch_hoplite dependency.
  config.init_config.output_head_metadatas = (
      _c(
          'train_utils.OutputHeadMetadata',
          key='murmur',
          class_list=_c('class_list_utils.SimpleClassList', namespace='murmur', classes=('negative', 'positive')),
          weight=1.0,
      ),
  )

  # Model config: encoder EfficientNetV2-S, no frontend (mels in pipeline)
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c('efficientnet_v2.EfficientNetV2', model_name='efficientnetv2-s', include_top=True)
  model_config.taxonomy_loss_weight = 0.0
  model_config.frontend = None
  config.init_config.model_config = model_config

  # Partial FT: small LR on encoder, bigger on head
  config.init_config.optimizer = _c(
      'train_utils.build_partial_ft_optimizer',
      head_learning_rate=1e-3,
      encoder_learning_rate=1e-5,
      head_weight_decay=1e-4,
  )

  # Train/eval loop configs
  config.train_config = config_dict.ConfigDict()
  config.train_config.num_train_steps = config.get_ref('num_train_steps')
  config.train_config.log_every_steps = 250
  config.train_config.checkpoint_every_steps = 5_000

  config.eval_config = config_dict.ConfigDict()
  config.eval_config.num_train_steps = config.get_ref('num_train_steps')
  config.eval_config.eval_steps_per_checkpoint = 1000

  # Preprocessing Pipelines: use shared murmur pipeline builders
  # TRAIN: patient split + windowing + mel + label map + batch
  config.train_dataset_config = config_dict.ConfigDict()
  config.train_dataset_config.split = 'train'
  config.train_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  config.train_dataset_config.dataset_directory = config.get_ref('dataset_directory')
  config.train_dataset_config.pipeline = murmur_pipelines.get_train_pipeline(config)

  # EVAL: patient split (valid) + deterministic windowing + mel + label + batch
  config.eval_dataset_config = config_dict.ConfigDict()
  config.eval_dataset_config.valid = config_dict.ConfigDict()
  config.eval_dataset_config.valid.split = 'valid'
  config.eval_dataset_config.valid.tfds_data_dir = config.get_ref('tfds_data_dir')
  config.eval_dataset_config.valid.dataset_directory = config.get_ref('dataset_directory')
  config.eval_dataset_config.valid.pipeline = murmur_pipelines.get_eval_pipeline(config)
  
  return config


