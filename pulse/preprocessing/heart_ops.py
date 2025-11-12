# coding=utf-8
"""Heart-specific preprocessing ops.

Includes:
- MapMurmurToBinary: converts CirCor 'murmur' ClassLabel to a binary 'label'.
- MapMurmurTo3Class: keeps full 3-class murmur labels for competition comparison.
- PatientHashSplit: filters examples into train/valid based on patient hash.
"""

import dataclasses

from chirp.preprocessing import pipeline as pipeline_
import tensorflow as tf
import tensorflow_datasets as tfds


Features = dict[str, tf.Tensor]


@dataclasses.dataclass
class MapMurmurToBinary(pipeline_.FeaturesPreprocessOp):
  """Create a binary 'label' from the CirCor 'murmur' ClassLabel.

  Mapping: Present -> 1; Absent/Unknown -> 0.
  """

  murmur_key: str = 'murmur'
  out_key: str = 'label'

  def __call__(self, features: Features, dataset_info: tfds.core.DatasetInfo) -> Features:
    features = features.copy()
    if self.murmur_key not in features:
      return features

    # 'murmur' is a ClassLabel in CirCor builder with names:
    # ['Absent', 'Present', 'Unknown']
    names = tuple(dataset_info.features[self.murmur_key].names)
    present_idx = names.index('Present') if 'Present' in names else 1

    murmur_val = features[self.murmur_key]
    # Ensure int64 for consistency downstream
    murmur_val = tf.cast(murmur_val, tf.int64)
    label = tf.where(
        tf.equal(murmur_val, present_idx), tf.ones_like(murmur_val), tf.zeros_like(murmur_val)
    )
    # Keep as int32 class index (0 or 1) for use with softmax_cross_entropy.
    # After batching becomes [B], compatible with model output logits [B, 2].
    label = tf.cast(label, tf.int32)
    features[self.out_key] = label
    return features


@dataclasses.dataclass
class MapMurmurTo3Class(pipeline_.FeaturesPreprocessOp):
  """Keep full 3-class murmur labels (Present/Unknown/Absent).
  
  For PhysioNet 2022 competition comparison, we need the full 3-class
  labels to compute the exact weighted accuracy metric.
  
  Mapping: Absent -> 0, Present -> 1, Unknown -> 2
  (This matches the CirCor TFDS ClassLabel encoding)
  """
  
  murmur_key: str = 'murmur'
  out_key: str = 'label'
  
  def __call__(self, features: Features, dataset_info: tfds.core.DatasetInfo) -> Features:
    features = features.copy()
    if self.murmur_key not in features:
      return features
    
    # Keep raw murmur ClassLabel as-is (0=Absent, 1=Present, 2=Unknown)
    murmur_val = features[self.murmur_key]
    # Cast to int32 for compatibility with model training
    label = tf.cast(murmur_val, tf.int32)
    features[self.out_key] = label
    return features


@dataclasses.dataclass
class PatientHashSplit(pipeline_.DatasetPreprocessOp):
  """Deterministic patient-wise split using hash of 'patient_id'.

  Examples where hash(patient_id) % modulo < train_threshold go to 'train'.
  Others go to 'valid'.
  """

  split: str = 'train'  # 'train' or 'valid'
  patient_key: str = 'patient_id'
  train_fraction: float = 0.8
  modulo: int = 1024

  def __call__(self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo) -> tf.data.Dataset:
    del dataset_info  # Unused

    train_threshold = tf.cast(tf.round(self.train_fraction * self.modulo), tf.int64)

    def _predicate(features: Features) -> tf.Tensor:
      pid = tf.cast(features[self.patient_key], tf.string)
      bucket = tf.strings.to_hash_bucket_fast(pid, num_buckets=self.modulo)
      if self.split == 'train':
        return bucket < train_threshold
      else:
        return bucket >= train_threshold

    return dataset.filter(_predicate)


