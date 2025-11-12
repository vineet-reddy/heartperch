"""CirCor heart sound dataset builder."""

import dataclasses

from chirp import audio_utils
from chirp.data import tfds_features
from etils import epath
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


@dataclasses.dataclass
class CircorConfig(tfds.core.BuilderConfig):
  sample_rate_hz: int = 32000
  resampling_method: str = 'polyphase'
  data_path: str = 'gs://seizurepredict-ds005873/data'


class Circor(tfds.core.GeneratorBasedBuilder):
  """CirCor PCG dataset: murmur detection from heart sounds."""
  
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Fixed recording-level labels to use Murmur locations column. '
               'Previously all recordings from a patient received the same label; '
               'now only recordings at murmur locations are labeled positive.',
  }
  BUILDER_CONFIGS = [CircorConfig(name='full_length')]
  
  # Explicitly set code_path to avoid TFDS auto-detection bug with GCS paths
  # The auto-detection fails with MultiplexedPath objects
  code_path = epath.Path(__file__)
  
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'audio': tfds_features.Int16AsFloatTensor(
                shape=[None],
                sample_rate=self.builder_config.sample_rate_hz,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            'recording_id': tfds.features.Text(),
            'patient_id': tfds.features.Text(),
            'location': tfds.features.ClassLabel(names=['AV', 'MV', 'PV', 'TV']),
            'murmur': tfds.features.ClassLabel(names=['Absent', 'Present', 'Unknown']),
            'outcome': tfds.features.ClassLabel(names=['Abnormal', 'Normal']),
            # Segment fields expected by chirp preprocessing pipeline
            'segment_id': tfds.features.Scalar(dtype=np.int64),
            'segment_start': tfds.features.Scalar(dtype=np.int64),
            'segment_end': tfds.features.Scalar(dtype=np.int64),
        }),
        supervised_keys=('audio', 'murmur'),
    )
  
  def _split_generators(self, dl_manager):
    data_path = epath.Path(self.builder_config.data_path)
    metadata = pd.read_csv(data_path / 'training_data.csv')
    return {'train': self._generate_examples(data_path, metadata)}
  
  def _generate_examples(self, data_path, metadata_df):
    audio_dir = data_path / 'training_data'
    
    for _, row in metadata_df.iterrows():
      patient_id = str(row['Patient ID'])
      locations_str = str(row['Recording locations:'])
      murmur_patient = str(row['Murmur'])  # Patient-level murmur status
      murmur_locations_str = str(row['Murmur locations'])  # Where murmur is audible
      outcome = str(row['Outcome'])
      
      if pd.isna(locations_str) or locations_str == 'nan':
        continue
      if murmur_patient not in ['Present', 'Absent', 'Unknown']:
        continue
      if outcome not in ['Normal', 'Abnormal']:
        continue
      
      # Parse murmur locations (e.g., "AV+MV+TV" or "TV")
      # Only relevant if patient has a murmur
      murmur_location_set = set()
      if murmur_patient == 'Present' and not pd.isna(murmur_locations_str) and murmur_locations_str != 'nan':
        murmur_location_set = set(loc.strip() for loc in murmur_locations_str.split('+'))
      
      for location in locations_str.split('+'):
        location = location.strip()
        if location not in ['AV', 'MV', 'PV', 'TV']:
          continue
          
        wav_file = audio_dir / f"{patient_id}_{location}.wav"
        if not wav_file.exists():
          continue
        
        try:
          audio = audio_utils.load_audio_file(
              str(wav_file),
              target_sample_rate=self.builder_config.sample_rate_hz,
              resampling_type=self.builder_config.resampling_method,
          )
          # Resampling can introduce artifacts that push the signal outside the
          # [-1, 1) interval. Following chirp's pattern from bird_taxonomy.py
          # and soundscapes.py.
          # TODO(vineet): Validate this approach for heart sounds. The 8x upsampling
          # (4kHz -> 32kHz) causes ~28% overshoots. Clipping may introduce harmonic
          # distortion that could affect murmur detection. Consider alternatives:
          # (1) Per-file peak normalization to preserve waveform shape
          # (2) Different resampling method (kaiser_best, soxr_hq)
          # (3) Lower target sample rate if Perch supports it
          # Run ablation study to measure impact on downstream classification.
          audio = np.clip(audio, -1.0, 1.0 - (1.0 / float(1 << 15)))
        except Exception:
          continue
        
        # Skip empty/silent recordings. Note: audio.max() scans the full array,
        # but worth the cost for data quality.
        if audio.shape[0] == 0 or audio.max() == 0.0:
          continue
        
        # Determine recording-level murmur label based on location
        if murmur_patient == 'Present':
          # Only mark as Present if this specific location has the murmur
          murmur_recording = 'Present' if location in murmur_location_set else 'Absent'
        else:
          # If patient-level is Absent or Unknown, recording inherits that status
          murmur_recording = murmur_patient
        
        yield f"{patient_id}_{location}", {
            'audio': audio,
            'recording_id': f"{patient_id}_{location}",
            'patient_id': patient_id,
            'location': location,
            'murmur': murmur_recording,  # Now location-specific!
            'outcome': outcome,
            # Add segment fields expected by chirp preprocessing pipeline
            'segment_id': 0,
            'segment_start': 0,
            'segment_end': len(audio),
        }
