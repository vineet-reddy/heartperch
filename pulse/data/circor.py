"""CirCor heart sound dataset builder."""

import dataclasses

from chirp import audio_utils
from chirp.data import tfds_features
from etils import epath
import pandas as pd
import tensorflow_datasets as tfds


@dataclasses.dataclass
class CircorConfig(tfds.core.BuilderConfig):
  sample_rate_hz: int = 4000
  resampling_method: str = 'polyphase'
  data_path: str = 'gs://seizurepredict-ds005873/data'


class Circor(tfds.core.GeneratorBasedBuilder):
  """CirCor PCG dataset: murmur detection from heart sounds."""
  
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}
  BUILDER_CONFIGS = [CircorConfig(name='full_length')]
  
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
      murmur = str(row['Murmur'])
      outcome = str(row['Outcome'])
      
      if pd.isna(locations_str) or locations_str == 'nan':
        continue
      if murmur not in ['Present', 'Absent', 'Unknown']:
        continue
      if outcome not in ['Normal', 'Abnormal']:
        continue
      
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
        except Exception:
          continue
        
        # Skip empty/silent recordings. Note: audio.max() scans the full array,
        # but worth the cost for data quality.
        if audio.shape[0] == 0 or audio.max() == 0.0:
          continue
        
        yield f"{patient_id}_{location}", {
            'audio': audio,
            'recording_id': f"{patient_id}_{location}",
            'patient_id': patient_id,
            'location': location,
            'murmur': murmur,
            'outcome': outcome,
        }
