"""Build CirCor TFDS dataset locally.

Reads raw audio from GCS (gs://seizurepredict-ds005873/data/training_data/)
and builds processed TFDS dataset to ~/tensorflow_datasets/.

Usage:
  poetry run python -m pulse.scripts.build_circor_dataset
"""

import os
import tensorflow_datasets as tfds
from pulse.data.circor import Circor


def main():
  """Build CirCor dataset to ~/tensorflow_datasets."""
  data_dir = os.path.expanduser('~/tensorflow_datasets')
  
  # Instantiate builder
  builder = Circor(config='full_length', data_dir=data_dir)
  
  print(f'Building CirCor dataset...')
  print(f'  Reading raw WAV files from: {builder.builder_config.data_path}/training_data/')
  print(f'  Writing TFDS dataset to: {data_dir}/circor/full_length/{builder.version}/')
  print()
  
  # Skip checksum verification (we don't download external files, just read from GCS)
  download_config = tfds.download.DownloadConfig(
      register_checksums=False,
  )
  
  builder.download_and_prepare(download_config=download_config)
  
  print('\n=== Dataset built successfully! ===')
  print(f'Location: {data_dir}/circor/full_length/{builder.version}/')


if __name__ == '__main__':
  main()

