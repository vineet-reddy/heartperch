import dataclasses
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf


def _ensure_tf_stubs() -> None:
    if 'tensorflow_datasets' not in sys.modules:
        tfds = types.ModuleType('tensorflow_datasets')

        class _Simple:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        @dataclasses.dataclass
        class BuilderConfig:
            name: str | None = None
            version: str | None = None
            description: str | None = None

        class GeneratorBasedBuilder(_Simple):
            def __init__(self, config=None, **kwargs):
                self.builder_config = config
                super().__init__(**kwargs)

        class DatasetInfo(_Simple):
            pass

        class Tensor(_Simple):
            def get_serialized_info(self):
                return {}

            def encode_example(self, example):
                return example

            def decode_example(self, example):
                return example

        Audio = Text = _Simple

        class ClassLabel:
            def __init__(self, names=()):
                self.names = tuple(names)

        class Encoding:
            ZLIB = 'zlib'
            NONE = 'none'

        tfds.core = types.SimpleNamespace(
            BuilderConfig=BuilderConfig,
            GeneratorBasedBuilder=GeneratorBasedBuilder,
            Version=str,
            DatasetInfo=DatasetInfo,
        )
        tfds.features = types.SimpleNamespace(
            Encoding=Encoding,
            Tensor=Tensor,
            Audio=Audio,
            Text=Text,
            ClassLabel=ClassLabel,
            FeaturesDict=dict,
            DocArg=object,
        )
        tfds.typing = types.SimpleNamespace(Shape=tuple, Dim=int)

        sys.modules['tensorflow_datasets'] = tfds
        sys.modules['tensorflow_datasets.core'] = tfds.core
        sys.modules['tensorflow_datasets.features'] = tfds.features
        sys.modules['tensorflow_datasets.typing'] = tfds.typing

    if 'tensorflow_io' not in sys.modules:
        sys.modules['tensorflow_io'] = types.ModuleType('tensorflow_io')


def _config_shim(base_config):
    values = base_config.to_dict()
    values['scaling_config'] = base_config.scaling_config
    shim = types.SimpleNamespace(**values)
    shim.get_ref = lambda name: getattr(shim, name)
    return shim


_ensure_tf_stubs()

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chirp import audio_utils  # pylint: disable=wrong-import-position
from etils import epath  # pylint: disable=wrong-import-position
from pulse.configs.heart_config import get_heart_config  # pylint: disable=wrong-import-position
from pulse.data.circor import Circor, CircorConfig  # pylint: disable=wrong-import-position
from pulse.preprocessing.heart_pipeline import heart_pipeline  # pylint: disable=wrong-import-position

_DATA_ROOT = epath.Path('gs://seizurepredict-ds005873/data')


def _first_example():
    builder = Circor(config=CircorConfig(name='full_length', data_path=str(_DATA_ROOT)))
    metadata = pd.read_csv(_DATA_ROOT / 'training_data.csv')
    return next(builder._generate_examples(_DATA_ROOT, metadata))


def test_circor_example_is_normalized():
    if not (_DATA_ROOT / 'training_data.csv').exists():
        pytest.skip('GCloud data not accessible')
    
    key, example = _first_example()
    audio = example['audio']

    assert example['recording_id'] == key
    assert audio.ndim == 1 and audio.size > 0
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert audio.min() >= -1.0
    assert audio.max() <= 1.0


def test_pipeline_matches_manual_log_mel():
    config = _config_shim(
        get_heart_config(
            min_gain=1.0,
            max_gain=1.0,
            batch_size=1,
            shuffle_buffer_size=1,
            mixin_prob=0.0,
            window_size_s=1.0,
            window_stride_s=1.0,
        )
    )
    pipeline = heart_pipeline(config, is_training=False)

    sample_rate = config.sample_rate_hz
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * 200.0 * samples).astype(np.float32)

    dataset = tf.data.Dataset.from_tensors(
        {
            'audio': tf.constant(tone),
            'label': tf.constant(0, tf.int64),
            'segment_start': tf.constant(0, tf.int64),
            'segment_end': tf.constant(len(tone), tf.int64),
        }
    )
    dataset_info = types.SimpleNamespace(
        features={'audio': types.SimpleNamespace(sample_rate=sample_rate)}
    )

    batch = next(iter(pipeline(dataset, dataset_info)))
    pipeline_spec = batch['audio'][0].numpy()

    kernel = int(config.win_length_s * sample_rate)
    hop = int(config.hop_length_s * sample_rate)
    stfts = audio_utils.stft_tf(
        tf.constant(tone),
        nperseg=kernel,
        noverlap=kernel - hop,
        nfft=config.n_fft,
        padded=False,
    )
    if tone.shape[0] % hop == 0:
        stfts = stfts[..., :-1]
    stfts = tf.experimental.numpy.swapaxes(stfts, -1, -2)
    magnitude = tf.math.abs(stfts)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        config.n_mels,
        magnitude.shape[-1],
        sample_rate,
        config.fmin,
        config.fmax,
    )
    manual_spec = tf.math.maximum(magnitude @ mel_matrix, config.scaling_config.floor)
    manual_spec = config.scaling_config.scalar * tf.math.log(
        manual_spec + config.scaling_config.offset
    )

    np.testing.assert_allclose(
        pipeline_spec,
        manual_spec.numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
