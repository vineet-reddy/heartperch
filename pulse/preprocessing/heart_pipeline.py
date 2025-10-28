"""Heart sound preprocessing pipeline."""

from chirp.preprocessing.pipeline import (
    Batch,
    ExtractStridedWindows,
    MelSpectrogram,
    MixAudio,
    OnlyKeep,
    Pipeline,
    RandomNormalizeAudio,
    RepeatPadding,
    Shuffle,
)
from ml_collections import config_dict


def heart_pipeline(config: config_dict.ConfigDict, is_training: bool = True) -> Pipeline:
    """Preprocessing pipeline: audio → log-mel spectrograms.
    
    Reuses chirp.preprocessing.pipeline components. Audio resampling to 4000 Hz
    happens in the dataset builder (pulse/data/circor.py).
    """
    ops = []
    
    ops.append(RandomNormalizeAudio(
        min_gain=config.get_ref('min_gain'),
        max_gain=config.get_ref('max_gain'),
        names=('audio',)
    ))
    
    ops.append(RepeatPadding(
        pad_size=config.get_ref('window_size_s'),
        sample_rate=config.get_ref('sample_rate_hz')
    ))
    
    ops.append(ExtractStridedWindows(
        window_length_sec=config.get_ref('window_size_s'),
        window_stride_sec=config.get_ref('window_stride_s'),
        sample_rate=config.get_ref('sample_rate_hz')
    ))
    
    # waveform → STFT → magnitude → mel → log
    ops.append(MelSpectrogram(
        features=config.get_ref('n_mels'),
        kernel_size=int(config.win_length_s * config.sample_rate_hz),
        stride=int(config.hop_length_s * config.sample_rate_hz),
        sample_rate=config.get_ref('sample_rate_hz'),
        freq_range=(config.fmin, config.fmax),
        name='audio',
        power=1.0,
        scaling_config=config.get_ref('scaling_config'),
        nfft=config.get_ref('n_fft')
    ))
    
    if is_training:
        ops.append(Shuffle(shuffle_buffer_size=config.get_ref('shuffle_buffer_size')))
        ops.append(MixAudio(mixin_prob=config.get_ref('mixin_prob')))
    
    ops.append(OnlyKeep(['audio', 'label']))
    ops.append(Batch(config.get_ref('batch_size'), drop_remainder=is_training))
    
    return Pipeline(ops=ops)