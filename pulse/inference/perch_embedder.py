# coding=utf-8
"""Wrapper for Perch 8 model for embedding extraction.

This module provides a simple interface to load the Perch 8 model and extract
1280-dimensional embeddings from 5-second audio windows at 32 kHz.
"""

import numpy as np
from perch_hoplite.zoo import model_configs


class PerchEmbedder:
  """Wrapper around Perch 8 model for embedding extraction.
  
  Perch 8 expects mono audio at 32 kHz in 5-second windows.
  Outputs 1280-dimensional embedding vectors.
  
  Example:
    embedder = PerchEmbedder()
    audio = np.random.randn(160000)  # 5s at 32kHz
    embedding = embedder.embed(audio)  # Shape: (1280,)
  """
  
  def __init__(self, model_name: str = 'perch_8'):
    """Initialize the Perch 8 model.
    
    Args:
      model_name: Name of the model to load. Default is 'perch_8'.
    """
    self.model_name = model_name
    self.sample_rate = 32000
    self.window_size_s = 5.0
    self.expected_samples = int(self.sample_rate * self.window_size_s)
    self.embedding_dim = 1280
    
    # Load model
    print(f'Loading {model_name} model...')
    self.model = model_configs.load_model_by_name(model_name)
    print(f'Model loaded. Sample rate: {self.model.sample_rate} Hz')
    
  def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
    """Prepare audio for Perch 8: ensure mono, correct length.
    
    Args:
      audio: Audio array of shape (T,) or (1, T). Should be at 32 kHz.
      
    Returns:
      Prepared audio of shape (expected_samples,) = (160000,)
    """
    # Ensure 1D
    if audio.ndim > 1:
      audio = audio.squeeze()
    if audio.ndim > 1:
      # Take first channel if still multi-channel
      audio = audio[0]
    
    # Ensure exactly expected_samples length
    if len(audio) < self.expected_samples:
      # Pad with zeros
      audio = np.pad(audio, (0, self.expected_samples - len(audio)), mode='constant')
    elif len(audio) > self.expected_samples:
      # Trim to exact length
      audio = audio[:self.expected_samples]
    
    return audio.astype(np.float32)
  
  def embed(self, audio: np.ndarray) -> np.ndarray:
    """Extract embedding from a single audio window.
    
    Args:
      audio: Audio array of shape (T,) at 32 kHz, ideally 5 seconds (160000 samples).
             Will be padded/trimmed to exactly 160000 samples.
      
    Returns:
      Embedding vector of shape (1280,)
    """
    audio = self._prepare_audio(audio)
    outputs = self.model.embed(audio)
    
    if outputs.embeddings is None:
      raise ValueError('Model did not produce embeddings!')
    
    # Embeddings may have shape [1, 1, D], [1, D], or [D]
    embedding = np.squeeze(outputs.embeddings)
    
    return embedding
  
  def embed_batch(self, audio_batch: np.ndarray) -> np.ndarray:
    """Extract embeddings from a batch of audio windows.
    
    Args:
      audio_batch: Batch of audio arrays of shape (B, T) at 32 kHz.
      
    Returns:
      Embedding matrix of shape (B, 1280)
    """
    batch_size = audio_batch.shape[0]
    embeddings = np.zeros((batch_size, self.embedding_dim), dtype=np.float32)
    
    for i in range(batch_size):
      embeddings[i] = self.embed(audio_batch[i])
    
    return embeddings

