# coding=utf-8
"""Simple class list utilities for heart configs."""

import dataclasses


@dataclasses.dataclass
class SimpleClassList:
  """Class list for binary/simple classification tasks."""
  namespace: str
  classes: tuple[str, ...]

