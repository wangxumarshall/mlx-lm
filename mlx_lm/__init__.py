# Copyright © 2023-2024 Apple Inc.

import os

from ._version import __version__

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .convert import convert
from .generate import batch_generate, generate, stream_generate
from .utils import load
from .deep_thinking import (
    DeepThinkingConfig,
    DeepThinkingResult,
    ThinkAtN,
    compute_deep_thinking_ratio,
    create_deep_thinking_config,
)
from .qwen35_optimizer import Qwen35DeepThinkingOptimizer

__all__ = [
    "__version__",
    "convert",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
    "DeepThinkingConfig",
    "DeepThinkingResult",
    "ThinkAtN",
    "compute_deep_thinking_ratio",
    "create_deep_thinking_config",
    "Qwen35DeepThinkingOptimizer",
]
