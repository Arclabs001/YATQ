"""
TurboQuant Integrations

Integration modules for various LLM frameworks.
"""

from .hf_integration import TurboQuantHF, TurboQuantHFWithCache, apply_turboquant
from .qwen3_integration import Qwen3ForwardWithTurboQuant, ChunkedKVCacheQJL

__all__ = [
    # HuggingFace (MSE-only, works with any model)
    'TurboQuantHF',
    'TurboQuantHFWithCache',
    'apply_turboquant',
    # Qwen3 (Full QJL support)
    'Qwen3ForwardWithTurboQuant',
    'ChunkedKVCacheQJL',
]