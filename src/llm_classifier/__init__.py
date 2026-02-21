"""llm-classifier: LLM-based classification and prediction with structured outputs."""

from .classifier import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    BatchResult,
    LLMClassifier,
    PredictResult,
    create_wrapped_model,
    unwrap_response,
)
from .cluster import (
    DEFAULT_CLUSTER_SYSTEM_PROMPT,
    DEFAULT_CLUSTER_USER_PROMPT,
    ClusterItem,
    ClusterResult,
    ClusterValidationError,
    ContextLengthError,
    LLMCluster,
)

__version__ = "0.1.0"
__all__ = [
    "LLMClassifier",
    "PredictResult",
    "BatchResult",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "LLMCluster",
    "ClusterResult",
    "ClusterItem",
    "ClusterValidationError",
    "ContextLengthError",
    "DEFAULT_CLUSTER_SYSTEM_PROMPT",
    "DEFAULT_CLUSTER_USER_PROMPT",
    "create_wrapped_model",
    "unwrap_response",
]
