"""Python SDK for the ai-prophet forecasting datasets registry."""

from .models import (
    DatasetMetadata,
    ReleaseMetadata,
    ResolvedOutcome,
    Task,
)
from .registry import Dataset, Registry, Release, ReleaseSummary
from .validation import SchemaError

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "Registry",
    "Release",
    "ReleaseMetadata",
    "ReleaseSummary",
    "ResolvedOutcome",
    "SchemaError",
    "Task",
]

__version__ = "0.1.0"
