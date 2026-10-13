"""Model registry and adapters. Every adapter wraps one registered model behind a small
interface so models can be replaced without touching the pipeline."""
from .registry import MODELS, ModelUnavailable, all_status, available, download, model_record, require

__all__ = ["MODELS", "ModelUnavailable", "all_status", "available", "download", "model_record", "require"]
