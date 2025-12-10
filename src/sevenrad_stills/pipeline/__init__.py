"""Video processing pipeline."""

from sevenrad_stills.pipeline.models import (
    ImageOperationStep,
    OutputConfig,
    PipelineConfig,
    SegmentConfig,
    SourceConfig,
)
from sevenrad_stills.pipeline.yaml_loader import (
    PipelineLoadError,
    load_pipeline_config,
    validate_pipeline_yaml,
)


def __getattr__(name: str) -> object:
    """
    Lazy import for modules that cause circular imports.

    PipelineExecutor and VideoProcessor import from operations, which causes
    circular imports when operations import from pipeline.protocols/types.
    """
    if name == "PipelineExecutor":
        from sevenrad_stills.pipeline.executor import PipelineExecutor

        return PipelineExecutor
    if name == "VideoProcessor":
        from sevenrad_stills.pipeline.processor import VideoProcessor

        return VideoProcessor
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ImageOperationStep",
    "OutputConfig",
    "PipelineConfig",
    "PipelineExecutor",
    "PipelineLoadError",
    "SegmentConfig",
    "SourceConfig",
    "VideoProcessor",
    "load_pipeline_config",
    "validate_pipeline_yaml",
]
