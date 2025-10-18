"""
Pydantic models for YAML pipeline configuration.

These models define and validate pipeline configurations for video processing
and image transformation workflows.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SourceConfig(BaseModel):
    """Video source configuration."""

    youtube_url: str = Field(description="YouTube video URL to download and process")

    @field_validator("youtube_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate YouTube URL format."""
        if not v or not isinstance(v, str):
            msg = "YouTube URL must be a non-empty string"
            raise ValueError(msg)
        # Basic validation - actual URL validation happens in downloader
        return v.strip()


class SegmentConfig(BaseModel):
    """Video segment selection configuration."""

    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(gt=0.0, description="End time in seconds")
    interval: float = Field(
        gt=0.0, description="Interval between frame extractions in seconds"
    )

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: float, info: ValidationInfo) -> float:
        """Validate that end time is after start time."""
        if "start" in info.data and v <= info.data["start"]:
            msg = "End time must be greater than start time"
            raise ValueError(msg)
        return v


class OperationParams(BaseModel):
    """Base class for operation parameters - allows arbitrary fields."""

    model_config = ConfigDict(extra="allow")


class ImageOperationStep(BaseModel):
    """Single image operation step in the pipeline."""

    name: str = Field(description="Human-readable name for this step")
    operation: str = Field(
        description="Operation type identifier (e.g., 'saturation', 'blur')"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific parameters"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate step name is valid for filesystem."""
        if not v or not v.strip():
            msg = "Step name cannot be empty"
            raise ValueError(msg)
        # Remove characters that might cause filesystem issues
        cleaned = "".join(c for c in v if c.isalnum() or c in ("-", "_", " "))
        return cleaned.strip()


class OutputConfig(BaseModel):
    """Output directory configuration."""

    base_dir: Path = Field(
        default=Path("./pipeline_output"),
        description="Base directory for all pipeline outputs",
    )
    intermediate_dir: Path = Field(
        default=Path("./pipeline_output/intermediate"),
        description="Directory for intermediate step outputs",
    )
    final_dir: Path = Field(
        default=Path("./pipeline_output/final"),
        description="Directory for final outputs",
    )

    @field_validator("base_dir", "intermediate_dir", "final_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: str | Path) -> Path:
        """Expand and resolve directory paths."""
        return Path(v).expanduser().resolve()


class PipelineConfig(BaseModel):
    """Root pipeline configuration model."""

    source: SourceConfig = Field(description="Video source configuration")
    segment: SegmentConfig = Field(description="Video segment selection")
    pipeline: dict[str, list[ImageOperationStep]] = Field(
        description="Pipeline steps configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline_structure(
        cls, v: dict[str, list[ImageOperationStep]]
    ) -> dict[str, list[ImageOperationStep]]:
        """Validate pipeline has steps key with at least one step."""
        if "steps" not in v:
            msg = "Pipeline must contain 'steps' key"
            raise ValueError(msg)
        if not v["steps"]:
            msg = "Pipeline must contain at least one step"
            raise ValueError(msg)
        return v

    @property
    def steps(self) -> list[ImageOperationStep]:
        """Get pipeline steps for convenience."""
        return self.pipeline["steps"]
