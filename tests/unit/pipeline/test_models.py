"""Tests for pipeline configuration models."""

import pytest
from pydantic import ValidationError
from sevenrad_stills.pipeline.models import (
    ImageOperationStep,
    PipelineConfig,
    SegmentConfig,
    SourceConfig,
)


class TestSourceConfig:
    """Tests for SourceConfig model."""

    def test_valid_url(self) -> None:
        """Test valid YouTube URL."""
        config = SourceConfig(youtube_url="https://youtube.com/watch?v=test")
        assert config.youtube_url == "https://youtube.com/watch?v=test"

    def test_url_stripped(self) -> None:
        """Test URL whitespace is stripped."""
        config = SourceConfig(youtube_url="  https://youtube.com/watch?v=test  ")
        assert config.youtube_url == "https://youtube.com/watch?v=test"

    def test_empty_url_raises_error(self) -> None:
        """Test empty URL raises validation error."""
        with pytest.raises(ValidationError):
            SourceConfig(youtube_url="")


class TestSegmentConfig:
    """Tests for SegmentConfig model."""

    def test_valid_segment(self) -> None:
        """Test valid segment configuration."""
        config = SegmentConfig(start=10.0, end=30.0, interval=1.0)
        assert config.start == 10.0
        assert config.end == 30.0
        assert config.interval == 1.0

    def test_end_before_start_raises_error(self) -> None:
        """Test end before start raises validation error."""
        with pytest.raises(ValidationError):
            SegmentConfig(start=30.0, end=10.0, interval=1.0)

    def test_negative_start_raises_error(self) -> None:
        """Test negative start time raises error."""
        with pytest.raises(ValidationError):
            SegmentConfig(start=-1.0, end=10.0, interval=1.0)

    def test_zero_interval_raises_error(self) -> None:
        """Test zero interval raises error."""
        with pytest.raises(ValidationError):
            SegmentConfig(start=0.0, end=10.0, interval=0.0)


class TestImageOperationStep:
    """Tests for ImageOperationStep model."""

    def test_valid_step(self) -> None:
        """Test valid operation step."""
        step = ImageOperationStep(
            name="test_step",
            operation="saturation",
            params={"mode": "fixed", "value": 1.5},
        )
        assert step.name == "test_step"
        assert step.operation == "saturation"
        assert step.params == {"mode": "fixed", "value": 1.5}

    def test_name_sanitization(self) -> None:
        """Test step name is sanitized for filesystem."""
        step = ImageOperationStep(name="test@#$step", operation="saturation", params={})
        assert step.name == "teststep"

    def test_empty_name_raises_error(self) -> None:
        """Test empty name raises validation error."""
        with pytest.raises(ValidationError):
            ImageOperationStep(name="", operation="saturation", params={})


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_valid_pipeline(self) -> None:
        """Test valid pipeline configuration."""
        config = PipelineConfig(
            source=SourceConfig(youtube_url="https://youtube.com/watch?v=test"),
            segment=SegmentConfig(start=10.0, end=30.0, interval=1.0),
            pipeline={
                "steps": [
                    ImageOperationStep(
                        name="saturate", operation="saturation", params={}
                    )
                ]
            },
        )
        assert len(config.steps) == 1
        assert config.steps[0].name == "saturate"

    def test_missing_steps_key_raises_error(self) -> None:
        """Test missing 'steps' key raises error."""
        with pytest.raises(ValidationError):
            PipelineConfig(
                source=SourceConfig(youtube_url="https://youtube.com/watch?v=test"),
                segment=SegmentConfig(start=10.0, end=30.0, interval=1.0),
                pipeline={},
            )

    def test_empty_steps_raises_error(self) -> None:
        """Test empty steps list raises error."""
        with pytest.raises(ValidationError):
            PipelineConfig(
                source=SourceConfig(youtube_url="https://youtube.com/watch?v=test"),
                segment=SegmentConfig(start=10.0, end=30.0, interval=1.0),
                pipeline={"steps": []},
            )
