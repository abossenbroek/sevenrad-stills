"""Tests for extraction.strategies module."""

import pytest
from sevenrad_stills.extraction.strategies import (
    FPSExtractionStrategy,
    IntervalExtractionStrategy,
    create_extraction_strategy,
)


class TestFPSExtractionStrategy:
    """Tests for FPSExtractionStrategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = FPSExtractionStrategy(fps=2.5)
        assert strategy.fps == 2.5

    def test_ffmpeg_filter(self) -> None:
        """Test FFmpeg filter generation."""
        strategy = FPSExtractionStrategy(fps=1.0)
        assert strategy.get_ffmpeg_filter() == "fps=1.0"

        strategy2 = FPSExtractionStrategy(fps=0.5)
        assert strategy2.get_ffmpeg_filter() == "fps=0.5"

    def test_description(self) -> None:
        """Test description generation."""
        strategy = FPSExtractionStrategy(fps=2.0)
        desc = strategy.get_description()
        assert "2.0" in desc
        assert "FPS" in desc.upper()


class TestIntervalExtractionStrategy:
    """Tests for IntervalExtractionStrategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = IntervalExtractionStrategy(interval=30)
        assert strategy.interval == 30

    def test_ffmpeg_filter(self) -> None:
        """Test FFmpeg filter generation."""
        strategy = IntervalExtractionStrategy(interval=10)
        assert strategy.get_ffmpeg_filter() == "select='not(mod(n,10))'"

        strategy2 = IntervalExtractionStrategy(interval=60)
        assert strategy2.get_ffmpeg_filter() == "select='not(mod(n,60))'"

    def test_description(self) -> None:
        """Test description generation."""
        strategy = IntervalExtractionStrategy(interval=30)
        desc = strategy.get_description()
        assert "30" in desc
        assert "frames" in desc.lower()


class TestCreateExtractionStrategy:
    """Tests for create_extraction_strategy factory."""

    def test_create_fps_strategy(self) -> None:
        """Test creating FPS strategy."""
        strategy = create_extraction_strategy(fps=1.0)
        assert isinstance(strategy, FPSExtractionStrategy)
        assert strategy.fps == 1.0

    def test_create_interval_strategy(self) -> None:
        """Test creating interval strategy."""
        strategy = create_extraction_strategy(frame_interval=30)
        assert isinstance(strategy, IntervalExtractionStrategy)
        assert strategy.interval == 30

    def test_error_when_both_params(self) -> None:
        """Test error when both fps and frame_interval provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_extraction_strategy(fps=1.0, frame_interval=30)

    def test_error_when_no_params(self) -> None:
        """Test error when neither parameter provided."""
        with pytest.raises(ValueError, match="Must specify either"):
            create_extraction_strategy()
