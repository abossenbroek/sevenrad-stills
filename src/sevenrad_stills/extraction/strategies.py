"""
Frame extraction strategies for sevenrad-stills.

Defines different strategies for extracting frames from videos.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import ffmpeg

from sevenrad_stills.utils.exceptions import FFmpegError


class ExtractionStrategy(Protocol):
    """Protocol for frame extraction strategies."""

    def get_ffmpeg_filter(self) -> str:
        """
        Get FFmpeg filter string for this strategy.

        Returns:
            FFmpeg filter string

        """
        ...

    def get_description(self) -> str:
        """
        Get human-readable description of this strategy.

        Returns:
            Strategy description

        """
        ...


class FPSExtractionStrategy:
    """Extract frames at a specific FPS rate."""

    def __init__(self, fps: float) -> None:
        """
        Initialize FPS extraction strategy.

        Args:
            fps: Frames per second to extract

        """
        self.fps = fps

    def get_ffmpeg_filter(self) -> str:
        """Get FFmpeg filter for FPS-based extraction."""
        return f"fps={self.fps}"

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Extract at {self.fps} FPS"


class IntervalExtractionStrategy:
    """Extract every Nth frame."""

    def __init__(self, interval: int) -> None:
        """
        Initialize interval extraction strategy.

        Args:
            interval: Extract every Nth frame

        """
        self.interval = interval

    def get_ffmpeg_filter(self) -> str:
        """Get FFmpeg filter for interval-based extraction."""
        # select='not(mod(n,N))' selects every Nth frame
        return f"select='not(mod(n,{self.interval}))'"

    def get_description(self) -> str:
        """Get strategy description."""
        return f"Extract every {self.interval} frames"


def create_extraction_strategy(
    fps: float | None = None,
    frame_interval: int | None = None,
) -> ExtractionStrategy:
    """
    Create an extraction strategy based on parameters.

    Args:
        fps: Frames per second (mutually exclusive with frame_interval)
        frame_interval: Frame interval (mutually exclusive with fps)

    Returns:
        ExtractionStrategy instance

    Raises:
        ValueError: If both or neither parameters are provided

    """
    if fps is not None and frame_interval is not None:
        msg = "Cannot specify both fps and frame_interval"
        raise ValueError(msg)

    if fps is None and frame_interval is None:
        msg = "Must specify either fps or frame_interval"
        raise ValueError(msg)

    if fps is not None:
        return FPSExtractionStrategy(fps)

    return IntervalExtractionStrategy(frame_interval)  # type: ignore[arg-type]
