"""Custom exceptions for sevenrad-stills."""


class SevenradError(Exception):
    """Base exception for all sevenrad-stills errors."""


class ConfigError(SevenradError):
    """Configuration-related errors."""


class CacheError(SevenradError):
    """Cache management errors."""


class DownloadError(SevenradError):
    """Video download errors."""


class ExtractionError(SevenradError):
    """Frame extraction errors."""


class PipelineError(SevenradError):
    """Pipeline execution errors."""


class VideoNotFoundError(DownloadError):
    """Video not found or unavailable."""


class NetworkError(DownloadError):
    """Network-related download errors."""


class FFmpegError(ExtractionError):
    """FFmpeg execution errors."""


class InvalidConfigError(ConfigError):
    """Invalid configuration."""


class CacheSizeExceededError(CacheError):
    """Cache size limit exceeded."""
