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


class GPUError(SevenradError):
    """Base exception for GPU-related errors."""


class GPUNotAvailableError(GPUError):
    """Raised when GPU is required but not available."""


class GPUMemoryError(GPUError):
    """Raised when GPU memory allocation fails."""


class GPUOperationError(GPUError):
    """Raised when a GPU operation fails during execution."""


class TaichiInitializationError(GPUError):
    """Raised when Taichi fails to initialize."""
