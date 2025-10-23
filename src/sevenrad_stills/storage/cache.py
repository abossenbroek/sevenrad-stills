"""
Cache management for sevenrad-stills.

Manages the cache directory for downloaded videos and temporary files.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sevenrad_stills.settings.models import CacheSettings, ClearStrategy
from sevenrad_stills.storage.paths import ensure_directory
from sevenrad_stills.utils.exceptions import CacheError, CacheSizeExceededError


class CacheManager:
    """Manages cache directory lifecycle and cleanup."""

    def __init__(self, settings: CacheSettings) -> None:
        """
        Initialize cache manager.

        Args:
            settings: Cache configuration settings

        """
        self.settings = settings
        self.cache_dir = settings.cache_dir

    def setup(self) -> None:
        """
        Set up cache directory.

        Creates cache directory if it doesn't exist and performs cleanup
        based on clear strategy.

        Raises:
            CacheError: If cache directory cannot be created

        """
        try:
            ensure_directory(self.cache_dir)
        except OSError as e:
            msg = f"Failed to create cache directory {self.cache_dir}: {e}"
            raise CacheError(msg) from e

        # Clear cache if strategy is ON_START
        if self.settings.clear_strategy == ClearStrategy.ON_START:
            self.clear()

    def clear(self) -> None:
        """
        Clear all cached files.

        Removes all files in the cache directory while keeping the directory itself.
        """
        if not self.cache_dir.exists():
            return

        for item in self.cache_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except OSError:
                # Continue clearing even if some files fail
                pass

    def get_size(self) -> float:
        """
        Get current cache size in gigabytes.

        Returns:
            Cache size in GB

        """
        if not self.cache_dir.exists():
            return 0.0

        total_size = 0
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size / (1024**3)  # Convert to GB

    def check_size_limit(self) -> None:
        """
        Check if cache size exceeds the limit.

        Raises:
            CacheSizeExceededError: If cache size exceeds configured limit

        """
        current_size = self.get_size()
        if current_size > self.settings.max_size_gb:
            msg = (
                f"Cache size {current_size:.2f}GB exceeds limit "
                f"{self.settings.max_size_gb}GB"
            )
            raise CacheSizeExceededError(msg)

    def cleanup_old_files(self) -> int:
        """
        Remove files older than retention period.

        Returns:
            Number of files removed

        """
        if not self.cache_dir.exists():
            return 0

        cutoff_time = datetime.now() - timedelta(days=self.settings.retention_days)
        removed_count = 0

        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff_time:
                    try:
                        item.unlink()
                        removed_count += 1
                    except OSError:
                        pass

        return removed_count

    def get_temp_dir(self) -> Path:
        """
        Get temporary directory within cache.

        Returns:
            Path to temp directory

        """
        temp_dir = self.cache_dir / "temp"
        ensure_directory(temp_dir)
        return temp_dir

    def cleanup(self) -> None:
        """
        Perform cleanup based on clear strategy.

        Called at application exit.
        """
        if self.settings.clear_strategy == ClearStrategy.ON_EXIT:
            self.clear()
        else:
            # Always cleanup old files regardless of strategy
            self.cleanup_old_files()

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.cleanup()
