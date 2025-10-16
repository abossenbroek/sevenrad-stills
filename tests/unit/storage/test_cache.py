"""Tests for storage.cache module."""

import time
from pathlib import Path

import pytest
from sevenrad_stills.settings.models import CacheSettings, ClearStrategy
from sevenrad_stills.storage.cache import CacheManager
from sevenrad_stills.utils.exceptions import CacheSizeExceededError


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test cache manager initialization."""
        settings = CacheSettings(cache_dir=tmp_path / "cache")
        manager = CacheManager(settings)
        assert manager.cache_dir == tmp_path / "cache"

    def test_setup_creates_directory(self, tmp_path: Path) -> None:
        """Test setup creates cache directory."""
        cache_dir = tmp_path / "cache"
        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)

        manager.setup()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_setup_with_on_start_clears_cache(self, tmp_path: Path) -> None:
        """Test setup clears cache when strategy is ON_START."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_text("test")

        settings = CacheSettings(
            cache_dir=cache_dir,
            clear_strategy=ClearStrategy.ON_START,
        )
        manager = CacheManager(settings)
        manager.setup()

        assert not (cache_dir / "test.txt").exists()
        assert cache_dir.exists()  # Directory itself remains

    def test_clear_removes_all_files(self, tmp_path: Path) -> None:
        """Test clear removes all files and subdirectories."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "file1.txt").write_text("test1")
        (cache_dir / "file2.txt").write_text("test2")
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("test3")

        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)
        manager.clear()

        assert cache_dir.exists()
        assert len(list(cache_dir.iterdir())) == 0

    def test_clear_on_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test clear handles nonexistent directory gracefully."""
        cache_dir = tmp_path / "nonexistent"
        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)

        # Should not raise error
        manager.clear()

    def test_get_size_empty_cache(self, tmp_path: Path) -> None:
        """Test get_size returns 0 for empty cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)

        assert manager.get_size() == 0.0

    def test_get_size_with_files(self, tmp_path: Path) -> None:
        """Test get_size calculates total size."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create files with known size
        (cache_dir / "file1.txt").write_bytes(b"x" * 1024)  # 1 KB
        (cache_dir / "file2.txt").write_bytes(b"x" * 2048)  # 2 KB

        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)

        size_gb = manager.get_size()
        # Should be approximately 3KB / (1024^3)
        assert size_gb > 0
        assert size_gb < 0.001  # Less than 1MB

    def test_check_size_limit_within_limit(self, tmp_path: Path) -> None:
        """Test check_size_limit passes when under limit."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "small.txt").write_text("small file")

        settings = CacheSettings(cache_dir=cache_dir, max_size_gb=10.0)
        manager = CacheManager(settings)

        # Should not raise
        manager.check_size_limit()

    def test_check_size_limit_exceeds(self, tmp_path: Path) -> None:
        """Test check_size_limit raises when over limit."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a 1MB file
        (cache_dir / "large.txt").write_bytes(b"x" * (1024 * 1024))

        # Set very low limit
        settings = CacheSettings(cache_dir=cache_dir, max_size_gb=0.0001)
        manager = CacheManager(settings)

        with pytest.raises(CacheSizeExceededError, match="exceeds limit"):
            manager.check_size_limit()

    def test_cleanup_old_files(self, tmp_path: Path) -> None:
        """Test cleanup removes old files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create an old file by modifying its mtime
        old_file = cache_dir / "old.txt"
        old_file.write_text("old")
        old_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        Path(old_file).touch()
        import os

        os.utime(old_file, (old_time, old_time))

        # Create a recent file
        recent_file = cache_dir / "recent.txt"
        recent_file.write_text("recent")

        settings = CacheSettings(cache_dir=cache_dir, retention_days=7)
        manager = CacheManager(settings)

        removed = manager.cleanup_old_files()

        assert removed == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_get_temp_dir(self, tmp_path: Path) -> None:
        """Test get_temp_dir creates temp subdirectory."""
        cache_dir = tmp_path / "cache"
        settings = CacheSettings(cache_dir=cache_dir)
        manager = CacheManager(settings)
        manager.setup()

        temp_dir = manager.get_temp_dir()

        assert temp_dir == cache_dir / "temp"
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_cleanup_with_on_exit_strategy(self, tmp_path: Path) -> None:
        """Test cleanup clears cache with ON_EXIT strategy."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_text("test")

        settings = CacheSettings(
            cache_dir=cache_dir,
            clear_strategy=ClearStrategy.ON_EXIT,
        )
        manager = CacheManager(settings)
        manager.cleanup()

        assert len(list(cache_dir.iterdir())) == 0

    def test_cleanup_with_never_strategy(self, tmp_path: Path) -> None:
        """Test cleanup with NEVER strategy only removes old files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        recent_file = cache_dir / "recent.txt"
        recent_file.write_text("recent")

        settings = CacheSettings(
            cache_dir=cache_dir,
            clear_strategy=ClearStrategy.NEVER,
        )
        manager = CacheManager(settings)
        manager.cleanup()

        # Recent file should still exist
        assert recent_file.exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test cache manager as context manager."""
        cache_dir = tmp_path / "cache"
        settings = CacheSettings(
            cache_dir=cache_dir,
            clear_strategy=ClearStrategy.ON_EXIT,
        )

        with CacheManager(settings) as manager:
            assert cache_dir.exists()
            (cache_dir / "test.txt").write_text("test")

        # After exit, cache should be cleared
        assert len(list(cache_dir.iterdir())) == 0
