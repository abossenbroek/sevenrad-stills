"""Unit tests for TouchDesigner tool discovery and integration."""

import os
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from td_linter.td_tools import (
    TouchDesignerNotFoundError,
    ToeCollapseError,
    ToeExpandError,
    _get_common_td_locations,
    _has_td_tools,
    collapse_toe,
    expand_toe,
    find_touchdesigner,
    get_toecollapse,
    get_toeexpand,
)


class TestGetCommonTDLocations:
    """Tests for _get_common_td_locations."""

    def test_returns_list(self) -> None:
        """Should return a list of paths."""
        locations = _get_common_td_locations()
        assert isinstance(locations, list)
        assert all(isinstance(p, Path) for p in locations)

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_checks_applications(self) -> None:
        """On macOS, should check /Applications for TD apps."""
        locations = _get_common_td_locations()
        # If TD is installed, we should find it
        # Otherwise, just verify the function runs without error
        for loc in locations:
            assert "TouchDesigner" in str(loc) or loc.exists()


class TestHasTDTools:
    """Tests for _has_td_tools."""

    def test_nonexistent_directory(self) -> None:
        """Should return False for nonexistent directory."""
        assert _has_td_tools(Path("/nonexistent/path")) is False

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Should return False for empty directory."""
        assert _has_td_tools(tmp_path) is False

    def test_directory_with_toeexpand_only(self, tmp_path: Path) -> None:
        """Should return False if only toeexpand exists."""
        (tmp_path / "toeexpand").touch()
        assert _has_td_tools(tmp_path) is False

    def test_directory_with_both_tools(self, tmp_path: Path) -> None:
        """Should return True if both tools exist."""
        (tmp_path / "toeexpand").touch()
        (tmp_path / "toecollapse").touch()
        assert _has_td_tools(tmp_path) is True

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_exe_extensions(self, tmp_path: Path) -> None:
        """On Windows, should check for .exe extensions."""
        (tmp_path / "toeexpand.exe").touch()
        (tmp_path / "toecollapse.exe").touch()
        assert _has_td_tools(tmp_path) is True


class TestFindTouchdesigner:
    """Tests for find_touchdesigner."""

    def test_explicit_path(self, tmp_path: Path) -> None:
        """Should use explicit path if provided and valid."""
        (tmp_path / "toeexpand").touch()
        (tmp_path / "toecollapse").touch()
        result = find_touchdesigner(td_path=tmp_path)
        assert result == tmp_path

    def test_explicit_path_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fall back to other methods if explicit path is invalid."""
        # When explicit path is invalid, it should try other methods
        # If TD is installed, it may still find it via common locations
        # This test verifies the explicit path isn't used when invalid
        monkeypatch.delenv("TOUCHDESIGNER_PATH", raising=False)
        with patch("td_linter.td_tools._get_common_td_locations", return_value=[]):
            with patch("shutil.which", return_value=None):
                result = find_touchdesigner(td_path=tmp_path)
                assert result is None

    def test_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should check TOUCHDESIGNER_PATH env var."""
        (tmp_path / "toeexpand").touch()
        (tmp_path / "toecollapse").touch()
        monkeypatch.setenv("TOUCHDESIGNER_PATH", str(tmp_path))
        result = find_touchdesigner()
        assert result == tmp_path

    def test_env_var_app_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should handle app bundle path in env var (resolve to MacOS dir)."""
        macos_dir = tmp_path / "Contents" / "MacOS"
        macos_dir.mkdir(parents=True)
        (macos_dir / "toeexpand").touch()
        (macos_dir / "toecollapse").touch()
        monkeypatch.setenv("TOUCHDESIGNER_PATH", str(tmp_path))
        result = find_touchdesigner()
        assert result == macos_dir

    def test_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fall back to PATH lookup."""
        toeexpand = tmp_path / "toeexpand"
        toeexpand.touch()
        (tmp_path / "toecollapse").touch()

        # Clear env var and mock common locations to force PATH lookup
        monkeypatch.delenv("TOUCHDESIGNER_PATH", raising=False)
        with patch("td_linter.td_tools._get_common_td_locations", return_value=[]):
            with patch("shutil.which", return_value=str(toeexpand)):
                result = find_touchdesigner()
                assert result == tmp_path


class TestGetToeexpand:
    """Tests for get_toeexpand."""

    def test_returns_path_when_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return path to toeexpand when found."""
        (tmp_path / "toeexpand").touch()
        (tmp_path / "toecollapse").touch()
        monkeypatch.setenv("TOUCHDESIGNER_PATH", str(tmp_path))
        result = get_toeexpand()
        assert result == tmp_path / "toeexpand"

    def test_returns_none_when_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return None when TD not found."""
        monkeypatch.delenv("TOUCHDESIGNER_PATH", raising=False)
        with patch("td_linter.td_tools._get_common_td_locations", return_value=[]):
            with patch("shutil.which", return_value=None):
                result = get_toeexpand()
                assert result is None


class TestGetToecollapse:
    """Tests for get_toecollapse."""

    def test_returns_path_when_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return path to toecollapse when found."""
        (tmp_path / "toeexpand").touch()
        (tmp_path / "toecollapse").touch()
        monkeypatch.setenv("TOUCHDESIGNER_PATH", str(tmp_path))
        result = get_toecollapse()
        assert result == tmp_path / "toecollapse"


class TestExpandToe:
    """Tests for expand_toe."""

    def test_file_not_found(self) -> None:
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            expand_toe(Path("/nonexistent/file.toe"))

    def test_invalid_extension(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-.toe file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        with pytest.raises(ValueError, match="Expected .toe file"):
            expand_toe(test_file)

    def test_td_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise TouchDesignerNotFoundError when TD not installed."""
        test_file = tmp_path / "test.toe"
        test_file.touch()

        monkeypatch.delenv("TOUCHDESIGNER_PATH", raising=False)
        with patch("td_linter.td_tools._get_common_td_locations", return_value=[]):
            with patch("shutil.which", return_value=None):
                with pytest.raises(TouchDesignerNotFoundError):
                    expand_toe(test_file)

    def test_successful_expand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should expand .toe file and return .toe.dir path."""
        # Create test .toe file
        test_file = tmp_path / "test.toe"
        test_file.touch()

        # Create fake TD tools directory
        td_dir = tmp_path / "td"
        td_dir.mkdir()
        (td_dir / "toeexpand").touch()
        (td_dir / "toecollapse").touch()
        monkeypatch.setenv("TOUCHDESIGNER_PATH", str(td_dir))

        # Mock subprocess.run to simulate toeexpand creating the directory
        def mock_run(*args, **kwargs):
            # Simulate toeexpand creating the .toe.dir
            toe_dir = tmp_path / "test.toe.dir"
            toe_dir.mkdir()
            return MagicMock(returncode=1, stdout="", stderr="")

        with patch("subprocess.run", mock_run):
            result = expand_toe(test_file)
            assert result == tmp_path / "test.toe.dir"
            assert result.exists()


class TestCollapseToe:
    """Tests for collapse_toe."""

    def test_directory_not_found(self) -> None:
        """Should raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            collapse_toe(Path("/nonexistent/dir.toe.dir"))

    def test_invalid_extension(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-.toe.dir directory."""
        with pytest.raises(ValueError, match="Expected .toe.dir"):
            collapse_toe(tmp_path)

    def test_td_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise TouchDesignerNotFoundError when TD not installed."""
        test_dir = tmp_path / "test.toe.dir"
        test_dir.mkdir()

        monkeypatch.delenv("TOUCHDESIGNER_PATH", raising=False)
        with patch("td_linter.td_tools._get_common_td_locations", return_value=[]):
            with patch("shutil.which", return_value=None):
                with pytest.raises(TouchDesignerNotFoundError):
                    collapse_toe(test_dir)
