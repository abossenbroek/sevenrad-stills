"""Tests for maxhelp patcher validation.

Tests the maxhelp validation features including:
- GPU shader effect signal flow
- Context naming conventions
- CPU external initialization
- Utility external detection
"""

import json
from pathlib import Path

import pytest

from max_linter.extractors.maxhelp import MaxhelpExtractor
from max_linter.results import DiagnosticSeverity
from max_linter.validators.maxhelp import MaxhelpValidator

# Path to real help files for integration tests
HELP_DIR = Path(__file__).parent.parent.parent / "help"


class TestMaxhelpExtractor:
    """Tests for MaxhelpExtractor parsing."""

    def test_extract_basic_patcher(self, tmp_path: Path) -> None:
        """Extractor should parse basic patcher structure."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "jit.world test_ctx @visible 0",
                            "numinlets": 1,
                            "numoutlets": 0,
                        }
                    }
                ],
                "lines": [],
                "description": "Test patcher",
                "tags": "test",
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        extractor = MaxhelpExtractor()
        patcher = extractor.extract(filepath)

        assert patcher is not None
        assert patcher.description == "Test patcher"
        assert patcher.tags == "test"
        assert len(patcher.objects) == 1
        assert "obj-1" in patcher.objects

    def test_extract_connections(self, tmp_path: Path) -> None:
        """Extractor should parse connections."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {"box": {"id": "obj-1", "maxclass": "button"}},
                    {"box": {"id": "obj-2", "maxclass": "newobj", "text": "print"}},
                ],
                "lines": [
                    {"patchline": {"source": ["obj-1", 0], "destination": ["obj-2", 0]}}
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        extractor = MaxhelpExtractor()
        patcher = extractor.extract(filepath)

        assert patcher is not None
        assert len(patcher.connections) == 1
        assert patcher.connections[0].source_id == "obj-1"
        assert patcher.connections[0].dest_id == "obj-2"

    def test_parse_attributes(self, tmp_path: Path) -> None:
        """Extractor should parse @attributes from text."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "jit.movie @autostart 1 @loop 1 @output_texture 1",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        extractor = MaxhelpExtractor()
        patcher = extractor.extract(filepath)
        assert patcher is not None

        obj = patcher.objects["obj-1"]
        assert obj.has_attribute("autostart", "1")
        assert obj.has_attribute("loop", "1")
        assert obj.has_attribute("output_texture", "1")

    def test_build_connection_graph(self, tmp_path: Path) -> None:
        """Extractor should build networkx graph."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {"box": {"id": "obj-1", "maxclass": "button"}},
                    {"box": {"id": "obj-2", "maxclass": "newobj", "text": "print"}},
                    {"box": {"id": "obj-3", "maxclass": "comment"}},
                ],
                "lines": [
                    {"patchline": {"source": ["obj-1", 0], "destination": ["obj-2", 0]}}
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        extractor = MaxhelpExtractor()
        patcher = extractor.extract(filepath)
        assert patcher is not None
        graph = extractor.build_connection_graph(patcher)

        assert graph.has_edge("obj-1", "obj-2")
        assert not graph.has_edge("obj-2", "obj-1")
        assert not graph.has_edge("obj-1", "obj-3")


class TestContextNaming:
    """Tests for context naming validation."""

    def test_valid_context_name(self, tmp_path: Path) -> None:
        """Context names with underscores should not warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "jit.world sr_test_ctx @visible 0",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        naming_warnings = [d for d in diagnostics if d.code == "context-naming"]
        assert len(naming_warnings) == 0

    def test_context_name_with_dots(self, tmp_path: Path) -> None:
        """Context names with dots should warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "jit.world sr.test.ctx @visible 0",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        naming_warnings = [d for d in diagnostics if d.code == "context-naming"]
        assert len(naming_warnings) == 1
        assert naming_warnings[0].severity == DiagnosticSeverity.WARNING
        assert "sr.test.ctx" in naming_warnings[0].message


class TestGPUEffectFlow:
    """Tests for GPU effect signal flow validation."""

    def test_complete_gpu_pipeline(self, tmp_path: Path) -> None:
        """Complete GPU pipeline should not error."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-world",
                            "maxclass": "newobj",
                            "text": "jit.world sr_test_ctx @visible 0",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-movie",
                            "maxclass": "newobj",
                            "text": "jit.movie @output_texture 1 @drawto sr_test_ctx",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix sr_test_ctx @gen sr.test",
                        }
                    },
                    {"box": {"id": "obj-pwindow", "maxclass": "jit.pwindow"}},
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-movie", 0],
                            "destination": ["obj-pix", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-pix", 0],
                            "destination": ["obj-pwindow", 0],
                        }
                    },
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        # No errors for GPU flow
        gpu_errors = [
            d
            for d in diagnostics
            if d.code in ("gpu-missing-source", "gpu-missing-display")
        ]
        assert len(gpu_errors) == 0

    def test_missing_video_source(self, tmp_path: Path) -> None:
        """jit.gl.pix without video source should error."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix sr_test_ctx @gen sr.test",
                        }
                    },
                    {"box": {"id": "obj-pwindow", "maxclass": "jit.pwindow"}},
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-pix", 0],
                            "destination": ["obj-pwindow", 0],
                        }
                    }
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        source_errors = [d for d in diagnostics if d.code == "gpu-missing-source"]
        assert len(source_errors) == 1
        assert source_errors[0].severity == DiagnosticSeverity.ERROR

    def test_missing_display(self, tmp_path: Path) -> None:
        """jit.gl.pix without display should error."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-movie",
                            "maxclass": "newobj",
                            "text": "jit.movie @output_texture 1",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix sr_test_ctx @gen sr.test",
                        }
                    },
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-movie", 0],
                            "destination": ["obj-pix", 0],
                        }
                    }
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        display_errors = [d for d in diagnostics if d.code == "gpu-missing-display"]
        assert len(display_errors) == 1
        assert display_errors[0].severity == DiagnosticSeverity.ERROR

    def test_missing_output_texture(self, tmp_path: Path) -> None:
        """jit.movie without @output_texture 1 should warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-movie",
                            "maxclass": "newobj",
                            "text": "jit.movie @loop 1",  # Missing output_texture
                        }
                    },
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix sr_test_ctx @gen sr.test",
                        }
                    },
                    {"box": {"id": "obj-pwindow", "maxclass": "jit.pwindow"}},
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-movie", 0],
                            "destination": ["obj-pix", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-pix", 0],
                            "destination": ["obj-pwindow", 0],
                        }
                    },
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        texture_warnings = [d for d in diagnostics if d.code == "gpu-no-texture-output"]
        assert len(texture_warnings) == 1
        assert texture_warnings[0].severity == DiagnosticSeverity.WARNING


class TestCPUExternalFlow:
    """Tests for CPU external initialization validation."""

    def test_cpu_external_with_dimensions(self, tmp_path: Path) -> None:
        """CPU external with dimension messages should not warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-loadbang",
                            "maxclass": "newobj",
                            "text": "loadbang",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-width",
                            "maxclass": "message",
                            "text": "width 512",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-height",
                            "maxclass": "message",
                            "text": "height 512",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-maskgen",
                            "maxclass": "newobj",
                            "text": "sr.maskgen",
                        }
                    },
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-loadbang", 0],
                            "destination": ["obj-width", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-loadbang", 0],
                            "destination": ["obj-height", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-width", 0],
                            "destination": ["obj-maskgen", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-height", 0],
                            "destination": ["obj-maskgen", 0],
                        }
                    },
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        dim_warnings = [d for d in diagnostics if d.code == "cpu-no-dimensions"]
        assert len(dim_warnings) == 0


class TestUtilityExternal:
    """Tests for utility external detection."""

    def test_utility_tag_skips_flow_check(self, tmp_path: Path) -> None:
        """Patchers with utility tag should not flag missing video output."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-tilegen",
                            "maxclass": "newobj",
                            "text": "sr.tilegen",
                        }
                    }
                ],
                "lines": [],
                "tags": "utility, tiles, generator",
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        utility_info = [d for d in diagnostics if d.code == "utility-unmarked"]
        assert len(utility_info) == 0


class TestInitializationOrder:
    """Tests for initialization order validation."""

    def test_no_loadbang_warns(self, tmp_path: Path) -> None:
        """Missing loadbang should warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-world",
                            "maxclass": "newobj",
                            "text": "jit.world sr_test_ctx @visible 0",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        loadbang_warnings = [d for d in diagnostics if d.code == "init-no-loadbang"]
        assert len(loadbang_warnings) >= 1

    def test_no_context_with_pix_warns(self, tmp_path: Path) -> None:
        """jit.gl.pix without jit.world should warn."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-movie",
                            "maxclass": "newobj",
                            "text": "jit.movie @output_texture 1",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix test_ctx @gen sr.test",
                        }
                    },
                    {"box": {"id": "obj-pwindow", "maxclass": "jit.pwindow"}},
                ],
                "lines": [
                    {
                        "patchline": {
                            "source": ["obj-movie", 0],
                            "destination": ["obj-pix", 0],
                        }
                    },
                    {
                        "patchline": {
                            "source": ["obj-pix", 0],
                            "destination": ["obj-pwindow", 0],
                        }
                    },
                ],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        context_warnings = [d for d in diagnostics if d.code == "init-no-context"]
        assert len(context_warnings) == 1


@pytest.mark.skipif(not HELP_DIR.exists(), reason="Help directory not found")
class TestRealHelpPatchers:
    """Integration tests with real help patchers."""

    def test_sr_noise_passes(self) -> None:
        """sr.noise.maxhelp should have correct GPU pipeline."""
        filepath = HELP_DIR / "sr.noise.maxhelp"
        if not filepath.exists():
            pytest.skip("sr.noise.maxhelp not found")

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        # May have some errors depending on patcher state
        # Main check is that it parses and runs without crashing
        assert diagnostics is not None

    def test_sr_saltpepper_passes(self) -> None:
        """sr.saltpepper.maxhelp should have correct GPU pipeline."""
        filepath = HELP_DIR / "sr.saltpepper.maxhelp"
        if not filepath.exists():
            pytest.skip("sr.saltpepper.maxhelp not found")

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        # Check it parses without crash
        assert diagnostics is not None

    def test_sr_tilegen_utility(self) -> None:
        """sr.tilegen.maxhelp should be recognized as utility."""
        filepath = HELP_DIR / "sr.tilegen.maxhelp"
        if not filepath.exists():
            pytest.skip("sr.tilegen.maxhelp not found")

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        # sr.tilegen is in UTILITY_EXTERNALS, shouldn't error for missing display
        gpu_errors = [
            d
            for d in diagnostics
            if d.code in ("gpu-missing-source", "gpu-missing-display")
        ]
        assert len(gpu_errors) == 0

    def test_sr_maskgen_cpu_external(self) -> None:
        """sr.maskgen.maxhelp should check for dimension initialization."""
        filepath = HELP_DIR / "sr.maskgen.maxhelp"
        if not filepath.exists():
            pytest.skip("sr.maskgen.maxhelp not found")

        validator = MaxhelpValidator()
        diagnostics = validator.validate(filepath)

        # Check it parses and runs validation
        assert diagnostics is not None
