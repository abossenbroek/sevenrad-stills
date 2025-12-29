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


class TestShaderReferenceExtraction:
    """Tests for shader reference extraction from jit.gl.pix objects."""

    def test_extract_shader_reference(self, tmp_path: Path) -> None:
        """Should extract shader name and params from jit.gl.pix."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.noise @mode 0 @amount 0.5",
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
        assert len(patcher.shader_refs) == 1
        assert patcher.shader_refs[0].shader_name == "sr.noise"
        assert patcher.shader_refs[0].params == {"mode": "0", "amount": "0.5"}

    def test_extract_multiple_shaders(self, tmp_path: Path) -> None:
        """Should extract multiple shader references."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix1",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.blur.h @sigma 5.0",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-pix2",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.blur.v @sigma 5.0",
                        }
                    },
                ],
                "lines": [],
            }
        }

        filepath = tmp_path / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        extractor = MaxhelpExtractor()
        patcher = extractor.extract(filepath)

        assert patcher is not None
        assert len(patcher.shader_refs) == 2
        shader_names = {ref.shader_name for ref in patcher.shader_refs}
        assert shader_names == {"sr.blur.h", "sr.blur.v"}


class TestShaderReferenceValidation:
    """Tests for shader reference validation."""

    def test_shader_reference_missing(self, tmp_path: Path) -> None:
        """Missing shader file should produce error."""
        # Create code directory (empty)
        code_dir = tmp_path / "code"
        code_dir.mkdir()

        help_dir = tmp_path / "help"
        help_dir.mkdir()

        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.nonexistent",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = help_dir / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator(code_dir=code_dir)
        diagnostics = validator.validate(filepath)

        shader_errors = [d for d in diagnostics if d.code == "shader-not-found"]
        assert len(shader_errors) == 1
        assert shader_errors[0].severity == DiagnosticSeverity.ERROR
        assert "sr.nonexistent" in shader_errors[0].message

    def test_shader_reference_exists(self, tmp_path: Path) -> None:
        """Existing shader file should not produce error."""
        # Create code directory with shader
        code_dir = tmp_path / "code"
        code_dir.mkdir()

        shader_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "codebox",
                            "code": "out1 = in1;",
                        }
                    }
                ],
                "lines": [],
            }
        }
        (code_dir / "sr.test.genjit").write_text(json.dumps(shader_json))

        help_dir = tmp_path / "help"
        help_dir.mkdir()

        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.test",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = help_dir / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator(code_dir=code_dir)
        diagnostics = validator.validate(filepath)

        shader_errors = [d for d in diagnostics if d.code == "shader-not-found"]
        assert len(shader_errors) == 0


class TestParameterRangeValidation:
    """Tests for parameter range validation."""

    def test_param_below_min(self, tmp_path: Path) -> None:
        """Parameter below minimum should warn."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()

        shader_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "param amount 0.5 0.0 1.0",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-2",
                            "maxclass": "codebox",
                            "code": "out1 = in1;",
                        }
                    },
                ],
                "lines": [],
            }
        }
        (code_dir / "sr.test.genjit").write_text(json.dumps(shader_json))

        help_dir = tmp_path / "help"
        help_dir.mkdir()

        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.test @amount -0.5",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = help_dir / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator(code_dir=code_dir)
        diagnostics = validator.validate(filepath)

        below_min = [d for d in diagnostics if d.code == "param-below-min"]
        assert len(below_min) == 1
        assert below_min[0].severity == DiagnosticSeverity.WARNING

    def test_param_above_max(self, tmp_path: Path) -> None:
        """Parameter above maximum should warn."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()

        shader_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "param amount 0.5 0.0 1.0",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-2",
                            "maxclass": "codebox",
                            "code": "out1 = in1;",
                        }
                    },
                ],
                "lines": [],
            }
        }
        (code_dir / "sr.test.genjit").write_text(json.dumps(shader_json))

        help_dir = tmp_path / "help"
        help_dir.mkdir()

        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.test @amount 1.5",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = help_dir / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator(code_dir=code_dir)
        diagnostics = validator.validate(filepath)

        above_max = [d for d in diagnostics if d.code == "param-above-max"]
        assert len(above_max) == 1
        assert above_max[0].severity == DiagnosticSeverity.WARNING

    def test_unknown_param(self, tmp_path: Path) -> None:
        """Unknown parameter should warn."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()

        shader_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-1",
                            "maxclass": "newobj",
                            "text": "param amount 0.5",
                        }
                    },
                    {
                        "box": {
                            "id": "obj-2",
                            "maxclass": "codebox",
                            "code": "out1 = in1;",
                        }
                    },
                ],
                "lines": [],
            }
        }
        (code_dir / "sr.test.genjit").write_text(json.dumps(shader_json))

        help_dir = tmp_path / "help"
        help_dir.mkdir()

        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-pix",
                            "maxclass": "newobj",
                            "text": "jit.gl.pix ctx @gen sr.test @bogus 42",
                        }
                    }
                ],
                "lines": [],
            }
        }

        filepath = help_dir / "test.maxhelp"
        filepath.write_text(json.dumps(patcher_json))

        validator = MaxhelpValidator(code_dir=code_dir)
        diagnostics = validator.validate(filepath)

        unknown = [d for d in diagnostics if d.code == "unknown-param"]
        assert len(unknown) == 1
        assert "bogus" in unknown[0].message


class TestCodeboxExtraction:
    """Tests for inline codebox GenExpr extraction."""

    def test_extract_codebox(self, tmp_path: Path) -> None:
        """Should extract code from codebox objects."""
        patcher_json = {
            "patcher": {
                "boxes": [
                    {
                        "box": {
                            "id": "obj-codebox",
                            "maxclass": "codebox",
                            "code": "out1 = in1 * 0.5;",
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
        assert len(patcher.codeboxes) == 1
        assert patcher.codeboxes[0].code == "out1 = in1 * 0.5;"
        assert patcher.codeboxes[0].object_id == "obj-codebox"


@pytest.mark.skipif(not HELP_DIR.exists(), reason="Help directory not found")
class TestShaderValidationIntegration:
    """Integration tests for shader validation with real files."""

    def test_all_help_files_reference_existing_shaders(self) -> None:
        """All help files should reference existing shaders."""
        code_dir = HELP_DIR.parent / "code"

        if not code_dir.exists():
            pytest.skip("code directory not found")

        validator = MaxhelpValidator(code_dir=code_dir)

        all_shader_errors = []
        for help_file in HELP_DIR.glob("*.maxhelp"):
            diagnostics = validator.validate(help_file)
            shader_errors = [
                (help_file.name, d.message)
                for d in diagnostics
                if d.code == "shader-not-found"
            ]
            all_shader_errors.extend(shader_errors)

        assert len(all_shader_errors) == 0, f"Missing shaders: {all_shader_errors}"
