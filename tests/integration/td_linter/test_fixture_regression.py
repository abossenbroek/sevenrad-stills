"""Regression tests for TD-verified fixture projects.

These fixtures have been manually verified to open successfully in TouchDesigner.
If the linter reports errors on these files, it's a false positive in our linter.

Run these tests to ensure linter changes don't introduce false positives:
    uv run pytest tests/integration/td_linter/test_fixture_regression.py -v
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from td_linter.cli import app

runner = CliRunner()

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent.parent / "docs/touchdesigner/fixtures/projects"
)

# TD-verified fixtures: These open successfully in TouchDesigner
# Add new fixtures here after verifying they open in TD
TD_VERIFIED_FIXTURES = [
    # Format: (filename, author/source)
    # Marco Kornke tutorials
    ("Circle_Visual Marco Kornke.toe", "Marco Kornke - TOPs particle systems tutorial"),
    ("MakingSimpleParticleSystemsWithTOPS Marco Kornke.toe", "Marco Kornke - TOPs particle systems tutorial"),
    # VHS tutorial
    ("VHS#tutorial.toe", "VHS effect tutorial"),
    # Introduction to TouchDesigner book - Interactive Immersive HQ
    ("01_Moving_particles_with_textures.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("02_source_geometry.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("12-6_example_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("12-7-4 adding forces.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("12-7-5 random velocity.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("3D Waveform.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Audio Responsive Geometry.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_2D_add.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_2D_buffers.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_2D_composite.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_2D_multi.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_2D_transform.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D_LFO.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D_red.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D_scaled.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D_texture.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Basic_3D_transform.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Camera_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Camera_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Color Picker.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("common_chops.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Cooking_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Cooking_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Instancing.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Movie Playlist.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Null_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Null_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("OSC Control.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("OSC Slave A.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("OSC Slave B.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Paths.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Perform_mode.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Phong.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Picture Viewer.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Point_Sprite.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Preloading.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Raster.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_10.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_3.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_4.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_5.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_6.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_7.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_8.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Rendering_9.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Resolution_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Resolution_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Sample_rates_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Sample_rates_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Scripting_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Scripting_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Scripting_3.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Select_1.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Select_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Simple_blend.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Textport_2.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("UI Example.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("UI.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Video Switcher.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Window.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
    ("Yahoo Weather API.toe", "Interactive Immersive HQ - Introduction to TouchDesigner"),
]


@pytest.mark.slow
@pytest.mark.integration
class TestTDVerifiedFixtures:
    """Test that TD-verified fixtures pass linting without errors.

    Ground truth: If TouchDesigner opens a file successfully,
    any linter errors are false positives in our linter.
    """

    @pytest.mark.parametrize(
        "fixture_name,source",
        TD_VERIFIED_FIXTURES,
        ids=[f[0] for f in TD_VERIFIED_FIXTURES],
    )
    def test_fixture_passes_lint(self, fixture_name: str, source: str) -> None:
        """TD-verified fixture should pass linting with no violations."""
        toe_file = FIXTURE_DIR / fixture_name
        if not toe_file.exists():
            pytest.skip(f"Fixture not found: {fixture_name}")

        result = runner.invoke(app, ["lint", str(toe_file)])

        assert result.exit_code == 0, (
            f"Linter reported errors on TD-verified fixture '{fixture_name}' "
            f"(source: {source}).\n"
            f"This indicates a FALSE POSITIVE in the linter.\n"
            f"Output:\n{result.output}"
        )

    @pytest.mark.parametrize(
        "fixture_name,source",
        TD_VERIFIED_FIXTURES,
        ids=[f[0] for f in TD_VERIFIED_FIXTURES],
    )
    def test_fixture_no_parse_errors(self, fixture_name: str, source: str) -> None:
        """TD-verified fixture should have no parse errors."""
        toe_file = FIXTURE_DIR / fixture_name
        if not toe_file.exists():
            pytest.skip(f"Fixture not found: {fixture_name}")

        result = runner.invoke(app, ["lint", str(toe_file), "--format", "json"])

        # Even if exit code is non-zero, check specifically for parse errors
        if "parse-error" in result.output.lower():
            pytest.fail(
                f"Parse errors on TD-verified fixture '{fixture_name}' "
                f"(source: {source}).\n"
                f"This indicates a grammar issue in the linter.\n"
                f"Output:\n{result.output}"
            )


def test_all_verified_fixtures_exist() -> None:
    """Verify all listed fixtures actually exist."""
    missing = []
    for fixture_name, _ in TD_VERIFIED_FIXTURES:
        toe_file = FIXTURE_DIR / fixture_name
        if not toe_file.exists():
            missing.append(fixture_name)

    if missing:
        pytest.fail(f"Missing TD-verified fixtures: {missing}")
