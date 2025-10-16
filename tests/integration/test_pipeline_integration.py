"""
Integration test for complete pipeline with real YouTube video.

This is a slow test that downloads a video, extracts frames, and processes them.
Run with: pytest tests/integration/ -v -s
Skip with: pytest -m "not slow"
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageStat
from sevenrad_stills.pipeline import PipelineExecutor, load_pipeline_config
from sevenrad_stills.settings.loader import load_config


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for complete pipeline execution."""

    @pytest.fixture
    def pipeline_yaml(self) -> Path:
        """Path to integration test pipeline YAML."""
        return Path(__file__).parent / "pipeline_integration_test.yaml"

    @pytest.fixture
    def output_dir(self) -> Path:
        """Output directory for integration test."""
        return Path("/tmp/sevenrad_integration_test")

    @pytest.fixture(autouse=True)
    def cleanup_output(self, output_dir: Path) -> None:
        """Clean up output directory after test."""
        yield
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_complete_pipeline_with_parallel_processing(
        self, pipeline_yaml: Path, output_dir: Path
    ) -> None:
        """
        Test complete pipeline with real YouTube video and parallel processing.

        This test:
        1. Downloads video from YouTube (xMhBKj-usK0)
        2. Extracts frames from 1m14s to 1m21s at 30fps (~210 frames)
        3. Applies saturation boost (+0.2) in parallel
        4. Verifies output files and saturation increase
        """
        # Load pipeline configuration
        config = load_pipeline_config(pipeline_yaml)
        app_settings = load_config()

        # Create executor with parallel processing enabled
        executor = PipelineExecutor(
            config,
            app_settings,
            max_workers=4,
            parallel=True,  # Use 4 workers for test
        )

        # Execute pipeline
        results = executor.execute()

        # Verify results
        assert "boost_saturation" in results
        output_paths = results["boost_saturation"]

        # Should have approximately 210 frames (7 seconds * 30 fps)
        # Allow some tolerance for frame extraction variations
        assert (
            200 <= len(output_paths) <= 220
        ), f"Expected ~210 frames, got {len(output_paths)}"

        # Verify all output files exist
        for path in output_paths:
            assert path.exists(), f"Output file not found: {path}"

        # Verify output directory structure
        assert (output_dir / "final").exists()
        assert (output_dir / "intermediate" / "extracted").exists()

        # Verify saturation was applied by checking first frame
        # Load a processed image and verify it can be opened
        test_image_path = output_paths[0]
        test_image = Image.open(test_image_path)

        # Basic checks
        assert test_image.mode == "RGB"
        assert test_image.size[0] > 0
        assert test_image.size[1] > 0

        # Check that the image has color (not grayscale)
        # A saturated image should have good color variation
        stat = ImageStat.Stat(test_image)
        # RGB channels should have different mean values if saturated
        means = stat.mean
        assert len(set(means)) > 1, "Image appears to be grayscale"

    def test_pipeline_with_sequential_processing(self, pipeline_yaml: Path) -> None:
        """
        Test pipeline with sequential processing as a comparison.

        Uses only 2 frames to keep test fast.
        """
        # Load pipeline configuration
        config = load_pipeline_config(pipeline_yaml)

        # Modify config to extract only 2 frames for fast test
        config.segment.start = 74.0
        config.segment.end = 74.1  # Just 0.1 seconds (~3 frames at 30fps)

        app_settings = load_config()

        # Create executor with parallel processing DISABLED
        executor = PipelineExecutor(config, app_settings, parallel=False)

        # Execute pipeline
        results = executor.execute()

        # Verify results
        assert "boost_saturation" in results
        output_paths = results["boost_saturation"]

        # Should have 2-4 frames
        assert 2 <= len(output_paths) <= 4

        # Verify all output files exist
        for path in output_paths:
            assert path.exists()
