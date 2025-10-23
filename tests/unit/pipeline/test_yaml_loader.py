"""Tests for YAML pipeline loader."""

import pytest
from sevenrad_stills.pipeline.yaml_loader import (
    PipelineLoadError,
    validate_pipeline_yaml,
)


class TestValidatePipelineYaml:
    """Tests for validate_pipeline_yaml function."""

    def test_valid_yaml(self) -> None:
        """Test loading valid pipeline YAML."""
        yaml_content = """
source:
  youtube_url: "https://youtube.com/watch?v=test"

segment:
  start: 10.0
  end: 30.0
  interval: 1.0

pipeline:
  steps:
    - name: "saturate"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.5

output:
  base_dir: "./output"
"""
        config = validate_pipeline_yaml(yaml_content)
        assert config.source.youtube_url == "https://youtube.com/watch?v=test"
        assert config.segment.start == 10.0
        assert len(config.steps) == 1

    def test_invalid_yaml_syntax_raises_error(self) -> None:
        """Test invalid YAML syntax raises error."""
        yaml_content = """
source:
  youtube_url: "test
  invalid syntax here
"""
        with pytest.raises(PipelineLoadError, match="Invalid YAML syntax"):
            validate_pipeline_yaml(yaml_content)

    def test_missing_required_field_raises_error(self) -> None:
        """Test missing required field raises error."""
        yaml_content = """
source:
  youtube_url: "https://youtube.com/watch?v=test"

# Missing segment
pipeline:
  steps:
    - name: "saturate"
      operation: "saturation"
      params: {}
"""
        with pytest.raises(PipelineLoadError, match="validation failed"):
            validate_pipeline_yaml(yaml_content)

    def test_non_dict_yaml_raises_error(self) -> None:
        """Test non-dictionary YAML raises error."""
        yaml_content = "- item1\n- item2"
        with pytest.raises(PipelineLoadError, match="must be a YAML dictionary"):
            validate_pipeline_yaml(yaml_content)
