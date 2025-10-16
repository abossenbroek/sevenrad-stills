"""
YAML pipeline configuration loader.

Loads and validates YAML pipeline configuration files.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sevenrad_stills.pipeline.models import PipelineConfig
from sevenrad_stills.utils.exceptions import SevenradError
from sevenrad_stills.utils.logging import get_logger


class PipelineLoadError(SevenradError):
    """Error loading or validating pipeline configuration."""


def load_pipeline_config(config_path: Path | str) -> PipelineConfig:
    """
    Load and validate pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig instance

    Raises:
        PipelineLoadError: If file cannot be read or validation fails

    """
    logger = get_logger()
    config_path = Path(config_path)

    if not config_path.exists():
        msg = f"Pipeline configuration file not found: {config_path}"
        raise PipelineLoadError(msg)

    try:
        logger.debug("Loading pipeline configuration from: %s", config_path)
        with config_path.open("r") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            msg = "Pipeline configuration must be a YAML dictionary"
            raise PipelineLoadError(msg)

        logger.debug("Validating pipeline configuration")
        config = PipelineConfig(**raw_config)
        logger.info("Successfully loaded pipeline with %d steps", len(config.steps))
        return config

    except yaml.YAMLError as e:
        msg = f"Invalid YAML syntax: {e}"
        raise PipelineLoadError(msg) from e
    except ValidationError as e:
        msg = f"Pipeline configuration validation failed: {e}"
        raise PipelineLoadError(msg) from e
    except Exception as e:
        msg = f"Unexpected error loading pipeline configuration: {e}"
        raise PipelineLoadError(msg) from e


def validate_pipeline_yaml(yaml_content: str) -> PipelineConfig:
    """
    Validate pipeline YAML content without loading from file.

    Useful for testing and validation.

    Args:
        yaml_content: YAML content as string

    Returns:
        Validated PipelineConfig instance

    Raises:
        PipelineLoadError: If validation fails

    """
    try:
        raw_config: dict[str, Any] = yaml.safe_load(yaml_content)
        if not isinstance(raw_config, dict):
            msg = "Pipeline configuration must be a YAML dictionary"
            raise PipelineLoadError(msg)
        return PipelineConfig(**raw_config)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML syntax: {e}"
        raise PipelineLoadError(msg) from e
    except ValidationError as e:
        msg = f"Pipeline configuration validation failed: {e}"
        raise PipelineLoadError(msg) from e
