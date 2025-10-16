"""
CLI commands for sevenrad-stills.

Command-line interface using Click.
"""

from pathlib import Path
from typing import Any

import click

from sevenrad_stills.pipeline.processor import VideoProcessor
from sevenrad_stills.settings.loader import load_config
from sevenrad_stills.settings.models import ClearStrategy
from sevenrad_stills.storage.cache import CacheManager
from sevenrad_stills.utils.exceptions import SevenradError
from sevenrad_stills.utils.logging import setup_logging


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """sevenrad-stills: Extract and transform video stills for poetic interpretation."""


@cli.command()
@click.argument("url")
@click.option(
    "--fps",
    type=float,
    help="Frames per second to extract",
)
@click.option(
    "--frame-interval",
    type=int,
    help="Extract every Nth frame",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),  # type: ignore[type-var]
    help="Output directory for extracted frames",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),  # type: ignore[type-var]
    help="Path to configuration file",
)
@click.option(
    "--clear-cache",
    type=click.Choice(["on_start", "on_exit", "manual", "never"]),
    help="Cache clearing strategy",
)
def extract(
    url: str,
    fps: float | None,
    frame_interval: int | None,
    output_dir: Path | None,
    config: Path | None,
    clear_cache: str | None,
) -> None:
    """
    Extract frames from a video URL.

    URL: YouTube or other supported video URL
    """
    try:
        # Load configuration
        config_overrides: dict[str, Any] = {}
        if clear_cache:
            config_overrides["cache"] = {"clear_strategy": clear_cache}

        settings = load_config(config, config_overrides)

        # Setup logging
        logger = setup_logging(settings.logging)

        # Validate extraction parameters
        if fps and frame_interval:
            click.echo(
                "Error: Cannot specify both --fps and --frame-interval", err=True
            )
            raise click.Abort

        # Create processor
        processor = VideoProcessor(settings)

        # Process video
        if fps or frame_interval or output_dir:
            video_info, frames = processor.process_with_custom_settings(
                url=url,
                fps=fps,
                frame_interval=frame_interval,
                output_dir=output_dir,
            )
        else:
            video_info, frames = processor.process(url)

        # Display results
        click.echo(f"\nSuccessfully processed: {video_info.title}")
        click.echo(f"Duration: {video_info.duration_str}")
        if video_info.resolution:
            click.echo(f"Resolution: {video_info.resolution}")
        click.echo(f"Extracted: {len(frames)} frames")
        click.echo(f"Output: {settings.extraction.output_dir}")

    except SevenradError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort from e
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        raise click.Abort from e


@cli.group()
def cache() -> None:
    """Manage cache directory."""


@cache.command("clear")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),  # type: ignore[type-var]
    help="Path to configuration file",
)
def cache_clear(config: Path | None) -> None:
    """Clear the cache directory."""
    try:
        settings = load_config(config)
        manager = CacheManager(settings.cache)

        size_before = manager.get_size()
        manager.clear()
        click.echo(f"Cache cleared ({size_before:.2f}GB freed)")

    except Exception as e:
        click.echo(f"Error clearing cache: {e}", err=True)
        raise click.Abort from e


@cache.command("info")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),  # type: ignore[type-var]
    help="Path to configuration file",
)
def cache_info(config: Path | None) -> None:
    """Display cache information."""
    try:
        settings = load_config(config)
        manager = CacheManager(settings.cache)

        size = manager.get_size()
        click.echo(f"Cache directory: {settings.cache.cache_dir}")
        click.echo(f"Current size: {size:.2f}GB")
        click.echo(f"Size limit: {settings.cache.max_size_gb}GB")
        click.echo(f"Retention: {settings.cache.retention_days} days")
        click.echo(f"Clear strategy: {settings.cache.clear_strategy.value}")

    except Exception as e:
        click.echo(f"Error getting cache info: {e}", err=True)
        raise click.Abort from e


@cli.group()
def config_cmd() -> None:
    """Manage configuration."""


@config_cmd.command("show")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),  # type: ignore[type-var]
    help="Path to configuration file",
)
def config_show(config: Path | None) -> None:
    """Display current configuration."""
    try:
        settings = load_config(config)

        click.echo("\n=== Cache Settings ===")
        click.echo(f"Directory: {settings.cache.cache_dir}")
        click.echo(f"Clear strategy: {settings.cache.clear_strategy.value}")
        click.echo(f"Max size: {settings.cache.max_size_gb}GB")
        click.echo(f"Retention: {settings.cache.retention_days} days")

        click.echo("\n=== Download Settings ===")
        click.echo(f"Format: {settings.download.format}")
        click.echo(f"Quality: {settings.download.quality}")
        click.echo(f"Output directory: {settings.download.output_dir}")

        click.echo("\n=== Extraction Settings ===")
        if settings.extraction.fps:
            click.echo(f"Mode: FPS ({settings.extraction.fps} fps)")
        else:
            click.echo(
                f"Mode: Interval (every {settings.extraction.frame_interval} frames)"
            )
        click.echo(f"Output format: {settings.extraction.output_format}")
        click.echo(f"Output directory: {settings.extraction.output_dir}")
        click.echo(f"Naming pattern: {settings.extraction.naming_pattern}")

        click.echo("\n=== Logging Settings ===")
        click.echo(f"Level: {settings.logging.level}")
        click.echo(f"Show progress: {settings.logging.show_progress}")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        raise click.Abort from e


if __name__ == "__main__":
    cli()
