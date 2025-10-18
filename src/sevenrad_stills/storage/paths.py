"""
Path utilities for sevenrad-stills.

Provides utilities for path resolution, sanitization, and file naming.
"""

import re
from pathlib import Path

# Maximum filename length for most filesystems
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use

    """
    # Remove or replace characters that are problematic for filesystems
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove control characters
    sanitized = re.sub(r"[\x00-\x1f\x7f]", "", sanitized)
    # Collapse multiple underscores
    sanitized = re.sub(r"_{2,}", "_", sanitized)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    # Limit length to maximum filename length
    if len(sanitized) > MAX_FILENAME_LENGTH:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        if ext:
            max_name_len = MAX_FILENAME_LENGTH - len(ext) - 1
            sanitized = f"{name[:max_name_len]}.{ext}"
        else:
            sanitized = name[:MAX_FILENAME_LENGTH]
    return sanitized or "unnamed"


def get_incremental_path(base_path: Path, start_index: int = 0) -> Path:
    """
    Get an incremental path that doesn't exist.

    If the path exists, append an incrementing number until a non-existing
    path is found. Supports both files and directories.

    Args:
        base_path: Base path to check
        start_index: Starting index for incrementing

    Returns:
        Path that doesn't exist

    Examples:
        >>> get_incremental_path(Path("output.jpg"))
        Path("output.jpg")  # if it doesn't exist
        Path("output_001.jpg")  # if output.jpg exists

    """
    if not base_path.exists():
        return base_path

    if base_path.is_dir():
        # For directories, append number before the name
        stem = base_path.name
        parent = base_path.parent
        index = start_index
        while True:
            new_path = parent / f"{stem}_{index:03d}"
            if not new_path.exists():
                return new_path
            index += 1
    else:
        # For files, insert number before extension
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        index = start_index
        while True:
            new_path = parent / f"{stem}_{index:03d}{suffix}"
            if not new_path.exists():
                return new_path
            index += 1


def format_frame_filename(
    pattern: str,
    video_id: str,
    index: int,
    timestamp: float | None = None,
    ext: str = "jpg",
) -> str:
    """
    Format a frame filename using the given pattern.

    Args:
        pattern: Naming pattern with placeholders
        video_id: Video identifier
        index: Frame index
        timestamp: Optional timestamp in seconds
        ext: File extension

    Returns:
        Formatted filename

    Available placeholders:
        - {video_id}: Video identifier
        - {index}: Frame index (supports formatting like {index:06d})
        - {timestamp}: Timestamp in seconds (supports formatting)
        - {ext}: File extension

    Raises:
        ValueError: If the formatted filename contains path separators

    """
    # Sanitize video_id
    safe_video_id = sanitize_filename(video_id)

    # Prepare format arguments
    format_args = {
        "video_id": safe_video_id,
        "index": index,
        "ext": ext,
    }

    if timestamp is not None:
        format_args["timestamp"] = timestamp

    try:
        formatted = pattern.format(**format_args)
    except (KeyError, ValueError):
        # Fallback to simple pattern if formatting fails
        formatted = f"{safe_video_id}_frame_{index:06d}.{ext}"

    # Security: Prevent path traversal by rejecting filenames with path separators
    if "/" in formatted or "\\" in formatted or ".." in formatted:
        msg = (
            f"Invalid filename pattern: '{formatted}' contains path separators. "
            "Filenames must not contain '/', '\\\\', or '..' to prevent path traversal."
        )
        raise ValueError(msg)

    return formatted


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The directory path

    Raises:
        OSError: If directory cannot be created

    """
    path.mkdir(parents=True, exist_ok=True)
    return path
