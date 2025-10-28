# Backend API Contract

**Version:** 1.0.0
**Last Updated:** 2025-10-25

## Overview

This document defines the API contract between the macOS Swift/SwiftUI frontend and the Python/MLX backend service for real-time image processing. The backend provides video downloading, frame extraction, and GPU-accelerated image effect pipelines.

### Communication Protocol

- **Primary**: XPC (Cross-Process Communication)
- **Image Transfer**: Shared memory buffers
- **Serialization**: JSON for structured data
- **Progress Updates**: Callback-based async notifications
- **GPU Acceleration**: Taichi kernels compiled to Metal for Apple Silicon

### Taichi Integration

The backend uses [Taichi](https://github.com/taichi-dev/taichi) for GPU-accelerated image processing:

- **Installation**: `pip install taichi`
- **Compilation Target**: Metal (Apple GPU) on macOS
- **Effect Implementation**: Each of the 17 effects is a separate Taichi kernel
- **Performance**: Taichi kernels execute on Apple Silicon GPU for <1s preview renders
- **Memory**: Direct integration with shared memory buffers for zero-copy transfer

### Performance SLAs

| Operation | Target | Maximum |
|-----------|--------|---------|
| Preview Render (960×540) | <1s | 2s |
| Pipeline Validation | <50ms | 100ms |
| Effect Introspection | <10ms | 50ms |
| Session Creation | <100ms | 500ms |
| Full Frame Render (1920×1080) | <5s | 10s |

---

## 1. Video Source Operations

### 1.1 Download YouTube Video

Downloads a YouTube video to local storage and validates the file.

**Signature:**
```python
def download_youtube(url: str, output_path: str) -> DownloadResult
```

**XPC Message:**
```json
{
  "method": "download_youtube",
  "params": {
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "output_path": "/tmp/sessions/abc123/source.mp4"
  },
  "request_id": "req_001"
}
```

**Response:**
```json
{
  "request_id": "req_001",
  "status": "success",
  "result": {
    "path": "/tmp/sessions/abc123/source.mp4",
    "size_bytes": 45839201,
    "duration_seconds": 212.5,
    "resolution": {"width": 1920, "height": 1080},
    "fps": 30.0,
    "codec": "h264"
  }
}
```

**Progress Callbacks:**
```json
{
  "request_id": "req_001",
  "type": "progress",
  "data": {
    "current_bytes": 12345678,
    "total_bytes": 45839201,
    "percent": 26.9,
    "status": "downloading",
    "message": "Downloading: 11.8 MB / 43.7 MB"
  }
}
```

**Errors:**
```json
{
  "request_id": "req_001",
  "status": "error",
  "error": {
    "code": "DOWNLOAD_FAILED",
    "message": "Failed to download video: Network timeout",
    "details": {
      "url": "https://www.youtube.com/watch?v=invalid",
      "reason": "HTTP 404"
    }
  }
}
```

**Error Codes:**
- `INVALID_URL` - Malformed or unsupported URL
- `DOWNLOAD_FAILED` - Network or YouTube extraction error
- `INVALID_VIDEO` - Downloaded file is corrupted or unsupported format
- `DISK_FULL` - Insufficient disk space
- `PERMISSION_DENIED` - Cannot write to output path

---

### 1.2 Extract Frames

Extracts a sequence of frames from a video file.

**Signature:**
```python
def extract_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    interval: float,
    output_dir: str
) -> ExtractionResult
```

**XPC Message:**
```json
{
  "method": "extract_frames",
  "params": {
    "video_path": "/tmp/sessions/abc123/source.mp4",
    "start_time": 192.0,
    "end_time": 195.0,
    "interval": 0.5,
    "output_dir": "/tmp/sessions/abc123/frames"
  },
  "request_id": "req_002"
}
```

**Response:**
```json
{
  "request_id": "req_002",
  "status": "success",
  "result": {
    "output_dir": "/tmp/sessions/abc123/frames",
    "frame_count": 6,
    "frames": [
      {
        "index": 0,
        "timestamp": 192.0,
        "path": "/tmp/sessions/abc123/frames/frame_0000.png",
        "size_bytes": 2841920
      },
      {
        "index": 1,
        "timestamp": 192.5,
        "path": "/tmp/sessions/abc123/frames/frame_0001.png",
        "size_bytes": 2839104
      }
    ],
    "resolution": {"width": 1920, "height": 1080},
    "format": "png"
  }
}
```

**Progress Callbacks:**
```json
{
  "request_id": "req_002",
  "type": "progress",
  "data": {
    "current": 3,
    "total": 6,
    "percent": 50.0,
    "status": "extracting",
    "message": "Extracting frame 3/6 at 193.5s"
  }
}
```

**Error Codes:**
- `VIDEO_NOT_FOUND` - Video file does not exist
- `INVALID_TIME_RANGE` - start_time >= end_time or outside video duration
- `INVALID_INTERVAL` - interval <= 0
- `EXTRACTION_FAILED` - FFmpeg extraction error
- `DISK_FULL` - Insufficient disk space for frames

---

### 1.3 Get Video Metadata

Retrieves metadata for a video file without extracting frames.

**Signature:**
```python
def get_video_metadata(video_path: str) -> VideoMetadata
```

**XPC Message:**
```json
{
  "method": "get_video_metadata",
  "params": {
    "video_path": "/tmp/sessions/abc123/source.mp4"
  },
  "request_id": "req_003"
}
```

**Response:**
```json
{
  "request_id": "req_003",
  "status": "success",
  "result": {
    "path": "/tmp/sessions/abc123/source.mp4",
    "duration_seconds": 212.5,
    "resolution": {"width": 1920, "height": 1080},
    "fps": 30.0,
    "codec": "h264",
    "bitrate_kbps": 3500,
    "size_bytes": 45839201,
    "has_audio": true,
    "audio_codec": "aac"
  }
}
```

**Error Codes:**
- `VIDEO_NOT_FOUND` - Video file does not exist
- `INVALID_VIDEO` - File is not a valid video
- `METADATA_READ_FAILED` - Cannot read video metadata

---

## 2. Frame Management

### 2.1 List Session Frames

Returns all extracted frames for a session.

**Signature:**
```python
def list_session_frames(session_id: str) -> FrameList
```

**XPC Message:**
```json
{
  "method": "list_session_frames",
  "params": {
    "session_id": "abc123"
  },
  "request_id": "req_004"
}
```

**Response:**
```json
{
  "request_id": "req_004",
  "status": "success",
  "result": {
    "session_id": "abc123",
    "frame_count": 6,
    "frames": [
      {
        "index": 0,
        "timestamp": 192.0,
        "path": "/tmp/sessions/abc123/frames/frame_0000.png",
        "thumbnail_path": "/tmp/sessions/abc123/frames/frame_0000_thumb.jpg",
        "size_bytes": 2841920
      }
    ],
    "resolution": {"width": 1920, "height": 1080}
  }
}
```

**Error Codes:**
- `SESSION_NOT_FOUND` - Session ID does not exist
- `NO_FRAMES_EXTRACTED` - Session exists but contains no frames

---

### 2.2 Get Frame Path

Retrieves the file path for a specific frame by index.

**Signature:**
```python
def get_frame_path(session_id: str, frame_index: int) -> FramePath
```

**XPC Message:**
```json
{
  "method": "get_frame_path",
  "params": {
    "session_id": "abc123",
    "frame_index": 2
  },
  "request_id": "req_005"
}
```

**Response:**
```json
{
  "request_id": "req_005",
  "status": "success",
  "result": {
    "index": 2,
    "path": "/tmp/sessions/abc123/frames/frame_0002.png",
    "timestamp": 193.0,
    "exists": true
  }
}
```

**Error Codes:**
- `SESSION_NOT_FOUND` - Session ID does not exist
- `FRAME_INDEX_OUT_OF_RANGE` - Index is negative or >= frame_count
- `FRAME_NOT_FOUND` - Frame file has been deleted

---

## 3. Pipeline Rendering (Core API)

### 3.1 Render Preview

Renders a low-resolution preview to shared memory for real-time UI updates.

**Signature:**
```python
def render_preview(
    frame_path: str,
    pipeline_json: str,
    output_buffer_id: str
) -> PreviewResult
```

**XPC Message:**
```json
{
  "method": "render_preview",
  "params": {
    "frame_path": "/tmp/sessions/abc123/frames/frame_0000.png",
    "pipeline_json": "{\"effects\":[{\"id\":\"uuid1\",\"name\":\"saturation\",\"enabled\":true,\"repeat\":4,\"params\":{\"factor\":1.5}}]}",
    "output_buffer_id": "preview_buffer_001"
  },
  "request_id": "req_006"
}
```

**Pipeline JSON Schema:**
```json
{
  "source": {
    "type": "youtube",
    "path": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "segment": {
      "start": 192.0,
      "end": 195.0
    }
  },
  "effects": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "saturation",
      "enabled": true,
      "repeat": 4,
      "trig_condition": "every_2nd",
      "probability": 0.75,
      "seed": 12345,
      "params": {
        "factor": 1.5
      }
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "name": "blur",
      "enabled": false,
      "repeat": 1,
      "trig_condition": "all",
      "probability": 1.0,
      "params": {
        "radius": {
          "value": 10.0,
          "ramp": {
            "enabled": true,
            "start": 5.0,
            "end": 20.0,
            "curve": "exponential_in"
          }
        }
      }
    }
  ]
}
```

**Advanced Parameter Control Fields (EPIC-010):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trig_condition` | string | No | When to apply effect: "all", "every_2nd", "every_3rd", "every_4th", "fill". Default: "all" |
| `probability` | float | No | Probability (0.0-1.0) effect applies to eligible frames. Default: 1.0 |
| `seed` | integer | No | Random seed for reproducible probability. If omitted, uses timestamp |

**Parameter Ramp Schema (EPIC-010):**

Parameters can specify either a simple value or a ramping configuration:

```json
// Simple value (legacy format, still supported)
"radius": 3.0

// Ramping value
"radius": {
  "value": 10.0,       // Current/default value
  "ramp": {
    "enabled": true,
    "start": 5.0,       // Value at frame 0
    "end": 20.0,        // Value at last frame
    "curve": "linear"   // "linear", "ease_in", "ease_out", "exponential"
  }
}
```

**Trig Condition Logic:**

- `all`: Apply effect to every frame
- `every_2nd`: Apply if `frame_index % 2 == 0`
- `every_3rd`: Apply if `frame_index % 3 == 0`
- `every_4th`: Apply if `frame_index % 4 == 0`
- `fill`: Apply only if `frame_index == total_frames - 1`

**Probability Logic:**

```python
if seed is not None:
    random.seed(seed)
apply_effect = random.random() < probability
```

**Ramp Interpolation:**

```python
def interpolate_ramp(start: float, end: float, frame_index: int, total_frames: int, curve: str) -> float:
    t = frame_index / (total_frames - 1) if total_frames > 1 else 0.0

    if curve == "linear":
        factor = t
    elif curve == "ease_in":
        factor = t ** 2
    elif curve == "ease_out":
        factor = 1.0 - (1.0 - t) ** 2
    elif curve == "exponential":
        factor = t ** 3
    else:
        raise ValueError(f"Unknown curve type: {curve}")

    return start + (end - start) * factor
```

**Response:**
```json
{
  "request_id": "req_006",
  "status": "success",
  "result": {
    "buffer_id": "preview_buffer_001",
    "resolution": {"width": 960, "height": 540},
    "format": "RGBA",
    "bytes_per_row": 3840,
    "total_bytes": 2073600,
    "render_time_ms": 847
  }
}
```

**Shared Memory Protocol:**

The backend writes rendered image data to a named shared memory region identified by `buffer_id`. The Swift frontend must:

1. Create shared memory region before calling `render_preview`
2. Pass the buffer ID to the backend
3. Read pixel data after receiving success response
4. Handle RGBA format (4 bytes per pixel, premultiplied alpha)

**Swift Example:**
```swift
// 1. Create shared memory
let bufferID = "preview_buffer_\(UUID())"
let width = 960, height = 540
let bytesPerPixel = 4
let totalBytes = width * height * bytesPerPixel

guard let sharedMemory = SharedMemory.create(
    name: bufferID,
    size: totalBytes
) else {
    print("Failed to create shared memory")
    return
}

// 2. Call backend
let request = XPCRequest(
    method: "render_preview",
    params: [
        "frame_path": framePath,
        "pipeline_json": pipelineJSON,
        "output_buffer_id": bufferID
    ]
)

xpcService.send(request) { response in
    guard response.status == "success" else {
        print("Render failed: \(response.error)")
        return
    }

    // 3. Read pixel data from shared memory
    let pixelData = sharedMemory.readBytes(count: totalBytes)

    // 4. Create CGImage
    let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * bytesPerPixel,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(
            rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        provider: CGDataProvider(data: pixelData as CFData)!,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    )
}
```

**Error Codes:**
- `FRAME_NOT_FOUND` - Input frame does not exist
- `INVALID_PIPELINE` - Pipeline JSON is malformed or invalid
- `BUFFER_NOT_FOUND` - Shared memory buffer does not exist
- `RENDER_FAILED` - GPU rendering error
- `EFFECT_NOT_FOUND` - Unknown effect name in pipeline
- `INVALID_PARAMETERS` - Effect parameters out of range

---

### 3.2 Render Frame

Renders a high-quality image to disk (non-preview).

**Signature:**
```python
def render_frame(
    frame_path: str,
    pipeline_json: str,
    output_path: str
) -> RenderResult
```

**XPC Message:**
```json
{
  "method": "render_frame",
  "params": {
    "frame_path": "/tmp/sessions/abc123/frames/frame_0000.png",
    "pipeline_json": "{\"effects\":[...]}",
    "output_path": "/tmp/sessions/abc123/output/result_0000.png"
  },
  "request_id": "req_007"
}
```

**Response:**
```json
{
  "request_id": "req_007",
  "status": "success",
  "result": {
    "output_path": "/tmp/sessions/abc123/output/result_0000.png",
    "resolution": {"width": 1920, "height": 1080},
    "format": "PNG",
    "size_bytes": 5839201,
    "render_time_ms": 3241
  }
}
```

**Error Codes:**
- `FRAME_NOT_FOUND` - Input frame does not exist
- `INVALID_PIPELINE` - Pipeline JSON is malformed
- `RENDER_FAILED` - GPU rendering error
- `DISK_FULL` - Cannot write output file
- `PERMISSION_DENIED` - Cannot write to output path

---

### 3.3 Render Sequence

Batch renders multiple frames with the same pipeline, with support for incremental non-destructive output.

**Signature:**
```python
def render_sequence(
    frame_paths: List[str],
    pipeline_json: str,
    output_dir: str,
    pipeline_name: str,
    version_increment: bool = True,
    frame_overrides: List[Dict[str, Any]] = None,
    progress_callback: Callable[[ProgressUpdate], None] = None
) -> SequenceResult
```

**XPC Message:**
```json
{
  "method": "render_sequence",
  "params": {
    "frame_paths": [
      "/tmp/sessions/abc123/frames/frame_0000.png",
      "/tmp/sessions/abc123/frames/frame_0001.png",
      "/tmp/sessions/abc123/frames/frame_0002.png"
    ],
    "pipeline_json": "{\"effects\":[...]}",
    "output_dir": "/tmp/sessions/abc123/output",
    "pipeline_name": "saturation_boost",
    "version_increment": true,
    "output_format": "png",
    "frame_overrides": [
      {
        "frame_index": 0,
        "effect_overrides": [
          {"effect_id": "550e8400-e29b-41d4-a716-446655440000", "enabled": false}
        ]
      },
      {
        "frame_index": 2,
        "effect_overrides": [
          {"effect_id": "550e8400-e29b-41d4-a716-446655440001", "enabled": true}
        ]
      }
    ]
  },
  "request_id": "req_008"
}
```

**Incremental Filename Format:**

When `version_increment: true` (default), output files use:
```
{pipeline_name}_{frame_stem}_v{version:03d}.{ext}
```

Examples:
```
saturation_boost_frame_0000_v001.png
saturation_boost_frame_0000_v002.png  # Second render
saturation_boost_frame_0001_v001.png
```

The backend:
1. Scans `output_dir` for existing files matching `{pipeline_name}_{frame_stem}_v*.{ext}`
2. Finds highest version number
3. Increments by 1 for new output
4. Never overwrites existing files (non-destructive mode)

When `version_increment: false`:
```
{pipeline_name}_{frame_stem}.{ext}
```

**Response:**
```json
{
  "request_id": "req_008",
  "status": "success",
  "result": {
    "output_dir": "/tmp/sessions/abc123/output",
    "pipeline_name": "saturation_boost",
    "version_used": 1,
    "frame_count": 3,
    "frames": [
      {
        "index": 0,
        "input_path": "/tmp/sessions/abc123/frames/frame_0000.png",
        "output_path": "/tmp/sessions/abc123/output/saturation_boost_frame_0000_v001.png",
        "size_bytes": 5839201,
        "render_time_ms": 3241
      },
      {
        "index": 1,
        "input_path": "/tmp/sessions/abc123/frames/frame_0001.png",
        "output_path": "/tmp/sessions/abc123/output/saturation_boost_frame_0001_v001.png",
        "size_bytes": 5841032,
        "render_time_ms": 3198
      }
    ],
    "total_render_time_ms": 9847,
    "average_render_time_ms": 3282
  }
}
```

**Progress Callbacks:**
```json
{
  "request_id": "req_008",
  "type": "progress",
  "data": {
    "current": 2,
    "total": 3,
    "percent": 66.7,
    "status": "processing",
    "message": "Rendering frame 2/3",
    "current_frame": {
      "index": 1,
      "path": "/tmp/sessions/abc123/frames/frame_0001.png"
    }
  }
}
```

**Frame Overrides (Optional Parameter):**

The `frame_overrides` parameter enables per-frame effect toggling for step sequencer functionality. When provided:

- Backend filters the pipeline's `effects` array before rendering each frame
- For each frame index with overrides:
  - If `enabled: false`, the effect is removed from the pipeline for that frame only
  - If `enabled: true`, the effect is explicitly included (default behavior)
- Frames without overrides use the full pipeline as-is
- Override validation:
  - `frame_index` must be < frame count (error if out of range)
  - `effect_id` must exist in the pipeline's `effects` array (error if unknown)

**Example Override Behavior:**
```json
Pipeline: [{"id": "A", "name": "blur"}, {"id": "B", "name": "saturation"}]
Overrides: [
  {"frame_index": 0, "effect_overrides": [{"effect_id": "A", "enabled": false}]}
]

Frame 0: Renders with [saturation] only (blur disabled)
Frame 1: Renders with [blur, saturation] (no overrides)
Frame 2: Renders with [blur, saturation] (no overrides)
```

**Error Codes:**
- `FRAME_NOT_FOUND` - One or more input frames do not exist
- `INVALID_PIPELINE` - Pipeline JSON is malformed
- `RENDER_FAILED` - GPU rendering error on one or more frames
- `DISK_FULL` - Cannot write output files
- `PARTIAL_FAILURE` - Some frames rendered successfully, others failed
- `INVALID_OVERRIDE` - frame_index out of range or effect_id not found

---

## 4. Effect Introspection

### 4.1 List Available Effects

Returns all 17 supported image effects.

**Signature:**
```python
def list_available_effects() -> EffectList
```

**XPC Message:**
```json
{
  "method": "list_available_effects",
  "params": {},
  "request_id": "req_009"
}
```

**Response:**
```json
{
  "request_id": "req_009",
  "status": "success",
  "result": {
    "effects": [
      {
        "name": "saturation",
        "display_name": "Saturation",
        "category": "color",
        "description": "Adjust color saturation intensity",
        "supports_repeat": true
      },
      {
        "name": "blur",
        "display_name": "Gaussian Blur",
        "category": "filter",
        "description": "Apply Gaussian blur filter",
        "supports_repeat": true
      },
      {
        "name": "edge_detection",
        "display_name": "Edge Detection",
        "category": "analysis",
        "description": "Detect edges using Sobel operator",
        "supports_repeat": false
      }
    ],
    "count": 17
  }
}
```

**Effect Categories:**
- `color` - Color manipulation (saturation, hue, brightness)
- `filter` - Spatial filters (blur, sharpen)
- `geometric` - Transformations (rotate, scale, distort)
- `analysis` - Feature detection (edges, contours)
- `artistic` - Stylization effects (oil paint, sketch)
- `glitch` - Digital artifacts (datamosh, pixelsort)

---

### 4.2 Get Effect Parameters

Returns the parameter schema for a specific effect.

**Signature:**
```python
def get_effect_parameters(effect_name: str) -> EffectParameters
```

**XPC Message:**
```json
{
  "method": "get_effect_parameters",
  "params": {
    "effect_name": "saturation"
  },
  "request_id": "req_010"
}
```

**Response:**
```json
{
  "request_id": "req_010",
  "status": "success",
  "result": {
    "name": "saturation",
    "display_name": "Saturation",
    "supports_repeat": true,
    "parameters": {
      "factor": {
        "type": "float",
        "min": 0.0,
        "max": 3.0,
        "default": 1.0,
        "step": 0.1,
        "description": "Saturation intensity (0=grayscale, 1=original, >1=boosted)"
      }
    }
  }
}
```

**Parameter Types:**

| Type | Description | Additional Fields |
|------|-------------|-------------------|
| `float` | Decimal number | `min`, `max`, `step`, `default` |
| `int` | Integer | `min`, `max`, `step`, `default` |
| `bool` | Boolean | `default` |
| `enum` | Enumeration | `options`, `default` |
| `color` | RGB/RGBA color | `default` (hex string) |

**Complex Example (Multiple Parameters):**
```json
{
  "name": "pixelsort",
  "display_name": "Pixel Sort",
  "supports_repeat": true,
  "parameters": {
    "threshold": {
      "type": "float",
      "min": 0.0,
      "max": 1.0,
      "default": 0.5,
      "step": 0.01,
      "description": "Brightness threshold for sorting"
    },
    "direction": {
      "type": "enum",
      "options": ["horizontal", "vertical", "diagonal"],
      "default": "horizontal",
      "description": "Sort direction"
    },
    "reverse": {
      "type": "bool",
      "default": false,
      "description": "Reverse sort order"
    },
    "mask_color": {
      "type": "color",
      "default": "#FF0000",
      "description": "Color for masked regions"
    }
  }
}
```

**Error Codes:**
- `EFFECT_NOT_FOUND` - Unknown effect name

---

### 4.3 Get Effect Defaults

Returns default parameter values for an effect (convenience method).

**Signature:**
```python
def get_effect_defaults(effect_name: str) -> EffectDefaults
```

**XPC Message:**
```json
{
  "method": "get_effect_defaults",
  "params": {
    "effect_name": "saturation"
  },
  "request_id": "req_011"
}
```

**Response:**
```json
{
  "request_id": "req_011",
  "status": "success",
  "result": {
    "name": "saturation",
    "repeat": 1,
    "params": {
      "factor": 1.0
    }
  }
}
```

---

## 5. Pipeline Management

### 5.1 Validate Pipeline

Checks if a pipeline JSON is well-formed and all effects/parameters are valid.

**Signature:**
```python
def validate_pipeline(pipeline_json: str) -> ValidationResult
```

**XPC Message:**
```json
{
  "method": "validate_pipeline",
  "params": {
    "pipeline_json": "{\"effects\":[...]}"
  },
  "request_id": "req_012"
}
```

**Response (Valid):**
```json
{
  "request_id": "req_012",
  "status": "success",
  "result": {
    "valid": true,
    "effect_count": 3,
    "total_operations": 7,
    "estimated_preview_time_ms": 850,
    "estimated_full_render_time_ms": 4200
  }
}
```

**Response (Invalid):**
```json
{
  "request_id": "req_012",
  "status": "success",
  "result": {
    "valid": false,
    "errors": [
      {
        "effect_id": "550e8400-e29b-41d4-a716-446655440001",
        "effect_name": "blur",
        "error_type": "INVALID_PARAMETER",
        "parameter": "radius",
        "message": "Parameter 'radius' value 100.0 exceeds maximum 50.0"
      },
      {
        "effect_id": "550e8400-e29b-41d4-a716-446655440002",
        "effect_name": "unknown_effect",
        "error_type": "EFFECT_NOT_FOUND",
        "message": "Effect 'unknown_effect' does not exist"
      }
    ]
  }
}
```

**Validation Error Types:**
- `JSON_PARSE_ERROR` - Malformed JSON
- `MISSING_REQUIRED_FIELD` - Pipeline missing required fields
- `EFFECT_NOT_FOUND` - Unknown effect name
- `INVALID_PARAMETER` - Parameter value out of range
- `MISSING_PARAMETER` - Required parameter not provided
- `INVALID_REPEAT_COUNT` - Repeat count < 1 or > 100

---

### 5.2 Export YAML

Saves a pipeline to YAML format for version control.

**Signature:**
```python
def export_yaml(pipeline_json: str, output_path: str) -> ExportResult
```

**XPC Message:**
```json
{
  "method": "export_yaml",
  "params": {
    "pipeline_json": "{\"effects\":[...]}",
    "output_path": "/tmp/sessions/abc123/pipeline.yaml"
  },
  "request_id": "req_013"
}
```

**Response:**
```json
{
  "request_id": "req_013",
  "status": "success",
  "result": {
    "output_path": "/tmp/sessions/abc123/pipeline.yaml",
    "size_bytes": 1024
  }
}
```

**Example YAML Output:**
```yaml
source:
  type: youtube
  path: https://www.youtube.com/watch?v=dQw4w9WgXcQ
  segment:
    start: 192.0
    end: 195.0

effects:
  - id: 550e8400-e29b-41d4-a716-446655440000
    name: saturation
    enabled: true
    repeat: 4
    trig_condition: every_2nd
    probability: 0.75
    seed: 12345
    params:
      factor: 1.5

  - id: 550e8400-e29b-41d4-a716-446655440001
    name: blur
    enabled: false
    repeat: 1
    trig_condition: all
    probability: 1.0
    params:
      radius:
        value: 10.0
        ramp:
          enabled: true
          start: 5.0
          end: 20.0
          curve: exponential_in
```

**Error Codes:**
- `INVALID_PIPELINE` - Pipeline JSON is malformed
- `PERMISSION_DENIED` - Cannot write to output path
- `DISK_FULL` - Insufficient disk space

---

### 5.3 Import YAML

Loads a YAML pipeline file into JSON format.

**Signature:**
```python
def import_yaml(yaml_path: str) -> ImportResult
```

**XPC Message:**
```json
{
  "method": "import_yaml",
  "params": {
    "yaml_path": "/tmp/sessions/abc123/pipeline.yaml"
  },
  "request_id": "req_014"
}
```

**Response:**
```json
{
  "request_id": "req_014",
  "status": "success",
  "result": {
    "pipeline_json": "{\"source\":{\"type\":\"youtube\",...},\"effects\":[...]}"
  }
}
```

**Error Codes:**
- `FILE_NOT_FOUND` - YAML file does not exist
- `YAML_PARSE_ERROR` - Malformed YAML
- `INVALID_PIPELINE` - YAML contains invalid pipeline structure

---

## 6. Session Management

### 6.1 Create Session

Initializes a new working session with temporary directory structure.

**Signature:**
```python
def create_session() -> SessionInfo
```

**XPC Message:**
```json
{
  "method": "create_session",
  "params": {},
  "request_id": "req_015"
}
```

**Response:**
```json
{
  "request_id": "req_015",
  "status": "success",
  "result": {
    "session_id": "abc123def456",
    "created_at": "2025-10-25T14:32:00Z",
    "base_dir": "/tmp/sessions/abc123def456",
    "directories": {
      "source": "/tmp/sessions/abc123def456/source",
      "frames": "/tmp/sessions/abc123def456/frames",
      "output": "/tmp/sessions/abc123def456/output",
      "cache": "/tmp/sessions/abc123def456/cache"
    }
  }
}
```

**Directory Structure:**
```
/tmp/sessions/abc123def456/
├── source/          # Downloaded videos
├── frames/          # Extracted frames
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── frame_0000_thumb.jpg  # Thumbnails (optional)
├── output/          # Rendered results
│   ├── result_0000.png
│   └── pipeline.yaml
└── cache/           # Temporary processing files
```

---

### 6.2 Cleanup Session

Removes all temporary files for a session.

**Signature:**
```python
def cleanup_session(session_id: str, keep_outputs: bool = False) -> CleanupResult
```

**XPC Message:**
```json
{
  "method": "cleanup_session",
  "params": {
    "session_id": "abc123def456",
    "keep_outputs": false
  },
  "request_id": "req_016"
}
```

**Response:**
```json
{
  "request_id": "req_016",
  "status": "success",
  "result": {
    "session_id": "abc123def456",
    "deleted_bytes": 125839201,
    "deleted_files": 47,
    "kept_outputs": false
  }
}
```

**Error Codes:**
- `SESSION_NOT_FOUND` - Session ID does not exist

---

## 7. XPC Protocol Specification

### 7.1 Message Format

All XPC messages use JSON serialization with the following structure:

**Request:**
```json
{
  "method": "method_name",
  "params": {
    "param1": "value1",
    "param2": 123
  },
  "request_id": "unique_request_id"
}
```

**Response:**
```json
{
  "request_id": "unique_request_id",
  "status": "success" | "error",
  "result": { ... }  // Only present if status == "success"
  "error": { ... }   // Only present if status == "error"
}
```

**Error Structure:**
```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "additional": "context"
  }
}
```

### 7.2 Request ID Generation

- Frontend generates unique request IDs (UUID recommended)
- Backend echoes request ID in all responses and progress callbacks
- Allows matching async responses to requests

### 7.3 Long-Running Operations

Operations that take >500ms should send progress callbacks:

**Progress Callback:**
```json
{
  "request_id": "req_001",
  "type": "progress",
  "data": {
    "current": 5,
    "total": 10,
    "percent": 50.0,
    "status": "processing" | "complete" | "error",
    "message": "Human-readable status",
    "metadata": {  // Optional
      "current_item": "frame_0005.png"
    }
  }
}
```

**Final Response:**
```json
{
  "request_id": "req_001",
  "type": "response",
  "status": "success",
  "result": { ... }
}
```

### 7.4 Swift XPC Client Example

```swift
import Foundation

class BackendXPCClient {
    private let connection: NSXPCConnection
    private var progressHandlers: [String: (ProgressUpdate) -> Void] = [:]

    init() {
        connection = NSXPCConnection(serviceName: "com.7rad.backend")
        connection.remoteObjectInterface = NSXPCInterface(with: BackendServiceProtocol.self)
        connection.resume()
    }

    func renderPreview(
        framePath: String,
        pipeline: Pipeline,
        bufferID: String,
        progress: @escaping (ProgressUpdate) -> Void,
        completion: @escaping (Result<PreviewResult, BackendError>) -> Void
    ) {
        let requestID = UUID().uuidString
        progressHandlers[requestID] = progress

        let pipelineJSON = try! JSONEncoder().encode(pipeline)
        let request = XPCRequest(
            method: "render_preview",
            params: [
                "frame_path": framePath,
                "pipeline_json": String(data: pipelineJSON, encoding: .utf8)!,
                "output_buffer_id": bufferID
            ],
            requestID: requestID
        )

        let proxy = connection.remoteObjectProxyWithErrorHandler { error in
            completion(.failure(.connectionError(error)))
        } as! BackendServiceProtocol

        proxy.sendRequest(request) { [weak self] response in
            self?.progressHandlers.removeValue(forKey: requestID)

            if response.status == "success" {
                let result = try! JSONDecoder().decode(
                    PreviewResult.self,
                    from: response.result!
                )
                completion(.success(result))
            } else {
                let error = try! JSONDecoder().decode(
                    BackendError.self,
                    from: response.error!
                )
                completion(.failure(error))
            }
        }
    }

    func handleProgressCallback(_ callback: XPCCallback) {
        guard callback.type == "progress",
              let handler = progressHandlers[callback.requestID] else {
            return
        }

        let progress = try! JSONDecoder().decode(
            ProgressUpdate.self,
            from: callback.data
        )
        handler(progress)
    }
}

// Swift types
struct Pipeline: Codable {
    let source: Source?
    let effects: [Effect]
}

struct Effect: Codable, Identifiable {
    let id: UUID
    let name: String
    let enabled: Bool
    let `repeat`: Int
    let params: [String: AnyCodable]
}

struct PreviewResult: Codable {
    let bufferID: String
    let resolution: Resolution
    let format: String
    let bytesPerRow: Int
    let totalBytes: Int
    let renderTimeMs: Int
}

struct ProgressUpdate: Codable {
    let current: Int
    let total: Int
    let percent: Double
    let status: String
    let message: String
}
```

---

## 8. Shared Memory Protocol

### 8.1 Buffer Creation

**Frontend Responsibilities:**
1. Create named shared memory region using POSIX `shm_open(O_CREAT | O_RDWR)`
2. Set appropriate size: `8 + (width × height × 4)` bytes (header + RGBA pixels)
3. Pass buffer name to backend via `output_buffer_id` parameter
4. Map memory region for reading after render completes
5. Clean up buffer with `shm_unlink()` after reading image

**Buffer Naming Convention:**
```
preview_buffer_<session_id>_<uuid>
```

**Buffer Header Format:**

The backend writes a header before pixel data to communicate image dimensions:

```
Offset 0-3:   Width (UInt32, little-endian)
Offset 4-7:   Height (UInt32, little-endian)
Offset 8+:    RGBA pixel data
```

Example for 960×540 image:
```
Total buffer size: 8 + (960 × 540 × 4) = 2,073,608 bytes

Bytes 0-3:   0xC0 0x03 0x00 0x00  (960 in little-endian)
Bytes 4-7:   0x1C 0x02 0x00 0x00  (540 in little-endian)
Bytes 8+:    RGBA pixel data (2,073,600 bytes)
```

This allows the frontend to read dimensions dynamically without hardcoding them.

### 8.2 Image Format

All preview renders use:
- **Pixel Format**: RGBA (4 bytes per pixel)
- **Byte Order**: Red, Green, Blue, Alpha
- **Alpha**: Premultiplied
- **Row Alignment**: No padding (stride = width × 4)
- **Color Space**: sRGB
- **Bit Depth**: 8 bits per component

### 8.3 Memory Layout

```
Offset 0: Red byte of pixel (0, 0)
Offset 1: Green byte of pixel (0, 0)
Offset 2: Blue byte of pixel (0, 0)
Offset 3: Alpha byte of pixel (0, 0)
Offset 4: Red byte of pixel (1, 0)
...
```

### 8.4 Swift Shared Memory Example

```swift
import Foundation
import CoreGraphics

class SharedMemory {
    private let name: String
    private let size: Int
    private var fileDescriptor: Int32 = -1
    private var pointer: UnsafeMutableRawPointer?

    static func create(name: String, size: Int) -> SharedMemory? {
        let shm = SharedMemory(name: name, size: size)

        // Create shared memory object
        shm.fileDescriptor = shm_open(
            name,
            O_CREAT | O_RDWR,
            S_IRUSR | S_IWUSR
        )

        guard shm.fileDescriptor != -1 else {
            return nil
        }

        // Set size
        guard ftruncate(shm.fileDescriptor, off_t(size)) == 0 else {
            close(shm.fileDescriptor)
            return nil
        }

        // Map memory
        shm.pointer = mmap(
            nil,
            size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            shm.fileDescriptor,
            0
        )

        guard shm.pointer != MAP_FAILED else {
            close(shm.fileDescriptor)
            return nil
        }

        return shm
    }

    func readBytes(count: Int) -> Data {
        guard let pointer = pointer else {
            return Data()
        }
        return Data(bytes: pointer, count: min(count, size))
    }

    func createCGImage(width: Int, height: Int) -> CGImage? {
        let pixelData = readBytes(count: width * height * 4)

        let dataProvider = CGDataProvider(data: pixelData as CFData)!

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            ),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    deinit {
        if let pointer = pointer {
            munmap(pointer, size)
        }
        if fileDescriptor != -1 {
            close(fileDescriptor)
            shm_unlink(name)
        }
    }

    private init(name: String, size: Int) {
        self.name = name
        self.size = size
    }
}
```

---

## 9. Error Handling Patterns

### 9.1 Error Code Categories

**Client Errors (4xx):**
- `INVALID_REQUEST` - Malformed request
- `INVALID_PARAMETER` - Parameter validation failed
- `RESOURCE_NOT_FOUND` - File, session, or effect not found
- `PERMISSION_DENIED` - Insufficient permissions

**Server Errors (5xx):**
- `RENDER_FAILED` - GPU rendering error
- `INTERNAL_ERROR` - Unexpected backend error
- `SERVICE_UNAVAILABLE` - Backend overloaded or crashed

**External Errors (6xx):**
- `DOWNLOAD_FAILED` - Network or YouTube error
- `DISK_FULL` - Insufficient storage
- `INVALID_VIDEO` - Corrupted or unsupported media

### 9.2 Error Recovery

**Frontend Retry Strategy:**

```swift
func renderPreviewWithRetry(
    framePath: String,
    pipeline: Pipeline,
    maxRetries: Int = 3
) async throws -> PreviewResult {
    var attempt = 0

    while attempt < maxRetries {
        do {
            return try await renderPreview(
                framePath: framePath,
                pipeline: pipeline
            )
        } catch let error as BackendError {
            attempt += 1

            switch error.code {
            case "SERVICE_UNAVAILABLE":
                // Exponential backoff
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt))) * 1_000_000_000)
                continue

            case "INVALID_PIPELINE", "INVALID_PARAMETER":
                // Don't retry client errors
                throw error

            default:
                if attempt >= maxRetries {
                    throw error
                }
            }
        }
    }

    throw BackendError(code: "MAX_RETRIES_EXCEEDED")
}
```

### 9.3 Graceful Degradation

If preview rendering fails or exceeds SLA:

1. **Fallback to Original Frame**: Show unprocessed frame with warning overlay
2. **Partial Pipeline Render**: Apply effects up to the point of failure
3. **Cached Previous Result**: If user is adjusting parameters, show last successful render

---

## 10. Performance Optimization

### 10.1 Preview Resolution Scaling

Target preview resolution based on viewport size:

| Viewport Width | Preview Width | Scale Factor |
|----------------|---------------|--------------|
| ≤ 800px | 640px | 0.33× |
| 801-1200px | 960px | 0.50× |
| 1201-1600px | 1280px | 0.67× |
| > 1600px | 1920px | 1.00× |

### 10.2 Effect Complexity Budgets

Each effect has an estimated render time:

| Effect | Complexity | Preview Time (960×540) |
|--------|-----------|----------------------|
| saturation | Low | ~20ms |
| blur (r=3) | Medium | ~80ms |
| edge_detection | Medium | ~100ms |
| pixelsort | High | ~200ms |
| datamosh | Very High | ~500ms |

Frontend should warn if total pipeline time exceeds 1.5 seconds.

### 10.3 Caching Strategy

**Backend Should Cache:**
- Loaded video metadata (avoid repeated FFmpeg probes)
- Effect parameter schemas (loaded once at startup)
- Intermediate render results (if same frame + partial pipeline)

**Cache Invalidation:**
- Session cleanup removes all cached data
- 1-hour TTL for video metadata
- Unlimited TTL for effect schemas (change requires backend restart)

---

## 11. Security Considerations

### 11.1 Path Validation

**Backend MUST:**
- Validate all file paths are within session directory
- Reject paths with `..` components
- Sanitize filenames to prevent directory traversal
- Use absolute paths internally

**Example:**
```python
def validate_session_path(session_id: str, path: str) -> bool:
    """Ensure path is within session directory."""
    session_dir = f"/tmp/sessions/{session_id}"
    resolved = os.path.realpath(path)
    return resolved.startswith(os.path.realpath(session_dir))
```

### 11.2 Resource Limits

**Backend MUST enforce:**
- Max 10 concurrent sessions per user
- Max 100 frames per extraction
- Max 50 effects per pipeline
- Max 100 repeat count per effect
- Max 10 GB total storage per session
- 30-minute timeout for long-running operations

### 11.3 Shared Memory Security

- Use unique, unpredictable buffer names (UUIDs)
- Set appropriate permissions (user-only read/write)
- Cleanup buffers after use
- Timeout buffer access after 60 seconds

---

## 12. Testing & Validation

### 12.1 Backend Unit Tests

Required test coverage:

```python
# tests/test_api.py

def test_render_preview_success():
    """Test successful preview render to shared memory."""
    result = render_preview(
        frame_path="/test/frame.png",
        pipeline_json='{"effects":[]}',
        output_buffer_id="test_buffer"
    )
    assert result.render_time_ms < 1000

def test_render_preview_invalid_pipeline():
    """Test pipeline validation errors."""
    with pytest.raises(InvalidPipelineError):
        render_preview(
            frame_path="/test/frame.png",
            pipeline_json='{"effects":[{"name":"unknown"}]}',
            output_buffer_id="test_buffer"
        )

def test_extract_frames_time_range():
    """Test frame extraction with time range."""
    result = extract_frames(
        video_path="/test/video.mp4",
        start_time=10.0,
        end_time=15.0,
        interval=1.0,
        output_dir="/test/output"
    )
    assert result.frame_count == 5
```

### 12.2 Integration Tests

```python
def test_full_pipeline_workflow():
    """Test complete workflow from download to render."""
    # Create session
    session = create_session()

    # Download video
    download_result = download_youtube(
        url="https://www.youtube.com/watch?v=test",
        output_path=f"{session.base_dir}/source/video.mp4"
    )

    # Extract frames
    frames = extract_frames(
        video_path=download_result.path,
        start_time=0,
        end_time=5,
        interval=1,
        output_dir=f"{session.base_dir}/frames"
    )

    # Render preview
    pipeline = {
        "effects": [
            {"name": "saturation", "enabled": True, "repeat": 1, "params": {"factor": 1.5}}
        ]
    }
    result = render_preview(
        frame_path=frames.frames[0].path,
        pipeline_json=json.dumps(pipeline),
        output_buffer_id="test_buffer"
    )

    assert result.render_time_ms < 1000

    # Cleanup
    cleanup_session(session.session_id)
```

### 12.3 Performance Benchmarks

```python
@pytest.mark.benchmark
def test_preview_render_performance(benchmark):
    """Benchmark preview render time."""
    result = benchmark(
        render_preview,
        frame_path="/test/1920x1080.png",
        pipeline_json='{"effects":[{"name":"blur","repeat":1,"params":{"radius":3}}]}',
        output_buffer_id="bench_buffer"
    )
    assert result.render_time_ms < 1000
```

---

## 13. Versioning & Compatibility

### 13.1 API Version Header

All requests should include API version:

```json
{
  "api_version": "1.0.0",
  "method": "render_preview",
  "params": { ... }
}
```

Backend responds with supported version range:

```json
{
  "api_version": "1.0.0",
  "min_supported": "1.0.0",
  "max_supported": "1.2.0",
  "deprecated_features": []
}
```

### 13.2 Breaking Changes

Version 1.x guarantees backward compatibility. Breaking changes require version 2.0.

**Example Migration (1.x → 2.0):**
- Removed: `render_frame` method (merged into `render_preview`)
- Changed: `pipeline_json` now requires `version` field
- Added: `render_preview` supports quality parameter

### 13.3 Deprecation Policy

Features marked deprecated will:
1. Log warnings for 2 minor versions
2. Return errors after 3rd minor version
3. Be removed in next major version

---

## 14. Monitoring & Observability

### 14.1 Metrics

Backend should expose metrics for monitoring:

```json
{
  "metrics": {
    "requests_total": 1234,
    "requests_failed": 23,
    "render_preview_time_p50_ms": 450,
    "render_preview_time_p95_ms": 850,
    "render_preview_time_p99_ms": 1200,
    "active_sessions": 3,
    "shared_memory_buffers_active": 5,
    "disk_usage_bytes": 458392011
  }
}
```

### 14.2 Logging

All operations should log structured JSON:

```json
{
  "timestamp": "2025-10-25T14:32:00Z",
  "level": "INFO",
  "method": "render_preview",
  "request_id": "req_001",
  "session_id": "abc123",
  "duration_ms": 847,
  "status": "success"
}
```

### 14.3 Health Check

```json
{
  "method": "health_check",
  "params": {}
}
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "mlx_available": true,
  "gpu_memory_available_mb": 4096
}
```

---

## Appendix A: Complete Effect List

All 17 effects must be implemented as Taichi kernels for GPU acceleration.

| Name | Category | Description | Key Parameters | Taichi Kernel |
|------|----------|-------------|----------------|---------------|
| saturation | color | Adjust color saturation | factor (0-3) | `@ti.kernel def saturation(...)` |
| hue_shift | color | Rotate hue | degrees (0-360) | `@ti.kernel def hue_shift(...)` |
| brightness | color | Adjust brightness | factor (0-3) | `@ti.kernel def brightness(...)` |
| contrast | color | Adjust contrast | factor (0-3) | `@ti.kernel def contrast(...)` |
| invert | color | Invert colors | - | `@ti.kernel def invert(...)` |
| blur | filter | Gaussian blur | radius (0-50) | `@ti.kernel def blur(...)` |
| sharpen | filter | Sharpen edges | strength (0-10) | `@ti.kernel def sharpen(...)` |
| edge_detection | analysis | Sobel edge detection | threshold (0-1) | `@ti.kernel def edge_detection(...)` |
| pixelsort | glitch | Sort pixels by brightness | threshold, direction | `@ti.kernel def pixelsort(...)` |
| datamosh | glitch | Simulate compression artifacts | strength (0-10) | `@ti.kernel def datamosh(...)` |
| glitch_lines | glitch | Horizontal line displacement | intensity (0-100) | `@ti.kernel def glitch_lines(...)` |
| rotate | geometric | Rotate image | degrees (0-360) | `@ti.kernel def rotate(...)` |
| scale | geometric | Scale image | factor (0.1-10) | `@ti.kernel def scale(...)` |
| flip_horizontal | geometric | Mirror horizontally | - | `@ti.kernel def flip_horizontal(...)` |
| flip_vertical | geometric | Mirror vertically | - | `@ti.kernel def flip_vertical(...)` |
| oil_paint | artistic | Oil painting effect | brush_size (1-20) | `@ti.kernel def oil_paint(...)` |
| sketch | artistic | Pencil sketch effect | detail (0-10) | `@ti.kernel def sketch(...)` |

**Implementation Notes:**
- Each kernel operates on Taichi fields (ti.field) for GPU memory
- Kernels should be compiled once at backend startup
- Input/output images converted to/from NumPy arrays for Taichi processing
- All kernels target Metal backend on macOS: `ti.init(arch=ti.metal)`

---

## Appendix B: Quick Reference

### Most Common Operations

**1. Extract frames from YouTube video:**
```json
create_session() → session_id
download_youtube(url, path) → video_path
extract_frames(video_path, start, end, interval, output_dir) → frames
```

**2. Render preview:**
```json
render_preview(frame_path, pipeline_json, buffer_id) → result
```

**3. Batch render sequence:**
```json
render_sequence(frame_paths, pipeline_json, output_dir) → results
```

**4. Build effect pipeline:**
```json
list_available_effects() → effects
get_effect_parameters(effect_name) → params
validate_pipeline(pipeline_json) → validation
```

---

## Appendix C: Python Backend Implementation Stub

```python
# backend/api.py

from typing import List, Dict, Any, Callable
import json

class BackendAPI:
    """Main API implementation for Swift/Python XPC communication."""

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route XPC request to appropriate handler."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("request_id")

        try:
            if method == "render_preview":
                result = self.render_preview(**params)
            elif method == "extract_frames":
                result = self.extract_frames(**params)
            elif method == "list_available_effects":
                result = self.list_available_effects()
            # ... other methods
            else:
                raise ValueError(f"Unknown method: {method}")

            return {
                "request_id": request_id,
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": type(e).__name__,
                    "message": str(e)
                }
            }

    def render_preview(
        self,
        frame_path: str,
        pipeline_json: str,
        output_buffer_id: str
    ) -> Dict[str, Any]:
        """Render preview to shared memory."""
        # Implementation here
        pass

    # ... other methods
```

---

**Document End**

For questions or clarifications, contact the backend development team.
