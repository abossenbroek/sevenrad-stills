#include <metal_stdlib>
using namespace metal;

/// Optimized buffer corruption shader using hybrid per-pixel dispatch
/// Performance target: 5-10ms on 4K images (vs 168ms in v1)
///
/// Key optimizations:
/// - Per-pixel dispatch (width × height) instead of per-tile
/// - Small tile grid lookup (2KB) instead of huge mask arrays (2.4MB)
/// - Single struct for parameters instead of 6 separate buffers
/// - Coalesced memory access patterns
/// - Minimal CPU↔GPU data transfer

/// Fast hash-based random number generator
/// Deterministic per-pixel based on coordinates and seed
uint hash(uint x, uint y, uint seed) {
    uint h = seed;
    h ^= x * 0x9e3779b9u;
    h ^= y * 0x9e3779b9u;
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    return h ^ (h >> 16);
}

/// Shuffle RGB channels based on hash value
uchar3 shuffle_channels(uchar3 rgb, uint hash_val) {
    // 6 possible permutations of RGB
    switch (hash_val % 6) {
        case 0: return rgb.rgb;  // RGB (original)
        case 1: return rgb.rbg;  // RBG
        case 2: return rgb.grb;  // GRB
        case 3: return rgb.gbr;  // GBR
        case 4: return rgb.brg;  // BRG
        case 5: return rgb.bgr;  // BGR
        default: return rgb.rgb;
    }
}

/// Apply XOR corruption to a pixel
uchar3 apply_xor(uchar3 pixel, uint hash_val, uint magnitude) {
    // Extract 3 bytes from hash for RGB channels
    uchar mask_r = (hash_val >> 0) & 0xFF;
    uchar mask_g = (hash_val >> 8) & 0xFF;
    uchar mask_b = (hash_val >> 16) & 0xFF;

    // Clamp to magnitude
    mask_r = mask_r % (magnitude + 1);
    mask_g = mask_g % (magnitude + 1);
    mask_b = mask_b % (magnitude + 1);

    return uchar3(
        pixel.r ^ mask_r,
        pixel.g ^ mask_g,
        pixel.b ^ mask_b
    );
}

/// Apply bitwise inversion
uchar3 apply_invert(uchar3 pixel) {
    return uchar3(255) - pixel;
}

/// Corruption parameters (packed in single struct for efficiency)
struct CorruptionParams {
    uint width;
    uint height;
    uint tile_size;
    uint seed;
    uint corruption_type;  // 0=XOR, 1=INVERT, 2=CHANNEL_SHUFFLE
    uint magnitude;        // For XOR mode
    uint grid_width;       // Tile grid width (computed: ceil(width / tile_size))
    uint grid_height;      // Tile grid height (computed: ceil(height / tile_size))
};

/// Optimized buffer corruption kernel - Hybrid Per-Pixel Dispatch
///
/// Dispatched with grid size (width, height) - one thread per pixel
///
/// Each thread:
/// 1. Calculates which tile it belongs to
/// 2. Looks up tile_grid to check if that tile is corrupted
/// 3. If yes, applies corruption using hash-based RNG
///
/// This achieves:
/// - Massive parallelism (e.g., 8.3M threads on 4K)
/// - Preserves block corruption visual effect
/// - Minimal data transfer (tiny tile grid vs huge masks)
/// - Coalesced memory access
///
kernel void buffer_corruption_v2(
    device uchar4 *image [[buffer(0)]],              // Image buffer (RGBA)
    constant CorruptionParams &params [[buffer(1)]], // Single struct with all parameters
    constant uchar *tile_grid [[buffer(2)]],         // Boolean grid marking corrupted tiles
    uint2 gid [[thread_position_in_grid]])           // (x, y) position in image
{
    // Bounds check
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // Calculate which tile this pixel belongs to
    uint tile_x = gid.x / params.tile_size;
    uint tile_y = gid.y / params.tile_size;

    // Lookup in tile grid: is this tile corrupted?
    // Grid is stored row-major: grid[tile_y * grid_width + tile_x]
    uint tile_idx = tile_y * params.grid_width + tile_x;

    if (tile_grid[tile_idx] == 0) {
        // This tile is not corrupted - early exit
        return;
    }

    // This tile IS corrupted - apply corruption to this pixel

    // Get pixel index and current value
    uint pixel_idx = gid.y * params.width + gid.x;
    uchar4 pixel = image[pixel_idx];

    // Generate deterministic random value for this pixel
    uint rand = hash(gid.x, gid.y, params.seed);

    // Apply corruption based on type
    uchar3 rgb = pixel.rgb;

    switch (params.corruption_type) {
        case 0:  // XOR
            rgb = apply_xor(rgb, rand, params.magnitude);
            break;
        case 1:  // INVERT
            rgb = apply_invert(rgb);
            break;
        case 2:  // CHANNEL_SHUFFLE
            rgb = shuffle_channels(rgb, rand);
            break;
    }

    // Write result (preserve alpha channel)
    pixel.rgb = rgb;
    image[pixel_idx] = pixel;
}
