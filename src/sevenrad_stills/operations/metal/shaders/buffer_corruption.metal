#include <metal_stdlib>
using namespace metal;

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

/// Shuffle RGB channels based on permutation index (0-5)
uchar3 shuffle_channels(uchar3 rgb, uint pattern) {
    // 6 possible permutations of RGB
    switch (pattern % 6) {
        case 0: return rgb.rgb;  // RGB (original)
        case 1: return rgb.rbg;  // RBG
        case 2: return rgb.grb;  // GRB
        case 3: return rgb.gbr;  // GBR
        case 4: return rgb.brg;  // BRG
        case 5: return rgb.bgr;  // BGR
        default: return rgb.rgb;
    }
}

/// Apply XOR corruption to a single pixel
uchar3 apply_xor(uchar3 pixel, uint rand_value, uint magnitude) {
    uchar mask = (rand_value >> 24) & 0xFF;
    mask = mask % (magnitude + 1);  // Clamp to magnitude
    return pixel ^ mask;
}

/// Apply inversion to a single pixel
uchar3 apply_invert(uchar3 pixel) {
    return uchar3(255) - pixel;
}

/// Main buffer corruption kernel
/// Processes only ACTIVE tiles (not all pixels) to eliminate divergence
/// Matches Taichi's approach for maximum throughput
kernel void buffer_corruption(
    device uchar4 *image [[buffer(0)]],
    constant uint &width [[buffer(1)]],
    constant uint &height [[buffer(2)]],
    constant uint &seed [[buffer(3)]],
    constant uint &corruption_type [[buffer(4)]],  // 0=XOR, 1=INVERT, 2=CHANNEL_SHUFFLE
    constant uint &magnitude [[buffer(5)]],
    constant uint &tile_size [[buffer(6)]],
    constant uint2 *active_tiles [[buffer(7)]],    // List of active tile coordinates (tile_x, tile_y)
    uint3 gid [[thread_position_in_grid]])         // (tile_idx, local_y, local_x)
{
    // Each threadgroup processes one active tile
    // gid.x = tile index in active_tiles array
    // gid.y = local Y coordinate within tile
    // gid.z = local X coordinate within tile

    // Get active tile coordinates
    uint2 tile_coord = active_tiles[gid.x];
    uint tile_x = tile_coord.x;
    uint tile_y = tile_coord.y;

    // Calculate global pixel coordinates
    uint pixel_x = tile_x * tile_size + gid.z;
    uint pixel_y = tile_y * tile_size + gid.y;

    // Bounds check (tiles at image edges may be partial)
    if (pixel_x >= width || pixel_y >= height) {
        return;
    }

    // Get pixel index and current value
    uint idx = pixel_y * width + pixel_x;
    uchar4 pixel = image[idx];

    // Generate random value for this pixel
    uint rand = hash(pixel_x, pixel_y, seed);

    // Apply corruption based on type
    uchar3 rgb = pixel.rgb;

    switch (corruption_type) {
        case 0:  // XOR
            rgb = apply_xor(rgb, rand, magnitude);
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
    image[idx] = pixel;
}
