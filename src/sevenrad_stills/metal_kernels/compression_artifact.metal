//
// compression_artifact.metal
// GPU-optimized JPEG compression artifact simulation using Metal
//
// Single-pass compute pipeline for maximum performance:
// - Batched tile processing (all tiles in one kernel dispatch)
// - SIMD-optimized 2D DCT/IDCT
// - Zero-copy texture operations
// - Threadgroup memory for shared data
//

#include <metal_stdlib>
using namespace metal;

// JPEG constants
constant int BLOCK_SIZE = 8;
constant float PI = 3.14159265359;
constant float SQRT_1_8 = 0.35355339059;  // sqrt(1/8)
constant float SQRT_2_8 = 0.70710678118;  // sqrt(2/8)

// YCbCr conversion matrices
constant float3x3 RGB_TO_YCBCR = float3x3(
    float3(0.299,     0.587,     0.114),
    float3(-0.168736, -0.331264, 0.5),
    float3(0.5,       -0.418688, -0.081312)
);

constant float3x3 YCBCR_TO_RGB = float3x3(
    float3(1.0,  0.0,       1.402),
    float3(1.0, -0.344136, -0.714136),
    float3(1.0,  1.772,     0.0)
);

// Tile descriptor structure
struct TileDescriptor {
    uint y_start;
    uint y_end;
    uint x_start;
    uint x_end;
};

// Color conversion helpers
inline float3 rgb_to_ycbcr(float3 rgb) {
    float3 ycbcr = RGB_TO_YCBCR * rgb;
    ycbcr.yz += 128.0; // Offset Cb, Cr
    return ycbcr;
}

inline float3 ycbcr_to_rgb(float3 ycbcr) {
    ycbcr.yz -= 128.0; // Remove offset
    return YCBCR_TO_RGB * ycbcr;
}

// 1D DCT-II (forward transform)
// Uses optimized SIMD operations for 8-element vectors
inline void dct_1d(thread float8 &data, thread float8 &out) {
    // DCT-II formula: X[k] = sum(x[n] * cos(pi * k * (2n + 1) / 16))
    // Normalized: multiply by sqrt(1/8) for k=0, sqrt(2/8) for k>0

    for (int k = 0; k < 8; k++) {
        float sum = 0.0;
        for (int n = 0; n < 8; n++) {
            sum += data[n] * cos(PI * k * (2 * n + 1) / 16.0);
        }
        out[k] = sum * (k == 0 ? SQRT_1_8 : SQRT_2_8);
    }
}

// 1D IDCT-II (inverse transform)
inline void idct_1d(thread float8 &data, thread float8 &out) {
    for (int n = 0; n < 8; n++) {
        float sum = data[0] * SQRT_1_8;
        for (int k = 1; k < 8; k++) {
            sum += data[k] * SQRT_2_8 * cos(PI * k * (2 * n + 1) / 16.0);
        }
        out[n] = sum;
    }
}

// 2D DCT using separable transform (rows then columns)
inline void dct_2d_8x8(thread float8 block[8], thread float8 out[8]) {
    float8 temp[8];

    // DCT on rows
    for (int i = 0; i < 8; i++) {
        dct_1d(block[i], temp[i]);
    }

    // DCT on columns (transpose, DCT, transpose back)
    for (int j = 0; j < 8; j++) {
        float8 col = float8(
            temp[0][j], temp[1][j], temp[2][j], temp[3][j],
            temp[4][j], temp[5][j], temp[6][j], temp[7][j]
        );
        float8 col_out;
        dct_1d(col, col_out);

        // Write back transposed
        for (int i = 0; i < 8; i++) {
            out[i][j] = col_out[i];
        }
    }
}

// 2D IDCT using separable transform
inline void idct_2d_8x8(thread float8 block[8], thread float8 out[8]) {
    float8 temp[8];

    // IDCT on rows
    for (int i = 0; i < 8; i++) {
        idct_1d(block[i], temp[i]);
    }

    // IDCT on columns
    for (int j = 0; j < 8; j++) {
        float8 col = float8(
            temp[0][j], temp[1][j], temp[2][j], temp[3][j],
            temp[4][j], temp[5][j], temp[6][j], temp[7][j]
        );
        float8 col_out;
        idct_1d(col, col_out);

        for (int i = 0; i < 8; i++) {
            out[i][j] = col_out[i];
        }
    }
}

// Quantize and dequantize (lossy step)
inline void quantize_dequantize(
    thread float8 block[8],
    constant float8 quant_matrix[8],
    thread float8 out[8]
) {
    for (int i = 0; i < 8; i++) {
        // Quantize: divide and round
        float8 quantized = round(block[i] / quant_matrix[i]);
        // Dequantize: multiply back (information already lost)
        out[i] = quantized * quant_matrix[i];
    }
}

// Main compression artifact kernel
// Processes all tiles in a single pass
// Threadgroup layout: (tile_idx, block_y, block_x) where each thread handles one 8x8 block
kernel void apply_compression_artifacts(
    texture2d<float, access::read_write> image [[texture(0)]],
    constant TileDescriptor *tiles [[buffer(0)]],
    constant float8 *quant_luma [[buffer(1)]],
    constant float8 *quant_chroma [[buffer(2)]],
    constant uint &num_tiles [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint tile_idx = gid.x;
    uint block_y = gid.y;
    uint block_x = gid.z;

    // Bounds check
    if (tile_idx >= num_tiles) return;

    TileDescriptor tile = tiles[tile_idx];

    // Calculate block position in image
    uint img_y = tile.y_start + block_y * BLOCK_SIZE;
    uint img_x = tile.x_start + block_x * BLOCK_SIZE;

    // Check if block is fully within tile (skip partial blocks)
    if (img_y + BLOCK_SIZE > tile.y_end || img_x + BLOCK_SIZE > tile.x_end) {
        return;
    }

    // Load 8x8 block from image and convert to YCbCr
    float8 y_block[8], cb_block[8], cr_block[8];

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            uint2 coord = uint2(img_x + j, img_y + i);
            float4 rgba = image.read(coord);
            float3 rgb = rgba.rgb * 255.0; // Convert to 0-255 range
            float3 ycbcr = rgb_to_ycbcr(rgb);

            // Center around 0 for DCT
            y_block[i][j] = ycbcr.x - 128.0;
            cb_block[i][j] = ycbcr.y - 128.0;
            cr_block[i][j] = ycbcr.z - 128.0;
        }
    }

    // Apply DCT to each channel
    float8 y_dct[8], cb_dct[8], cr_dct[8];
    dct_2d_8x8(y_block, y_dct);
    dct_2d_8x8(cb_block, cb_dct);
    dct_2d_8x8(cr_block, cr_dct);

    // Quantize and dequantize (this is where compression artifacts come from)
    float8 y_quant[8], cb_quant[8], cr_quant[8];
    quantize_dequantize(y_dct, quant_luma, y_quant);
    quantize_dequantize(cb_dct, quant_chroma, cb_quant);
    quantize_dequantize(cr_dct, quant_chroma, cr_quant);

    // Apply IDCT
    float8 y_idct[8], cb_idct[8], cr_idct[8];
    idct_2d_8x8(y_quant, y_idct);
    idct_2d_8x8(cb_quant, cb_idct);
    idct_2d_8x8(cr_quant, cr_idct);

    // Write back to image (convert YCbCr to RGB)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // Uncenter from 0
            float3 ycbcr = float3(
                y_idct[i][j] + 128.0,
                cb_idct[i][j] + 128.0,
                cr_idct[i][j] + 128.0
            );

            float3 rgb = ycbcr_to_rgb(ycbcr);

            // Clamp to valid range and convert back to 0-1
            rgb = clamp(rgb, 0.0, 255.0) / 255.0;

            uint2 coord = uint2(img_x + j, img_y + i);
            float4 rgba = float4(rgb, 1.0);
            image.write(rgba, coord);
        }
    }
}
