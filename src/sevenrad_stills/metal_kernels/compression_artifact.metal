//
// compression_artifact.metal
// GPU-optimized JPEG compression artifact simulation using Metal
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
inline void dct_1d(thread const float *data, thread float *out) {
    for (int k = 0; k < 8; k++) {
        float sum = 0.0;
        for (int n = 0; n < 8; n++) {
            sum += data[n] * cos(PI * k * (2 * n + 1) / 16.0);
        }
        out[k] = sum * (k == 0 ? SQRT_1_8 : SQRT_2_8);
    }
}

// 1D IDCT-II (inverse transform)
inline void idct_1d(thread const float *data, thread float *out) {
    for (int n = 0; n < 8; n++) {
        float sum = data[0] * SQRT_1_8;
        for (int k = 1; k < 8; k++) {
            sum += data[k] * SQRT_2_8 * cos(PI * k * (2 * n + 1) / 16.0);
        }
        out[n] = sum;
    }
}

// 2D DCT using separable transform (rows then columns)
inline void dct_2d_8x8(thread const float block[8][8], thread float out[8][8]) {
    float temp[8][8];

    // DCT on rows
    for (int i = 0; i < 8; i++) {
        dct_1d(block[i], temp[i]);
    }

    // DCT on columns (extract column, DCT, write back)
    for (int j = 0; j < 8; j++) {
        float col[8];
        for (int i = 0; i < 8; i++) {
            col[i] = temp[i][j];
        }

        float col_out[8];
        dct_1d(col, col_out);

        for (int i = 0; i < 8; i++) {
            out[i][j] = col_out[i];
        }
    }
}

// 2D IDCT using separable transform
inline void idct_2d_8x8(thread const float block[8][8], thread float out[8][8]) {
    float temp[8][8];

    // IDCT on rows
    for (int i = 0; i < 8; i++) {
        idct_1d(block[i], temp[i]);
    }

    // IDCT on columns
    for (int j = 0; j < 8; j++) {
        float col[8];
        for (int i = 0; i < 8; i++) {
            col[i] = temp[i][j];
        }

        float col_out[8];
        idct_1d(col, col_out);

        for (int i = 0; i < 8; i++) {
            out[i][j] = col_out[i];
        }
    }
}

// Quantize and dequantize (lossy step)
inline void quantize_dequantize(
    thread const float block[8][8],
    constant const float *quant_matrix,
    thread float out[8][8]
) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;
            float quantized = round(block[i][j] / quant_matrix[idx]);
            out[i][j] = quantized * quant_matrix[idx];
        }
    }
}

// Main compression artifact kernel
kernel void apply_compression_artifacts(
    texture2d<float, access::read_write> image [[texture(0)]],
    constant TileDescriptor *tiles [[buffer(0)]],
    constant float *quant_luma [[buffer(1)]],
    constant float *quant_chroma [[buffer(2)]],
    constant uint &num_tiles [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint tile_idx = gid.x;
    uint block_y = gid.y;
    uint block_x = gid.z;

    // Bounds check
    if (tile_idx >= num_tiles) {
        return;
    }

    TileDescriptor tile = tiles[tile_idx];
    uint img_y = tile.y_start + block_y * BLOCK_SIZE;
    uint img_x = tile.x_start + block_x * BLOCK_SIZE;

    // Check if block is fully within tile
    if (img_y + BLOCK_SIZE > tile.y_end || img_x + BLOCK_SIZE > tile.x_end) {
        return;
    }

    // Load 8x8 block from image and convert to YCbCr
    float y_block[8][8], cb_block[8][8], cr_block[8][8];

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            uint2 coord = uint2(img_x + j, img_y + i);
            float4 rgba = image.read(coord);
            float3 rgb = rgba.rgb * 255.0;
            float3 ycbcr = rgb_to_ycbcr(rgb);

            // Center around 0 for DCT
            y_block[i][j] = ycbcr.x - 128.0;
            cb_block[i][j] = ycbcr.y - 128.0;
            cr_block[i][j] = ycbcr.z - 128.0;
        }
    }

    // Apply DCT to each channel
    float y_dct[8][8], cb_dct[8][8], cr_dct[8][8];
    dct_2d_8x8(y_block, y_dct);
    dct_2d_8x8(cb_block, cb_dct);
    dct_2d_8x8(cr_block, cr_dct);

    // Quantize and dequantize
    float y_quant[8][8], cb_quant[8][8], cr_quant[8][8];
    quantize_dequantize(y_dct, quant_luma, y_quant);
    quantize_dequantize(cb_dct, quant_chroma, cb_quant);
    quantize_dequantize(cr_dct, quant_chroma, cr_quant);

    // Apply IDCT
    float y_idct[8][8], cb_idct[8][8], cr_idct[8][8];
    idct_2d_8x8(y_quant, y_idct);
    idct_2d_8x8(cb_quant, cb_idct);
    idct_2d_8x8(cr_quant, cr_idct);

    // Write back to image
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // Uncenter from 0
            float3 ycbcr = float3(
                y_idct[i][j] + 128.0,
                cb_idct[i][j] + 128.0,
                cr_idct[i][j] + 128.0
            );

            float3 rgb = ycbcr_to_rgb(ycbcr);
            rgb = clamp(rgb / 255.0, 0.0, 1.0);

            uint2 coord = uint2(img_x + j, img_y + i);
            float4 rgba = image.read(coord);
            image.write(float4(rgb, rgba.a), coord);
        }
    }
}
