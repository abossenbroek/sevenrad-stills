/**
 * @file sr_utils.h
 * @brief Common utility macros and functions for Sevenrad Max externals.
 */

#ifndef SR_UTILS_H
#define SR_UTILS_H

#include <stdint.h>
#include <stdlib.h>

/* Clamp a value between min and max */
#define SR_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

/* Min/Max macros */
#define SR_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SR_MAX(a, b) ((a) > (b) ? (a) : (b))

/* Linear interpolation */
#define SR_LERP(a, b, t) ((a) + (t) * ((b) - (a)))

/* RGB channel permutation indices */
typedef enum {
    SR_PERM_RGB = 0,  /* No change */
    SR_PERM_RBG = 1,
    SR_PERM_GRB = 2,
    SR_PERM_GBR = 3,
    SR_PERM_BRG = 4,
    SR_PERM_BGR = 5
} sr_channel_perm_t;

/**
 * @brief Tile structure for band swap and corruption effects.
 */
typedef struct {
    int x;          /* Top-left x coordinate */
    int y;          /* Top-left y coordinate */
    int width;      /* Tile width */
    int height;     /* Tile height */
    int perm;       /* Channel permutation (sr_channel_perm_t) */
} sr_tile_t;

/**
 * @brief Apply channel permutation to RGB values.
 *
 * @param r Pointer to red channel value
 * @param g Pointer to green channel value
 * @param b Pointer to blue channel value
 * @param perm Permutation index (0-5)
 */
static inline void sr_apply_permutation(float* r, float* g, float* b, int perm) {
    float temp_r = *r, temp_g = *g, temp_b = *b;

    switch (perm % 6) {
        case SR_PERM_RGB: /* No change */
            break;
        case SR_PERM_RBG:
            *g = temp_b;
            *b = temp_g;
            break;
        case SR_PERM_GRB:
            *r = temp_g;
            *g = temp_r;
            break;
        case SR_PERM_GBR:
            *r = temp_g;
            *g = temp_b;
            *b = temp_r;
            break;
        case SR_PERM_BRG:
            *r = temp_b;
            *g = temp_r;
            *b = temp_g;
            break;
        case SR_PERM_BGR:
            *r = temp_b;
            *b = temp_r;
            break;
    }
}

/**
 * @brief Convert float [0,1] to 8-bit integer.
 */
static inline uint8_t sr_float_to_u8(float f) {
    int i = (int)(f * 255.0f + 0.5f);
    return (uint8_t)SR_CLAMP(i, 0, 255);
}

/**
 * @brief Convert 8-bit integer to float [0,1].
 */
static inline float sr_u8_to_float(uint8_t u) {
    return (float)u / 255.0f;
}

#endif /* SR_UTILS_H */
