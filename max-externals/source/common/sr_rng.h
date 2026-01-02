/**
 * @file sr_rng.h
 * @brief PCG-based random number generation utilities for Sevenrad Max externals.
 *
 * This header provides C implementations of the PCG (Permuted Congruential Generator)
 * hash functions used throughout the Sevenrad effects. These functions produce
 * deterministic, reproducible random values based on pixel coordinates and seed.
 *
 * The same PCG algorithm is implemented in:
 * - GenExpr shaders (inline)
 * - This C header (for CPU externals)
 * - Python/Taichi (reference implementation)
 *
 * All implementations produce identical results for the same inputs.
 */

#ifndef SR_RNG_H
#define SR_RNG_H

#include <stdint.h>
#include <math.h>

/* PCG constants for high-quality hashing */
#define SR_PCG_MULT      747796405u
#define SR_PCG_INC       2891336453u
#define SR_PCG_FACTOR    277803737u

/* Coordinate mixing primes (minimize correlation) */
#define SR_COORD_PRIME_X 374761393u
#define SR_COORD_PRIME_Y 668265263u
#define SR_COORD_PRIME_C 73856093u

/* Normalization constant: 2^32 as float */
#define SR_UINT32_MAX_F  4294967296.0

/* Mathematical constants */
#define SR_TWO_PI        6.28318530717958647692
#define SR_EPSILON       1e-10

/**
 * @brief PCG-derived hash function for generating pseudo-random integers.
 *
 * Uses the PCG algorithm's output permutation to produce well-distributed
 * hash values with better statistical properties than simpler hashes.
 *
 * @param input_seed Input seed value to hash
 * @return Hashed 32-bit unsigned integer
 */
static inline uint32_t sr_pcg_hash(uint32_t input_seed) {
    uint32_t state = input_seed * SR_PCG_MULT + SR_PCG_INC;
    uint32_t word = ((state >> ((state >> 28) + 4)) ^ state) * SR_PCG_FACTOR;
    return (word >> 22) ^ word;
}

/**
 * @brief Generate deterministic random float in [0, 1) based on position and seed.
 *
 * Produces a uniform random value that is reproducible - calling with the same
 * (x, y, seed) always returns the same value.
 *
 * @param x Pixel x-coordinate (column index)
 * @param y Pixel y-coordinate (row index)
 * @param seed Random seed value
 * @return Uniform random float in range [0.0, 1.0)
 */
static inline float sr_rand_float(int x, int y, int seed) {
    uint32_t combined = (uint32_t)(x * SR_COORD_PRIME_X +
                                    y * SR_COORD_PRIME_Y +
                                    seed);
    uint32_t h = sr_pcg_hash(combined);
    return (float)h / (float)SR_UINT32_MAX_F;
}

/**
 * @brief Generate deterministic random float for specific channel.
 *
 * Similar to sr_rand_float but includes channel index for generating
 * independent random values per color channel.
 *
 * @param x Pixel x-coordinate
 * @param y Pixel y-coordinate
 * @param channel Channel index (0=R, 1=G, 2=B, 3=A)
 * @param seed Random seed value
 * @return Uniform random float in range [0.0, 1.0)
 */
static inline float sr_rand_float_channel(int x, int y, int channel, int seed) {
    uint32_t combined = (uint32_t)(x * SR_COORD_PRIME_X +
                                    y * SR_COORD_PRIME_Y +
                                    channel * SR_COORD_PRIME_C +
                                    seed);
    uint32_t h = sr_pcg_hash(combined);
    return (float)h / (float)SR_UINT32_MAX_F;
}

/**
 * @brief Generate Gaussian (normal) distributed random value using Box-Muller.
 *
 * Produces normally distributed values with mean=0 and given sigma.
 *
 * @param x Pixel x-coordinate
 * @param y Pixel y-coordinate
 * @param seed Random seed value
 * @param sigma Standard deviation of the Gaussian distribution
 * @return Gaussian random value with mean=0, std=sigma
 */
static inline float sr_rand_gaussian(int x, int y, int seed, float sigma) {
    float u1 = sr_rand_float(x, y, seed);
    float u2 = sr_rand_float(x, y, seed + 1);

    /* Avoid log(0) */
    if (u1 < SR_EPSILON) u1 = (float)SR_EPSILON;

    /* Box-Muller transform */
    float mag = sqrtf(-2.0f * logf(u1));
    float z = mag * cosf((float)SR_TWO_PI * u2);

    return z * sigma;
}

/**
 * @brief Generate Gaussian random value for specific channel.
 *
 * @param x Pixel x-coordinate
 * @param y Pixel y-coordinate
 * @param channel Channel index
 * @param seed Random seed value
 * @param sigma Standard deviation
 * @return Gaussian random value
 */
static inline float sr_rand_gaussian_channel(int x, int y, int channel, int seed, float sigma) {
    float u1 = sr_rand_float_channel(x, y, channel, seed);
    float u2 = sr_rand_float_channel(x, y, channel, seed + 12345);

    if (u1 < SR_EPSILON) u1 = (float)SR_EPSILON;

    float mag = sqrtf(-2.0f * logf(u1));
    float z = mag * cosf((float)SR_TWO_PI * u2);

    return z * sigma;
}

#endif /* SR_RNG_H */
