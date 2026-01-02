"""
Tests for PCG RNG implementation consistency.

This test suite validates that the PCG (Permuted Congruential Generator)
hash-based random number generation produces consistent, deterministic results.

The tests document expected values for known inputs, ensuring compatibility
between Python/NumPy reference implementations and GenExpr implementations
in Max/MSP externals.

Reference Implementation:
    Located at src/sevenrad_stills/operations/taichi_kernels/random.py

Expected Behavior:
    - pcg_hash produces deterministic hashes from seed values
    - rand_float produces uniform values in [0, 1) range
    - Same inputs always produce same outputs (reproducibility)
    - NumPy and Taichi implementations produce identical results

Test Cases Document:
    1. Known hash values for specific seeds
    2. Coordinate-based random float generation
    3. Determinism across multiple calls
    4. Range validation for random floats
    5. Channel-independent random values

These test values serve as the reference for implementing identical
PCG RNG in GenExpr/Max externals to ensure cross-platform consistency.

"""
# ruff: noqa: S101 PLR2004

import math

import pytest

# Import PCG RNG functions from sevenrad_stills package
from sevenrad_stills.operations.taichi_kernels.random import (
    COORD_PRIME_C,
    COORD_PRIME_X,
    COORD_PRIME_Y,
    EPSILON,
    PCG_FACTOR,
    PCG_INC,
    PCG_MULT,
    TWO_PI,
    UINT32_MAX_F,
    pcg_hash_numpy,
    rand_float_numpy,
    rand_gaussian_numpy,
)


class TestPCGConstants:
    """
    Test that PCG constants are correctly defined.

    These constants must match exactly in any GenExpr implementation.
    """

    def test_pcg_mult_constant(self) -> None:
        """Test PCG_MULT constant value."""
        assert PCG_MULT == 747796405
        assert isinstance(PCG_MULT, int)

    def test_pcg_inc_constant(self) -> None:
        """Test PCG_INC constant value."""
        assert PCG_INC == 2891336453
        assert isinstance(PCG_INC, int)
        # Verify it exceeds i32 max but fits in u32
        assert PCG_INC > 2147483647  # i32 max
        assert PCG_INC < 4294967296  # u32 max

    def test_pcg_factor_constant(self) -> None:
        """Test PCG_FACTOR constant value."""
        assert PCG_FACTOR == 277803737
        assert isinstance(PCG_FACTOR, int)

    def test_coord_prime_x_constant(self) -> None:
        """Test COORD_PRIME_X constant value."""
        assert COORD_PRIME_X == 374761393
        assert isinstance(COORD_PRIME_X, int)

    def test_coord_prime_y_constant(self) -> None:
        """Test COORD_PRIME_Y constant value."""
        assert COORD_PRIME_Y == 668265263
        assert isinstance(COORD_PRIME_Y, int)

    def test_coord_prime_c_constant(self) -> None:
        """Test COORD_PRIME_C constant value."""
        assert COORD_PRIME_C == 73856093
        assert isinstance(COORD_PRIME_C, int)

    def test_uint32_max_f_constant(self) -> None:
        """Test UINT32_MAX_F normalization constant."""
        assert UINT32_MAX_F == 4294967296.0
        assert UINT32_MAX_F == 2.0**32

    def test_two_pi_constant(self) -> None:
        """Test TWO_PI constant for Box-Muller transform."""
        assert 2.0 * math.pi == TWO_PI
        assert abs(TWO_PI - 6.283185307179586) < 1e-15

    def test_epsilon_constant(self) -> None:
        """Test EPSILON constant for numerical stability."""
        assert EPSILON == 1e-10


class TestPCGHashKnownValues:
    """
    Test pcg_hash with known input/output pairs.

    These are REFERENCE VALUES for GenExpr implementation.
    Any GenExpr PCG hash must produce identical outputs.
    """

    def test_pcg_hash_zero(self) -> None:
        """
        Test pcg_hash(0) produces known value.

        GenExpr implementation must produce: 129708002
        """
        result = pcg_hash_numpy(0)
        assert result == 129708002
        assert isinstance(result, int)

    def test_pcg_hash_one(self) -> None:
        """
        Test pcg_hash(1) produces known value.

        GenExpr implementation must produce: 2831084092
        """
        result = pcg_hash_numpy(1)
        assert result == 2831084092

    def test_pcg_hash_12345(self) -> None:
        """
        Test pcg_hash(12345) produces known value.

        GenExpr implementation must produce: 4099845390
        """
        result = pcg_hash_numpy(12345)
        assert result == 4099845390

    def test_pcg_hash_42(self) -> None:
        """
        Test pcg_hash(42) produces known value.

        GenExpr implementation must produce: 1223963391
        """
        result = pcg_hash_numpy(42)
        assert result == 1223963391

    def test_pcg_hash_large_value(self) -> None:
        """
        Test pcg_hash with large seed value.

        GenExpr implementation must produce: 16200633
        """
        result = pcg_hash_numpy(999999999)
        assert result == 16200633

    def test_pcg_hash_max_u32(self) -> None:
        """
        Test pcg_hash with maximum u32 value.

        GenExpr implementation must produce: 3861530882
        """
        result = pcg_hash_numpy(0xFFFFFFFF)
        assert result == 3861530882

    def test_pcg_hash_power_of_two(self) -> None:
        """Test pcg_hash with power of 2 values."""
        # pcg_hash(256) = 3397515543
        assert pcg_hash_numpy(256) == 3397515543

        # pcg_hash(1024) = 4173518760
        assert pcg_hash_numpy(1024) == 4173518760

        # pcg_hash(65536) = 688544357
        assert pcg_hash_numpy(65536) == 688544357

    def test_pcg_hash_sequential_values(self) -> None:
        """
        Test pcg_hash with sequential inputs.

        Documents that sequential inputs produce well-distributed outputs.
        """
        results = [pcg_hash_numpy(i) for i in range(10)]

        # Expected values for seeds 0-9
        expected = [
            129708002,  # hash(0)
            2831084092,  # hash(1)
            2055130248,  # hash(2)
            2131687100,  # hash(3)
            678955108,  # hash(4)
            2161170183,  # hash(5)
            4048597412,  # hash(6)
            2120684060,  # hash(7)
            460041413,  # hash(8)
            1301776676,  # hash(9)
        ]

        assert results == expected


class TestPCGHashProperties:
    """Test mathematical properties of pcg_hash."""

    def test_pcg_hash_deterministic(self) -> None:
        """Test that pcg_hash is deterministic."""
        seed = 12345
        result1 = pcg_hash_numpy(seed)
        result2 = pcg_hash_numpy(seed)
        result3 = pcg_hash_numpy(seed)

        assert result1 == result2 == result3

    def test_pcg_hash_output_range(self) -> None:
        """Test that pcg_hash outputs are valid u32 values."""
        test_seeds = [0, 1, 42, 12345, 999999, 0xFFFFFFFF]

        for seed in test_seeds:
            result = pcg_hash_numpy(seed)
            assert 0 <= result < 0x100000000  # Valid u32 range

    def test_pcg_hash_different_inputs_different_outputs(self) -> None:
        """Test that different seeds produce different hashes."""
        # Generate hashes for first 100 seeds
        hashes = [pcg_hash_numpy(i) for i in range(100)]

        # All should be unique (statistical property of good hash)
        assert len(set(hashes)) == len(hashes)

    def test_pcg_hash_wrapping_behavior(self) -> None:
        """Test that pcg_hash handles 32-bit overflow correctly."""
        # Values that exceed i32 max but fit in u32
        large_seed = 3000000000
        result = pcg_hash_numpy(large_seed)

        # Should produce valid u32 output
        assert 0 <= result < 0x100000000
        assert result == 2611329461


class TestRandFloatKnownValues:
    """
    Test rand_float with known coordinate/seed combinations.

    These are REFERENCE VALUES for GenExpr implementation.
    """

    def test_rand_float_zero_zero_zero(self) -> None:
        """
        Test rand_float(0, 0, 0) produces known value.

        GenExpr implementation must produce: 0.03019999759271741
        """
        result = rand_float_numpy(0, 0, 0)
        assert isinstance(result, float)
        # Use high precision comparison
        assert abs(result - 0.03019999759271741) < 1e-10

    def test_rand_float_100_200_42(self) -> None:
        """
        Test rand_float(100, 200, 42) produces known value.

        GenExpr implementation must produce: 0.55267843487672508
        """
        result = rand_float_numpy(100, 200, 42)
        assert abs(result - 0.55267843487672508) < 1e-10

    def test_rand_float_1_1_1(self) -> None:
        """
        Test rand_float(1, 1, 1) produces known value.

        GenExpr implementation must produce: 0.92225170740857720
        """
        result = rand_float_numpy(1, 1, 1)
        assert abs(result - 0.92225170740857720) < 1e-10

    def test_rand_float_common_coordinates(self) -> None:
        """Test rand_float with commonly used coordinate values."""
        # (10, 20, seed=0)
        result = rand_float_numpy(10, 20, 0)
        assert abs(result - 0.18934893771074712) < 1e-10

        # (50, 50, seed=100)
        result = rand_float_numpy(50, 50, 100)
        assert abs(result - 0.12730784993618727) < 1e-10

        # (256, 256, seed=42)
        result = rand_float_numpy(256, 256, 42)
        assert abs(result - 0.09737738990224898) < 1e-10

    def test_rand_float_negative_coordinates(self) -> None:
        """
        Test rand_float handles negative coordinates.

        Negative coordinates get wrapped via u32 conversion.
        """
        # (-1, -1, 0)
        result = rand_float_numpy(-1, -1, 0)
        assert 0.0 <= result < 1.0
        assert abs(result - 0.70282620494253933) < 1e-10

        # (-10, 20, 42)
        result = rand_float_numpy(-10, 20, 42)
        assert 0.0 <= result < 1.0
        assert abs(result - 0.16783784516155720) < 1e-10


class TestRandFloatProperties:
    """Test mathematical properties of rand_float."""

    def test_rand_float_deterministic(self) -> None:
        """Test that rand_float is deterministic."""
        x, y, seed = 100, 200, 42

        result1 = rand_float_numpy(x, y, seed)
        result2 = rand_float_numpy(x, y, seed)
        result3 = rand_float_numpy(x, y, seed)

        assert result1 == result2 == result3

    def test_rand_float_output_range(self) -> None:
        """Test that rand_float produces values in [0, 1) range."""
        test_cases = [
            (0, 0, 0),
            (1, 1, 1),
            (100, 200, 42),
            (999, 999, 999),
            (-1, -1, -1),
            (256, 256, 0),
        ]

        for x, y, seed in test_cases:
            result = rand_float_numpy(x, y, seed)
            assert 0.0 <= result < 1.0

    def test_rand_float_never_equals_one(self) -> None:
        """Test that rand_float never returns exactly 1.0."""
        # Test many combinations
        for x in range(0, 100, 10):
            for y in range(0, 100, 10):
                for seed in [0, 42, 12345]:
                    result = rand_float_numpy(x, y, seed)
                    assert result < 1.0

    def test_rand_float_different_coords_different_outputs(self) -> None:
        """Test that different coordinates produce different values."""
        seed = 42

        # Generate values for 10x10 grid
        values = []
        for x in range(10):
            for y in range(10):
                values.append(rand_float_numpy(x, y, seed))

        # All should be unique (statistical property)
        assert len(set(values)) == len(values)

    def test_rand_float_different_seeds_different_outputs(self) -> None:
        """Test that different seeds produce different values."""
        x, y = 100, 200

        values = [rand_float_numpy(x, y, seed) for seed in range(100)]

        # All should be unique
        assert len(set(values)) == len(values)

    def test_rand_float_uniform_distribution_property(self) -> None:
        """
        Test that rand_float appears uniformly distributed.

        Not a rigorous statistical test, but checks basic distribution.
        """
        # Generate many samples
        samples = []
        for x in range(100):
            for y in range(100):
                samples.append(rand_float_numpy(x, y, seed=42))

        # Check mean is approximately 0.5
        mean = sum(samples) / len(samples)
        assert 0.45 < mean < 0.55

        # Check values span the range
        assert min(samples) < 0.1
        assert max(samples) > 0.9


class TestRandFloatCoordinateMixing:
    """
    Test that coordinate mixing function works correctly.

    The mixing formula is: combined = x * COORD_PRIME_X + y * COORD_PRIME_Y + seed
    """

    def test_coordinate_mixing_formula(self) -> None:
        """Test that coordinate mixing matches expected formula."""
        x, y, seed = 10, 20, 42

        # Calculate expected combined value
        combined = (x * COORD_PRIME_X + y * COORD_PRIME_Y + seed) & 0xFFFFFFFF
        expected_combined = (10 * 374761393 + 20 * 668265263 + 42) & 0xFFFFFFFF

        assert combined == expected_combined

        # Verify this produces expected hash
        h = pcg_hash_numpy(combined)
        expected_value = h / UINT32_MAX_F
        assert abs(rand_float_numpy(x, y, seed) - expected_value) < 1e-15

    def test_coordinate_primes_minimize_correlation(self) -> None:
        """
        Test that coordinate primes produce uncorrelated results.

        Adjacent coordinates should produce very different values.
        """
        seed = 0

        # Compare adjacent pixels
        v1 = rand_float_numpy(10, 10, seed)
        v2 = rand_float_numpy(11, 10, seed)
        v3 = rand_float_numpy(10, 11, seed)

        # Should be significantly different (not close)
        assert abs(v1 - v2) > 0.01
        assert abs(v1 - v3) > 0.01

    def test_diagonal_coordinates_uncorrelated(self) -> None:
        """Test that diagonal coordinates produce uncorrelated values."""
        seed = 42

        values = [rand_float_numpy(i, i, seed) for i in range(10)]

        # Values should be uncorrelated (no pattern)
        # Simple check: consecutive values should differ significantly
        for i in range(len(values) - 1):
            assert abs(values[i] - values[i + 1]) > 0.01


class TestRandGaussianKnownValues:
    """
    Test rand_gaussian with known inputs.

    These are REFERENCE VALUES for GenExpr implementation.
    """

    def test_rand_gaussian_zero_zero_zero(self) -> None:
        """
        Test rand_gaussian(0, 0, 0, sigma=1.0) produces known value.

        This uses Box-Muller transform on two rand_float calls.
        GenExpr implementation must produce: -1.42937331449142424
        """
        result = rand_gaussian_numpy(0, 0, 0, sigma=1.0)
        assert abs(result - (-1.42937331449142424)) < 1e-10

    def test_rand_gaussian_100_200_42(self) -> None:
        """
        Test rand_gaussian(100, 200, 42, sigma=0.5) produces known value.

        GenExpr implementation must produce: 0.18896663642460299
        """
        result = rand_gaussian_numpy(100, 200, 42, sigma=0.5)
        assert abs(result - 0.18896663642460299) < 1e-10

    def test_rand_gaussian_sigma_scaling(self) -> None:
        """Test that sigma parameter scales the output correctly."""
        x, y, seed = 10, 20, 0

        # Generate gaussian with sigma=1.0
        result_sigma1 = rand_gaussian_numpy(x, y, seed, sigma=1.0)

        # Generate gaussian with sigma=2.0
        result_sigma2 = rand_gaussian_numpy(x, y, seed, sigma=2.0)

        # Result should scale linearly with sigma
        assert abs(result_sigma2 - 2.0 * result_sigma1) < 1e-10

    def test_rand_gaussian_uses_two_random_values(self) -> None:
        """
        Test that rand_gaussian uses seed and seed+1 for Box-Muller.

        Box-Muller requires two uniform random values.
        """
        x, y, seed = 50, 50, 100

        # Get the two uniform values used
        u1 = max(rand_float_numpy(x, y, seed), EPSILON)
        u2 = rand_float_numpy(x, y, seed + 1)

        # Calculate expected gaussian using Box-Muller
        mag = math.sqrt(-2.0 * math.log(u1))
        expected = mag * math.cos(TWO_PI * u2) * 1.0  # sigma=1.0

        result = rand_gaussian_numpy(x, y, seed, sigma=1.0)

        assert abs(result - expected) < 1e-10


class TestRandGaussianProperties:
    """Test mathematical properties of rand_gaussian."""

    def test_rand_gaussian_deterministic(self) -> None:
        """Test that rand_gaussian is deterministic."""
        x, y, seed, sigma = 100, 200, 42, 0.5

        result1 = rand_gaussian_numpy(x, y, seed, sigma)
        result2 = rand_gaussian_numpy(x, y, seed, sigma)
        result3 = rand_gaussian_numpy(x, y, seed, sigma)

        assert result1 == result2 == result3

    def test_rand_gaussian_can_be_negative(self) -> None:
        """
        Test that gaussian can produce negative values.

        Unlike uniform [0,1), gaussian is centered at 0.
        """
        # Find a case that produces negative value
        result = rand_gaussian_numpy(0, 0, 0, sigma=1.0)
        assert result < 0  # This specific case produces -1.669

    def test_rand_gaussian_distribution_property(self) -> None:
        """
        Test that rand_gaussian appears normally distributed.

        Not rigorous, but checks basic distribution properties.
        """
        sigma = 1.0
        samples = []

        for x in range(50):
            for y in range(50):
                samples.append(rand_gaussian_numpy(x, y, seed=42, sigma=sigma))

        # Mean should be approximately 0
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.2  # Allow some variance

        # Standard deviation should be approximately sigma
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std = math.sqrt(variance)
        assert 0.8 < std < 1.2  # Within 20% of expected sigma

    def test_rand_gaussian_zero_sigma(self) -> None:
        """Test that sigma=0 produces zero output."""
        result = rand_gaussian_numpy(100, 200, 42, sigma=0.0)
        assert result == 0.0

    def test_rand_gaussian_epsilon_clamping(self) -> None:
        """
        Test that u1 is clamped to avoid log(0).

        The Box-Muller transform uses log(u1), so u1 must be > 0.
        """
        # This is tested implicitly - if not clamped, would raise ValueError
        # Just verify no errors for many samples
        for x in range(20):
            for y in range(20):
                for seed in range(20):
                    result = rand_gaussian_numpy(x, y, seed, sigma=1.0)
                    assert isinstance(result, float)
                    assert not math.isnan(result)
                    assert not math.isinf(result)


class TestCrossImplementationConsistency:
    """
    Test cases specifically for verifying GenExpr/Max implementation.

    These tests document the exact values that a GenExpr implementation
    must produce to be considered correct.
    """

    def test_genexpr_reference_case_1(self) -> None:
        """
        Reference case 1: Origin pixel, no seed.

        GenExpr must produce these exact values:
        - pcg_hash(0) = 129708002
        - rand_float(0, 0, 0) = 0.03019999759271741
        """
        assert pcg_hash_numpy(0) == 129708002
        assert abs(rand_float_numpy(0, 0, 0) - 0.03019999759271741) < 1e-10

    def test_genexpr_reference_case_2(self) -> None:
        """
        Reference case 2: Standard test coordinates.

        GenExpr must produce these exact values:
        - rand_float(100, 200, 42) = 0.55267843487672508
        - rand_gaussian(100, 200, 42, 0.5) = 0.18896663642460299
        """
        assert abs(rand_float_numpy(100, 200, 42) - 0.55267843487672508) < 1e-10
        assert abs(rand_gaussian_numpy(100, 200, 42, 0.5) - 0.18896663642460299) < 1e-10

    def test_genexpr_reference_case_3(self) -> None:
        """
        Reference case 3: Common seed value 12345.

        GenExpr must produce these exact values:
        - pcg_hash(12345) = 4099845390
        - This verifies the PCG algorithm implementation
        """
        assert pcg_hash_numpy(12345) == 4099845390

    def test_genexpr_reference_batch_hashes(self) -> None:
        """
        Reference batch: First 10 hash values.

        GenExpr implementation can verify against this sequence:
        """
        expected_hashes = [
            129708002,  # hash(0)
            2831084092,  # hash(1)
            2055130248,  # hash(2)
            2131687100,  # hash(3)
            678955108,  # hash(4)
            2161170183,  # hash(5)
            4048597412,  # hash(6)
            2120684060,  # hash(7)
            460041413,  # hash(8)
            1301776676,  # hash(9)
        ]

        for seed, expected in enumerate(expected_hashes):
            assert pcg_hash_numpy(seed) == expected

    def test_genexpr_reference_batch_floats(self) -> None:
        """
        Reference batch: rand_float for 3x3 grid, seed=0.

        GenExpr can verify spatial distribution with this grid:
        """
        expected_grid = [
            [0.03019999759271741, 0.71303768409416080, 0.26849375083111227],
            [0.71739902347326279, 0.58384185470640659, 0.36935825971886516],
            [0.44726293534040451, 0.13895388599485159, 0.13749840832315385],
        ]

        for x in range(3):
            for y in range(3):
                result = rand_float_numpy(x, y, seed=0)
                expected = expected_grid[x][y]
                assert abs(result - expected) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pcg_hash_overflow_handling(self) -> None:
        """Test that overflow is handled via u32 wrapping."""
        # Large seed that would overflow i32
        large_seed = 0xFFFFFFFF
        result = pcg_hash_numpy(large_seed)

        # Should produce valid u32
        assert 0 <= result < 0x100000000
        assert result == 3861530882

    def test_rand_float_extreme_coordinates(self) -> None:
        """Test rand_float with very large coordinates."""
        # Large coordinates
        result = rand_float_numpy(1000000, 2000000, 0)
        assert 0.0 <= result < 1.0

        # Very large coordinates
        result = rand_float_numpy(10000000, 20000000, 12345)
        assert 0.0 <= result < 1.0

    def test_rand_gaussian_large_sigma(self) -> None:
        """Test rand_gaussian with large sigma values."""
        result = rand_gaussian_numpy(100, 200, 42, sigma=100.0)

        # Should be proportional to sigma
        result_small = rand_gaussian_numpy(100, 200, 42, sigma=1.0)
        assert abs(result - 100.0 * result_small) < 1e-8

    def test_negative_seed_handling(self) -> None:
        """
        Test that negative seeds are handled correctly.

        Negative seeds get converted to u32 via wrapping.
        """
        # Negative seed
        result = rand_float_numpy(0, 0, -1)
        assert 0.0 <= result < 1.0

        # -1 as u32 is 0xFFFFFFFF
        seed_as_u32 = (-1) & 0xFFFFFFFF
        assert seed_as_u32 == 0xFFFFFFFF

    def test_consistent_across_sign_wrapping(self) -> None:
        """Test that negative coordinates wrap consistently."""
        # Test that -1 wraps to same as 0xFFFFFFFF
        result_neg = rand_float_numpy(-1, -1, 0)
        assert 0.0 <= result_neg < 1.0

        # Should be deterministic
        assert rand_float_numpy(-1, -1, 0) == result_neg


class TestNumericPrecision:
    """Test numeric precision and floating-point behavior."""

    def test_division_precision(self) -> None:
        """Test that division by UINT32_MAX_F maintains precision."""
        # Maximum hash value
        h = 0xFFFFFFFF
        result = h / UINT32_MAX_F

        # Should be very close to 1.0 but less than 1.0
        assert result < 1.0
        assert result > 0.999999

    def test_rand_float_precision_consistency(self) -> None:
        """Test that rand_float maintains precision across platforms."""
        # Known value that tests floating-point precision
        result = rand_float_numpy(123, 456, 789)

        # Should have at least 7 decimal places of precision
        # (verified against reference implementation)
        assert abs(result - 0.12873050663620234) < 1e-10

    def test_box_muller_precision(self) -> None:
        """Test that Box-Muller transform maintains precision."""
        result = rand_gaussian_numpy(123, 456, 789, sigma=1.0)

        # Should have high precision (verified against reference)
        assert abs(result - 0.97571254864898427) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
