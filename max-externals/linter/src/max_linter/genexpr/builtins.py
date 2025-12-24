"""GenExpr built-in functions and variables registry.

This module provides comprehensive type information for GenExpr's built-in
functions, variables, and common user-declared identifiers. Used by validators
and type checkers to distinguish between undefined variables and valid GenExpr
constructs.

References:
    - GenExpr documentation: Max SDK documentation
    - clangd.py: Original source of built-in declarations
"""

from __future__ import annotations

# Built-in variables with their types
# These are always available in GenExpr shaders without declaration
BUILTIN_VARIABLES: dict[str, str] = {
    # Input textures (vec4 RGBA)
    "in1": "vec4",
    "in2": "vec4",
    "in3": "vec4",
    "in4": "vec4",
    # Output textures (vec4 RGBA)
    "out1": "vec4",
    "out2": "vec4",
    "out3": "vec4",
    "out4": "vec4",
    "out": "vec4",  # Alias for out1
    # Coordinate variables (vec2)
    "norm": "vec2",  # Normalized coordinates [0,1]
    "cell": "vec2",  # Pixel coordinates
    "dim": "vec2",  # Texture dimensions
    "snorm": "vec2",  # Signed normalized coordinates [-1,1]
}

# Built-in functions with (min_args, max_args, return_type)
# GenExpr supports function overloading, so some functions accept variable args
BUILTIN_FUNCTIONS: dict[str, tuple[int, int, str]] = {
    # Texture sampling
    "sample": (2, 3, "vec4"),  # sample(input, coord) or sample(input, coord, boundary)
    "nearest": (
        2,
        3,
        "vec4",
    ),  # nearest(input, coord) or nearest(input, coord, boundary)
    # Vector construction
    "vec": (2, 4, "vec"),  # vec(r,g), vec(r,g,b), or vec(r,g,b,a) - return type varies
    "swiz": (2, 4, "vec"),  # Swizzle components
    # Parameters
    "param": (1, 3, "float"),  # param(name, default, min, max)
    "Param": (1, 3, "float"),  # Capital P variant
    # Buffer operations
    "poke": (3, 3, "void"),  # poke(buffer, coord, value)
    "peek": (2, 2, "vec4"),  # peek(buffer, coord)
    "splat": (2, 2, "void"),  # splat(buffer, value)
    # Numeric safety
    "fixdenorm": (1, 1, "float"),  # Fix denormalized numbers
    "fixnan": (1, 1, "float"),  # Replace NaN with 0
    "isnan": (1, 1, "int"),  # Test for NaN
    "isinf": (1, 1, "int"),  # Test for infinity
    # Range operations
    "foldback": (3, 3, "float"),  # foldback(x, lo, hi)
    "fold": (3, 3, "float"),  # fold(x, lo, hi)
    "wrap": (3, 3, "float"),  # wrap(x, lo, hi)
    "mirror": (2, 2, "float"),  # mirror(x, period)
    "scale": (5, 5, "float"),  # scale(x, inlo, inhi, outlo, outhi)
    # Signal processing
    "dcblock": (1, 1, "float"),  # DC blocking filter
    "latch": (2, 2, "float"),  # Sample and hold
    "interp": (3, 3, "float"),  # Linear interpolation
    "lookup": (2, 2, "vec4"),  # Lookup table
    # MIDI/frequency conversion
    "mtof": (1, 1, "float"),  # MIDI to frequency
    "ftom": (1, 1, "float"),  # Frequency to MIDI
    # Basic math functions
    "abs": (1, 1, "float"),  # Absolute value
    "ceil": (1, 1, "float"),  # Round up
    "floor": (1, 1, "float"),  # Round down
    "round": (1, 1, "float"),  # Round to nearest
    "trunc": (1, 1, "float"),  # Truncate to integer
    "fract": (1, 1, "float"),  # Fractional part
    "sign": (1, 1, "float"),  # Sign (-1, 0, 1)
    "mod": (2, 2, "float"),  # Modulo (Euclidean)
    "fmod": (2, 2, "float"),  # Floating-point remainder
    # Trigonometric functions
    "sin": (1, 1, "float"),
    "cos": (1, 1, "float"),
    "tan": (1, 1, "float"),
    "asin": (1, 1, "float"),
    "acos": (1, 1, "float"),
    "atan": (1, 1, "float"),
    "atan2": (2, 2, "float"),
    # Hyperbolic functions
    "sinh": (1, 1, "float"),
    "cosh": (1, 1, "float"),
    "tanh": (1, 1, "float"),
    "asinh": (1, 1, "float"),
    "acosh": (1, 1, "float"),
    "atanh": (1, 1, "float"),
    # Exponential and logarithmic
    "exp": (1, 1, "float"),
    "exp2": (1, 1, "float"),  # 2^x
    "log": (1, 1, "float"),  # Natural log
    "log2": (1, 1, "float"),  # Base-2 log
    "log10": (1, 1, "float"),  # Base-10 log
    "pow": (2, 2, "float"),  # x^y
    "sqrt": (1, 1, "float"),  # Square root
    "rsqrt": (1, 1, "float"),  # Reciprocal square root (1/sqrt(x))
    "cbrt": (1, 1, "float"),  # Cube root
    "hypot": (2, 2, "float"),  # Hypotenuse sqrt(x^2 + y^2)
    # Range and interpolation
    "min": (2, 2, "float"),
    "max": (2, 2, "float"),
    "clamp": (3, 3, "float"),  # clamp(x, lo, hi)
    "mix": (3, 3, "float"),  # mix(a, b, t) - linear interpolation
    "lerp": (3, 3, "float"),  # Alias for mix
    "step": (2, 2, "float"),  # step(edge, x) - 0 if x < edge else 1
    "smoothstep": (3, 3, "float"),  # smoothstep(e0, e1, x) - smooth Hermite
    # Vector operations
    "length": (1, 1, "float"),  # Vector length
    "distance": (2, 2, "float"),  # Distance between points
    "dot": (2, 2, "float"),  # Dot product
    "cross": (2, 2, "vec3"),  # Cross product (returns vec3)
    "normalize": (1, 1, "vec"),  # Normalize to unit length
    "reflect": (2, 2, "vec"),  # Reflect vector
    "refract": (3, 3, "vec"),  # Refract vector
    # Noise functions
    "noise": (1, 4, "float"),  # Perlin noise (1-4D)
    "pnoise": (1, 4, "float"),  # Periodic Perlin noise
    "snoise": (1, 4, "float"),  # Simplex noise
    # Type conversions
    "int": (1, 1, "int"),
    "float": (1, 1, "float"),
    "uint": (1, 1, "uint"),
    # Additional vector/type constructors
    "vec2": (2, 2, "vec2"),  # vec2(x, y)
    "vec3": (3, 3, "vec3"),  # vec3(x, y, z)
    "vec4": (4, 4, "vec4"),  # vec4(x, y, z, w)
    # Additional common shader functions
    "degrees": (1, 1, "float"),  # Radians to degrees
    "radians": (1, 1, "float"),  # Degrees to radians
    "inversesqrt": (1, 1, "float"),  # Alias for rsqrt
    "faceforward": (3, 3, "vec"),  # faceforward(N, I, Nref)
    "fma": (3, 3, "float"),  # Fused multiply-add: fma(a, b, c) = a*b + c
    "modf": (1, 1, "float"),  # Extract integer and fractional parts
    "frexp": (1, 1, "float"),  # Extract mantissa and exponent
    "ldexp": (2, 2, "float"),  # x * 2^exp
}

# Common variable names that users frequently declare
# These should not trigger "undefined identifier" warnings
# as they are idiomatic in shader programming
COMMON_VARIABLES: set[str] = {
    # Loop counters
    "i",
    "j",
    "k",
    "n",
    "m",
    # Coordinate names
    "x",
    "y",
    "z",
    "t",
    "u",
    "v",
    "w",
    # Color components
    "r",
    "g",
    "b",
    "a",
    "rgb",
    "rgba",
    # Common identifiers
    "color",
    "result",
    "sum",
    "count",
    "temp",
    "val",
    "value",
    "pixel",
    "sample_color",
    "output",
    "input",
    # Effect-specific common names
    "offset",
    "scale",
    "factor",
    "strength",
    "amount",
    "threshold",
    "delta",
    "diff",
    "blend",
}

# All built-in identifiers (for backward compatibility with clangd.py)
ALL_BUILTINS: set[str] = (
    set(BUILTIN_VARIABLES.keys()) | set(BUILTIN_FUNCTIONS.keys()) | COMMON_VARIABLES
)
