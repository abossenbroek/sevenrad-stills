"""Type definitions for TouchDesigner operator networks."""

from enum import Enum


class OperatorFamily(Enum):
    """TouchDesigner operator families."""

    TOP = "TOP"  # Texture Operators
    CHOP = "CHOP"  # Channel Operators
    SOP = "SOP"  # Surface Operators
    DAT = "DAT"  # Data Operators
    COMP = "COMP"  # Component Operators
    MAT = "MAT"  # Material Operators
    POP = "POP"  # Particle Operators (legacy, but still in some projects)
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> "OperatorFamily":
        """Convert string to OperatorFamily, returning UNKNOWN if not found."""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.UNKNOWN


# Operator families that can connect to each other
# Key: source family, Value: set of allowed target families
COMPATIBLE_CONNECTIONS: dict[OperatorFamily, set[OperatorFamily]] = {
    OperatorFamily.TOP: {OperatorFamily.TOP, OperatorFamily.MAT},
    OperatorFamily.CHOP: {OperatorFamily.CHOP},
    OperatorFamily.SOP: {OperatorFamily.SOP, OperatorFamily.MAT},
    OperatorFamily.DAT: {OperatorFamily.DAT},
    OperatorFamily.COMP: {OperatorFamily.COMP},
    OperatorFamily.MAT: {OperatorFamily.MAT},
    OperatorFamily.POP: {OperatorFamily.POP},
}

# Converter operators that allow cross-family connections
CONVERTER_OPERATORS: dict[str, tuple[OperatorFamily, OperatorFamily]] = {
    "chopto": (OperatorFamily.CHOP, OperatorFamily.TOP),
    "topto": (OperatorFamily.TOP, OperatorFamily.TOP),  # Actually TOP to different TOP
    "sopto": (OperatorFamily.SOP, OperatorFamily.TOP),
    "datto": (OperatorFamily.DAT, OperatorFamily.TOP),
    "chopto": (OperatorFamily.CHOP, OperatorFamily.TOP),
    "tochop": (OperatorFamily.TOP, OperatorFamily.CHOP),
    "sopchop": (OperatorFamily.SOP, OperatorFamily.CHOP),
    "datchop": (OperatorFamily.DAT, OperatorFamily.CHOP),
}

# Operators that allow feedback loops (cycles)
FEEDBACK_OPERATORS: frozenset[str] = frozenset(
    {
        "feedback",
        "feedbackchop",
        "timemachine",
        "delay",
        "lag",
        "speed",
        "filter",
    }
)
