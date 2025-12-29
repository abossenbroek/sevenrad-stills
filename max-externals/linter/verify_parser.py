#!/usr/bin/env python3
"""Quick verification that the parser module can be imported and works."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Attempting to import parser module...")
try:
    from max_linter.genexpr.parser import GenExprParser, ParseError

    print("✓ Successfully imported GenExprParser and ParseError")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nCreating parser instance...")
try:
    parser = GenExprParser()
    print("✓ Parser instance created successfully")
except Exception as e:
    print(f"✗ Parser creation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic parsing...")
try:
    tree = parser.parse("out1 = in1;")
    print("✓ Parsed simple statement successfully")
    print(f"  Tree type: {type(tree).__name__}")
    print(f"  Tree data: {tree.data}")
except Exception as e:
    print(f"✗ Parsing failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nTesting error handling...")
try:
    parser.parse("out1 = ;")
    print("✗ Should have raised ParseError!")
    sys.exit(1)
except ParseError as e:
    print("✓ ParseError raised correctly")
    print(f"  Line: {e.line}, Column: {e.column}")
    print(f"  Message: {e.message[:60]}...")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All basic verification tests passed!")
print("=" * 60)
