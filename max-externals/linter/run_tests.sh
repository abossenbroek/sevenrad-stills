#!/bin/bash
# Quick test runner for GenExpr parser and analyzer

set -e

echo "Testing GenExpr Parser..."
python3 tests/test_parser.py
echo ""

echo "Testing GenExpr Semantic Analyzer..."
python3 tests/test_analyzer.py
echo ""

echo "All tests completed successfully!"
