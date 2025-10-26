#!/bin/bash
#
# Build script for Metal compression artifact library
# Compiles Swift wrapper into a dynamic library that can be called from Python
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/build"

mkdir -p "${OUTPUT_DIR}"

echo "Building MetalCompressionArtifact.dylib..."

# Compile Swift code into a dynamic library
swiftc -emit-library \
    -o "${OUTPUT_DIR}/libMetalCompressionArtifact.dylib" \
    -module-name MetalCompressionArtifact \
    -emit-module \
    -emit-module-path "${OUTPUT_DIR}/" \
    -Xlinker -install_name -Xlinker "@rpath/libMetalCompressionArtifact.dylib" \
    -Xlinker -rpath -Xlinker "@loader_path" \
    -framework Metal \
    -framework MetalKit \
    -framework Foundation \
    -framework CoreGraphics \
    "${SCRIPT_DIR}/MetalCompressionArtifact.swift"

echo "Build complete: ${OUTPUT_DIR}/libMetalCompressionArtifact.dylib"

# Copy Metal shader to build directory
cp "${SCRIPT_DIR}/compression_artifact.metal" "${OUTPUT_DIR}/"

echo "Metal shader copied to: ${OUTPUT_DIR}/compression_artifact.metal"
