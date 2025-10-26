//
// MetalCompressionArtifact.swift
// High-performance Metal backend for JPEG compression artifact simulation
//

import Foundation
import Metal
import MetalKit

// Tile descriptor matching the Metal shader struct
struct TileDescriptor {
    var y_start: UInt32
    var y_end: UInt32
    var x_start: UInt32
    var x_end: UInt32
}

@objc public class MetalCompressionArtifact: NSObject {
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLComputePipelineState!
    private var textureLoader: MTKTextureLoader!

    @objc public override init() {
        super.init()

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = commandQueue

        self.textureLoader = MTKTextureLoader(device: device)

        setupPipeline()
    }

    private func setupPipeline() {
        // Load the Metal shader library
        let bundle = Bundle(for: type(of: self))
        guard let metalPath = bundle.path(forResource: "compression_artifact", ofType: "metal") else {
            fatalError("Could not find compression_artifact.metal in bundle")
        }

        do {
            let metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)
            let library = try device.makeLibrary(source: metalSource, options: nil)

            guard let kernelFunction = library.makeFunction(name: "apply_compression_artifacts") else {
                fatalError("Failed to load kernel function")
            }

            pipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            fatalError("Failed to setup Metal pipeline: \(error)")
        }
    }

    // Scale quantization matrix based on quality (1-20)
    private func scaleQuantizationMatrix(_ baseMatrix: [Float], quality: Int) -> [Float] {
        let scale: Float = quality < 50 ? 5000.0 / Float(quality) : 200.0 - 2.0 * Float(quality)

        return baseMatrix.map { value in
            let scaled = floor((value * scale + 50.0) / 100.0)
            return min(max(scaled, 1.0), 255.0)
        }
    }


    // Main entry point: apply compression artifacts to image
    @objc public func applyCompressionArtifacts(
        imageData: UnsafeMutablePointer<UInt8>,
        width: Int,
        height: Int,
        channels: Int,
        tiles: [[Int]],  // Array of [y_start, y_end, x_start, x_end]
        quality: Int
    ) -> Bool {
        // Validate inputs
        guard channels == 3 || channels == 4 else {
            print("Error: Only RGB (3) or RGBA (4) channels supported")
            return false
        }

        // Create texture from image data
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: channels == 4 ? .rgba8Unorm : .rgba8Unorm,  // Use RGBA for both
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Error: Failed to create texture")
            return false
        }

        // Upload image data to texture
        let bytesPerRow = width * 4  // Always use 4 bytes per pixel for Metal
        var rgbaData = [UInt8]()

        if channels == 3 {
            // Convert RGB to RGBA
            rgbaData.reserveCapacity(width * height * 4)
            for i in 0..<(width * height) {
                let offset = i * 3
                rgbaData.append(imageData[offset])
                rgbaData.append(imageData[offset + 1])
                rgbaData.append(imageData[offset + 2])
                rgbaData.append(255)  // Alpha = 1.0
            }
        } else {
            // Already RGBA
            rgbaData = Array(UnsafeBufferPointer(start: imageData, count: width * height * 4))
        }

        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: rgbaData,
            bytesPerRow: bytesPerRow
        )

        // JPEG quantization matrices
        let jpegQuantLuma: [Float] = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        ]

        let jpegQuantChroma: [Float] = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99
        ]

        // Scale quantization matrices
        var scaledLuma = scaleQuantizationMatrix(jpegQuantLuma, quality: quality)
        var scaledChroma = scaleQuantizationMatrix(jpegQuantChroma, quality: quality)

        // Create tile descriptors buffer
        var tileDescriptors = tiles.map { tile -> TileDescriptor in
            TileDescriptor(
                y_start: UInt32(tile[0]),
                y_end: UInt32(tile[1]),
                x_start: UInt32(tile[2]),
                x_end: UInt32(tile[3])
            )
        }

        guard let tilesBuffer = device.makeBuffer(
            bytes: &tileDescriptors,
            length: MemoryLayout<TileDescriptor>.stride * tileDescriptors.count,
            options: .storageModeShared
        ) else {
            print("Error: Failed to create tiles buffer")
            return false
        }

        // Create quantization matrix buffers
        guard let quantLumaBuffer = device.makeBuffer(
            bytes: &scaledLuma,
            length: MemoryLayout<Float>.stride * scaledLuma.count,
            options: .storageModeShared
        ) else {
            print("Error: Failed to create quant luma buffer")
            return false
        }

        guard let quantChromaBuffer = device.makeBuffer(
            bytes: &scaledChroma,
            length: MemoryLayout<Float>.stride * scaledChroma.count,
            options: .storageModeShared
        ) else {
            print("Error: Failed to create quant chroma buffer")
            return false
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Error: Failed to create command buffer")
            return false
        }

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error: Failed to create compute encoder")
            return false
        }

        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(texture, index: 0)
        computeEncoder.setBuffer(tilesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(quantLumaBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(quantChromaBuffer, offset: 0, index: 2)

        var numTiles = UInt32(tiles.count)
        computeEncoder.setBytes(&numTiles, length: MemoryLayout<UInt32>.size, index: 3)

        // Calculate threadgroup size and dispatch
        // For each tile, we process multiple 8x8 blocks
        let maxTileHeight = tiles.map { $0[1] - $0[0] }.max() ?? 0
        let maxTileWidth = tiles.map { $0[3] - $0[2] }.max() ?? 0
        let maxBlocksY = (maxTileHeight + 7) / 8
        let maxBlocksX = (maxTileWidth + 7) / 8

        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: tiles.count,
            height: maxBlocksY,
            depth: maxBlocksX
        )

        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read back texture data
        var outputData = [UInt8](repeating: 0, count: width * height * 4)
        texture.getBytes(
            &outputData,
            bytesPerRow: bytesPerRow,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )

        // Copy back to original buffer
        if channels == 3 {
            // Convert RGBA back to RGB
            for i in 0..<(width * height) {
                let srcOffset = i * 4
                let dstOffset = i * 3
                imageData[dstOffset] = outputData[srcOffset]
                imageData[dstOffset + 1] = outputData[srcOffset + 1]
                imageData[dstOffset + 2] = outputData[srcOffset + 2]
            }
        } else {
            // Copy RGBA directly
            memcpy(imageData, outputData, width * height * 4)
        }

        return true
    }
}

// MARK: - C-Compatible FFI Boundary

// Global instance for C function to use
private var globalMetalInstance: MetalCompressionArtifact?

// C-compatible wrapper function for Python ctypes
@_cdecl("metal_compression_apply")
public func metal_compression_apply(
    imageData: UnsafeMutablePointer<UInt8>,
    width: Int32,
    height: Int32,
    channels: Int32,
    tilesFlat: UnsafePointer<Int32>,
    numTiles: Int32,
    quality: Int32
) -> Bool {
    // Lazy initialization of Metal instance
    if globalMetalInstance == nil {
        globalMetalInstance = MetalCompressionArtifact()
    }

    guard let instance = globalMetalInstance else {
        print("Error: Failed to initialize Metal instance")
        return false
    }

    // Convert flat tiles array to Swift [[Int]]
    // Format: [y_start, y_end, x_start, x_end, y_start, y_end, x_start, x_end, ...]
    var tiles: [[Int]] = []
    tiles.reserveCapacity(Int(numTiles))

    for i in 0..<Int(numTiles) {
        let offset = i * 4
        let tile = [
            Int(tilesFlat[offset]),     // y_start
            Int(tilesFlat[offset + 1]), // y_end
            Int(tilesFlat[offset + 2]), // x_start
            Int(tilesFlat[offset + 3])  // x_end
        ]
        tiles.append(tile)
    }

    // Call the Swift implementation
    return instance.applyCompressionArtifacts(
        imageData: imageData,
        width: Int(width),
        height: Int(height),
        channels: Int(channels),
        tiles: tiles,
        quality: Int(quality)
    )
}
