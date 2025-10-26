# Makefile for compiling Metal shaders

SHADER_SRC = buffer_corruption.metal
SHADER_AIR = buffer_corruption.air
SHADER_LIB = buffer_corruption.metallib

# Default target
all: $(SHADER_LIB)

# Compile .metal to .air (intermediate representation)
$(SHADER_AIR): $(SHADER_SRC)
	xcrun -sdk macosx metal -c $(SHADER_SRC) -o $(SHADER_AIR)

# Link .air to .metallib (Metal library)
$(SHADER_LIB): $(SHADER_AIR)
	xcrun -sdk macosx metallib $(SHADER_AIR) -o $(SHADER_LIB)

# Clean intermediate and output files
clean:
	rm -f $(SHADER_AIR) $(SHADER_LIB)

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean rebuild
