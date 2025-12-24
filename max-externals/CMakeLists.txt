cmake_minimum_required(VERSION 3.19)
project(sevenrad VERSION 1.0.0 LANGUAGES C)

# ----------------------------------------------------------------------------
# Max SDK Configuration
# ----------------------------------------------------------------------------
# Set the path to Max SDK - user should configure this
if(NOT DEFINED MAX_SDK_DIR)
    if(DEFINED ENV{MAX_SDK_DIR})
        set(MAX_SDK_DIR $ENV{MAX_SDK_DIR})
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/max-sdk")
        # Max SDK as submodule within max-externals/
        set(MAX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/max-sdk")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../max-sdk")
        set(MAX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../max-sdk")
    else()
        message(FATAL_ERROR "MAX_SDK_DIR not set. Please run 'git submodule update --init --recursive' or set MAX_SDK_DIR to Max SDK 8.2.0+ directory")
    endif()
endif()

message(STATUS "Using Max SDK at: ${MAX_SDK_DIR}")

# Include Max SDK CMake modules
set(C74_MAX_API_DIR "${MAX_SDK_DIR}/source/max-sdk-base/c74support/max-includes")
set(C74_MSP_API_DIR "${MAX_SDK_DIR}/source/max-sdk-base/c74support/msp-includes")
set(C74_JIT_API_DIR "${MAX_SDK_DIR}/source/max-sdk-base/c74support/jit-includes")
set(C74_MAX_API_LIB_DIR "${MAX_SDK_DIR}/source/max-sdk-base/c74support/max-includes/x64")

# Platform-specific settings
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
    set(EXTERNAL_SUFFIX ".mxo")
elseif(WIN32)
    set(EXTERNAL_SUFFIX ".mxe64")
endif()

# ----------------------------------------------------------------------------
# Common Settings
# ----------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/source/common
    ${C74_MAX_API_DIR}
    ${C74_MSP_API_DIR}
    ${C74_JIT_API_DIR}
)

# Platform definitions
if(APPLE)
    add_definitions(-DMAC_VERSION)
elseif(WIN32)
    add_definitions(-DWIN_VERSION)
endif()

# ----------------------------------------------------------------------------
# Helper function for creating Max externals
# ----------------------------------------------------------------------------
function(add_max_external name)
    set(SOURCES ${ARGN})

    add_library(${name} MODULE ${SOURCES})

    # Set output name without prefix
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${name}"
        BUNDLE TRUE
        BUNDLE_EXTENSION "mxo"
    )

    if(APPLE)
        # macOS bundle configuration
        set_target_properties(${name} PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/source/Info.plist.in"
        )

        # Link against Max API framework
        target_link_libraries(${name} PRIVATE
            "-framework CoreFoundation"
        )

        # Find and link MaxAPI
        find_library(MAX_API_LIB MaxAPI PATHS "${C74_MAX_API_DIR}")
        if(MAX_API_LIB)
            target_link_libraries(${name} PRIVATE ${MAX_API_LIB})
        endif()

    elseif(WIN32)
        # Windows configuration
        target_link_libraries(${name} PRIVATE
            "${C74_MAX_API_LIB_DIR}/MaxAPI.lib"
        )
    endif()

    # Copy to externals directory
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_BUNDLE_DIR:${name}>"
            "${CMAKE_CURRENT_SOURCE_DIR}/externals/${name}${EXTERNAL_SUFFIX}"
    )
endfunction()

# ----------------------------------------------------------------------------
# Helper function for creating Jitter externals (requires JitterAPI)
# ----------------------------------------------------------------------------
function(add_jitter_external name)
    set(SOURCES ${ARGN})

    add_library(${name} MODULE ${SOURCES})

    # Set output name without prefix
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${name}"
        BUNDLE TRUE
        BUNDLE_EXTENSION "mxo"
    )

    if(APPLE)
        # macOS bundle configuration
        set_target_properties(${name} PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/source/Info.plist.in"
        )

        # Link against frameworks
        target_link_libraries(${name} PRIVATE
            "-framework CoreFoundation"
        )

        # Find and link MaxAPI
        find_library(MAX_API_LIB MaxAPI PATHS "${C74_MAX_API_DIR}")
        if(MAX_API_LIB)
            target_link_libraries(${name} PRIVATE ${MAX_API_LIB})
        endif()

        # Find and link JitterAPI
        # Note: We link weakly because Max provides JitterAPI.framework at runtime
        # The framework is not bundled with the external
        find_library(JIT_API_LIB JitterAPI PATHS "${C74_JIT_API_DIR}" NO_DEFAULT_PATH)
        if(JIT_API_LIB)
            target_link_libraries(${name} PRIVATE ${JIT_API_LIB})
            # Set install name for JitterAPI to use @rpath so Max can provide it at runtime
            set_target_properties(${name} PROPERTIES
                INSTALL_RPATH "@executable_path/../Frameworks"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()

    elseif(WIN32)
        # Windows configuration
        target_link_libraries(${name} PRIVATE
            "${C74_MAX_API_LIB_DIR}/MaxAPI.lib"
            "${C74_JIT_API_DIR}/JitterAPI.lib"
        )
    endif()

    # Copy to externals directory
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_BUNDLE_DIR:${name}>"
            "${CMAKE_CURRENT_SOURCE_DIR}/externals/${name}${EXTERNAL_SUFFIX}"
    )
endfunction()

# ----------------------------------------------------------------------------
# Externals
# ----------------------------------------------------------------------------

# sr.tilegen - Random tile bounds generator for bandswap/corruption effects
add_max_external(sr.tilegen
    source/sr.tilegen/sr.tilegen.c
)

# sr.maskgen - SLC-off wedge mask generator (requires Jitter for jit.matrix output)
add_jitter_external(sr.maskgen
    source/sr.maskgen/sr.maskgen.c
)

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------
# Install GenExpr shaders and help files to package
install(DIRECTORY code/ DESTINATION code FILES_MATCHING PATTERN "*.genjit")
install(DIRECTORY help/ DESTINATION help FILES_MATCHING PATTERN "*.maxhelp")
install(FILES package-info.json DESTINATION .)
