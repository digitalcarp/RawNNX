# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

cmake_minimum_required(VERSION 3.25)

project(RawNNX
    VERSION 0.0.1
    DESCRIPTION "Machine learning image processing tools using the ONNX runtime"
    LANGUAGES CXX C
)

include(cmake/PreventInSourceBuilds.cmake)

option(rawnnx_BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(rawnnx_BUILD_RUNTIME_LINK_LIBS "Build for dynamic linking at runtime" OFF)
option(rawnnx_BUILD_UTIL_TOOLS "Build utility CLI tools" ON)
# option(rawnnx_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(PROJECT_IS_TOP_LEVEL)
    option(rawnnx_BUILD_SANDBOX "Build sandbox demo" ON)
    option(rawnnx_BUILD_DEMOS "Build standalone model demos" ON)
endif()

# Header-only library for users that link RawNNX at runtime
add_library(rawnnx_api INTERFACE)
add_library(rawnnx::api ALIAS rawnnx_api)

target_include_directories(rawnnx_api INTERFACE api)

include(FetchContent)
# fmt::fmt
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 11.1.4
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(fmt)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc)

find_package(PkgConfig)
pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)

# Compile RawNNX library
set(LIB_SRC_FILES src/rawnnx.cc)

if(rawnnx_BUILD_RUNTIME_LINK_LIBS)
    add_library(rawnnx_lib MODULE ${LIB_SRC_FILES})
elseif(rawnnx_BUILD_SHARED_LIBS)
    add_library(rawnnx_lib SHARED ${LIB_SRC_FILES})
else()
    add_library(rawnnx_lib STATIC ${LIB_SRC_FILES})
endif()

add_library(rawnnx::rawnnx ALIAS rawnnx_lib)

function(rawnnx_target_cxx_standard CXX_TARGET)
    target_compile_features(${CXX_TARGET} PUBLIC cxx_std_17)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

rawnnx_target_cxx_standard(rawnnx_lib)

if(rawnnx_BUILD_UTIL_TOOLS)
    add_executable(onnx-dump demo/onnxdump.cc src/debug.cc)
    rawnnx_target_cxx_standard(onnx-dump)
    # target_include_directories(onnx-dump PRIVATE ${PROJECT_SRC_DIR})
    target_include_directories(onnx-dump PRIVATE src)
    target_link_libraries(onnx-dump PRIVATE fmt::fmt PkgConfig::onnxruntime)
endif()

if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "Configuring RawNNX as a top level project...")
else()
    return()
endif()

if(rawnnx_BUILD_DEMOS)
    set(DEMO_LIBS
        fmt::fmt
        opencv_core
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        PkgConfig::onnxruntime
    )

    add_executable(mobilesam demo/mobilesam_demo.cc)
    rawnnx_target_cxx_standard(mobilesam)
    target_link_libraries(mobilesam PRIVATE ${DEMO_LIBS})
endif()

if(rawnnx_BUILD_SANDBOX)
    add_executable(rawnnx_sandbox sandbox/main.cc)
    rawnnx_target_cxx_standard(rawnnx_sandbox)

    target_link_libraries(rawnnx_sandbox PRIVATE fmt::fmt)
endif()
