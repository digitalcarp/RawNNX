# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

cmake_minimum_required(VERSION 3.25)

project(RawNNX
    VERSION 0.0.1
    DESCRIPTION "Machine learning image processing tools using the ONNX runtime"
    LANGUAGES CXX C
)

include(cmake/StandardProjectSettings.cmake)
rnx_force_out_of_source_builds()
rnx_set_standard_settings()

option(rawnnx_BUILD_RUNTIME_LINK_LIBS "Build for dynamic linking at runtime" OFF)
option(rawnnx_BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(rawnnx_BUILD_UTIL_TOOLS "Build utility CLI tools" ON)

option(rawnnx_DEVELOPER_MODE "Enable RawNNX developer mode (for contributors)" OFF)
if(rawnnx_DEVELOPER_MODE)
    option(rawnnx_ENABLE_ASAN "Enable address sanitizer" OFF)
    option(rawnnx_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(rawnnx_ENABLE_CPPCHECK "Enable CppCheck" OFF)
    option(rawnnx_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

    include(cmake/StaticAnalyzers.cmake)
    if(rawnnx_ENABLE_CLANG_TIDY)
      rnx_enable_clang_tidy(${rawnnx_WARNINGS_AS_ERRORS})
    endif()

    if(rawnnx_ENABLE_CPPCHECK)
      rnx_enable_cppcheck(${rawnnx_WARNINGS_AS_ERRORS})
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
    option(rawnnx_BUILD_DEMOS "Build standalone model demos" ON)
    option(rawnnx_BUILD_SANDBOX "Build sandbox demo" ON)
endif()

include(cmake/Dependencies.cmake)
rnx_setup_dependencies()

rnx_disable_clang_tidy_for_third_party()

include(cmake/CompileOptions.cmake)

# Header-only library for users that link RawNNX at runtime
add_library(rawnnx_api INTERFACE)
add_library(rawnnx::api ALIAS rawnnx_api)

target_include_directories(rawnnx_api INTERFACE api)

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
rnx_target_cxx_standard(rawnnx_lib)
rnx_target_compile_options(rawnnx_lib)

add_subdirectory(demo)

if(NOT PROJECT_IS_TOP_LEVEL)
    return()
endif()

if(rawnnx_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
