# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

cmake_minimum_required(VERSION 3.25)

project(RawNNX
    VERSION 0.0.1
    DESCRIPTION "Machine learning image processing tools using the ONNX runtime"
    LANGUAGES CXX C
)

include(cmake/PreventInSourceBuilds.cmake)

add_library(rawnnx_api INTERFACE)
add_library(rawnnx::api ALIAS rawnnx_api)

target_include_directories(rawnnx_api INTERFACE api)
