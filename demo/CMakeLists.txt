# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

if(rawnnx_BUILD_UTIL_TOOLS)
    add_executable(onnx-dump onnxdump.cc ${PROJECT_SOURCE_DIR}/src/debug.cc)
    rnx_target_cxx_standard(onnx-dump)
    target_include_directories(onnx-dump PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(onnx-dump PRIVATE fmt::fmt PkgConfig::onnxruntime)
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

    add_executable(mobilesam mobilesam_demo.cc)
    rnx_target_cxx_standard(mobilesam)
    target_link_libraries(mobilesam PRIVATE ${DEMO_LIBS})
endif()
