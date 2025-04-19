# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

if(rawnnx_BUILD_UTIL_TOOLS)
    add_executable(onnx-dump onnxdump.cc ${PROJECT_SOURCE_DIR}/src/debug.cc)
    rnx_target_cxx_standard(onnx-dump)
    rnx_target_compile_options(onnx-dump)
    target_include_directories(onnx-dump PRIVATE ${PROJECT_SOURCE_DIR})
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

    function(rnx_setup_standalone_demo TARGET_NAME SRC)
        add_executable(${TARGET_NAME} ${SRC})
        rnx_target_cxx_standard(${TARGET_NAME})
        rnx_target_compile_options(${TARGET_NAME})
        target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${TARGET_NAME} PRIVATE ${DEMO_LIBS})
    endfunction()

    rnx_setup_standalone_demo(mobilesam mobilesam_demo.cc)
    rnx_setup_standalone_demo(sam-hq sam_hq_demo.cc)
endif()
