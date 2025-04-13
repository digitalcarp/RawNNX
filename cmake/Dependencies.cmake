# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

include(FetchContent)
find_package(PkgConfig)

function(rnx_setup_dependencies)

    # fmt::fmt
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG 11.1.4
        GIT_SHALLOW ON
    )

    # Don't use FetchContent_Declare after this
    FetchContent_MakeAvailable(fmt)

    pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)

    if(rawnnx_BUILD_DEMOS)
        find_package(OpenCV REQUIRED COMPONENTS
            core
            dnn
            highgui
            imgcodecs
            imgproc
        )
    endif()

endfunction()
