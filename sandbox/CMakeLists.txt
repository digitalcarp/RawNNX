# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

add_executable(rawnnx_sandbox
    imagearea.cc
    main.cc
    mainwindow.cc
)

rnx_target_cxx_standard(rawnnx_sandbox)
rnx_target_compile_options(rawnnx_sandbox)

target_link_libraries(rawnnx_sandbox PRIVATE
    fmt::fmt
    PkgConfig::GTKMM
)
