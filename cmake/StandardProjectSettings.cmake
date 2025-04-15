# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

function(rnx_target_cxx_standard CXX_TARGET)
    target_compile_features(${CXX_TARGET} PUBLIC cxx_std_20)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

function(rnx_force_out_of_source_builds)
    get_filename_component(SRC_DIR "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(BIN_DIR "${CMAKE_BINARY_DIR}" REALPATH)

    if("${SRC_DIR}" STREQUAL "${BIN_DIR}")
        message("############################################################")
        message("Warning: In-source builds are disabled.")
        message("Please create and run cmake from a separate build directory.")
        message("############################################################")
        message(FATAL_ERROR "Quitting configuration...")
    endif()
endfunction()

macro(rnx_set_standard_settings)
    # Set a default build type if none was specified
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
        set(CMAKE_BUILD_TYPE
            RelWithDebInfo
            CACHE STRING "Choose the type of build." FORCE
        )

        # Set the possible values of build type for cmake-gui, ccmake
        set_property(
            CACHE CMAKE_BUILD_TYPE
            PROPERTY STRINGS
            "Debug"
            "Release"
            "MinSizeRel"
            "RelWithDebInfo"
        )
    endif()

    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            CACHE PATH "Default install path" FORCE
        )
    endif()

    # Generate compile_commands.json to make it easier to work with clang based tools
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
