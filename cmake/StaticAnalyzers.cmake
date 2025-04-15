# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

macro(rnx_enable_clang_tidy WARNINGS_AS_ERRORS)
    find_program(CLANG_TIDY clang-tidy)

    if(NOT CLANG_TIDY)
        message(SEND_ERROR "clang-tidy executable not found")
        return()
    endif()

    set(CLANG_TIDY_OPTIONS
        ${CLANG_TIDY}
        --allow-no-checks
        -extra-arg=-Wno-unknown-warning-option
        -extra-arg=-Wno-ignored-optimization-argument
        -extra-arg=-Wno-unused-command-line-argument)

    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
        if("${CLANG_TIDY_OPTIONS_DRIVER_MODE}" STREQUAL "cl")
            list(APPEND CLANG_TIDY_OPTIONS -extra-arg=/std:c++${CMAKE_CXX_STANDARD})
        else()
            list(APPEND CLANG_TIDY_OPTIONS -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
        endif()
    endif()

    list(APPEND CLANG_TIDY_OPTIONS --header-filter="${PROJECT_SOURCE_DIR}/.*")
    list(APPEND CLANG_TIDY_OPTIONS --exclude-header-filter="")

    if(${WARNINGS_AS_ERRORS})
        list(APPEND CLANG_TIDY_OPTIONS -warnings-as-errors=*)
    endif()

    message("Enabling clang-tidy for all targets by default")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_OPTIONS})
    set(CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR "${CMAKE_BINARY_DIR}/clang-tidy-fixes")
    message("clang-tidy fixes will be exported to ${CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR}")
endmacro()

function(rnx_disable_clang_tidy_for_third_party)
    if(NOT rawnnx_ENABLE_CLANG_TIDY)
        return()
    endif()

    message(STATUS "Writing .clang-tidy to ${CMAKE_BINARY_DIR}")

    # Write .clang-tidy that disables all checks
    file(WRITE ${CMAKE_BINARY_DIR}/.clang-tidy "Checks: '-*'")
endfunction()

macro(rnx_enable_cppcheck)
    find_program(CPPCHECK cppcheck)

    if(NOT CPPCHECK)
        message(SEND_ERROR "cppcheck executable not found")
        return()
    endif()

    set(CMAKE_CXX_CPPCHECK ${CPPCHECK})

    if(CMAKE_GENERATOR MATCHES ".*Visual Studio.*")
        list(APPEND CMAKE_CXX_CPPCHECK --template=vs)
    else()
        list(APPEND CMAKE_CXX_CPPCHECK --template=gcc)
    endif()

    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
        list(APPEND CMAKE_CXX_CPPCHECK --std=c++${CMAKE_CXX_STANDARD})
    endif()

    if(${WARNINGS_AS_ERRORS})
        list(APPEND CMAKE_CXX_CPPCHECK --error-exitcode=2)
    endif()

    list(APPEND CMAKE_CXX_CPPCHECK
        --enable=style,performance,warning,portability
        --inline-suppr
        --suppress=*:${CMAKE_BINARY_DIR}/*
        --suppress=normalCheckLevelMaxBranches
        # We cannot act on a bug/missing feature of cppcheck.
        # If a file does not have an internalAstError, we get an
        # unmatchedSuppression error.
        --suppress=cppcheckError
        --suppress=internalAstError
        --suppress=unmatchedSuppression
        # Noisy and incorrect sometimes
        --suppress=passedByValue
        # Ignores code that cppcheck thinks is invalid C++
        --suppress=syntaxError
        --suppress=preprocessorErrorDirective
        --inconclusive)

    message("Enabling cppcheck for all targets by default")
endmacro()
