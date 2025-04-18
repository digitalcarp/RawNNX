# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

function(rnx_target_cxx_standard CXX_TARGET)
    target_compile_features(${CXX_TARGET} PUBLIC cxx_std_20)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(${CXX_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

function(rnx_target_compile_options TARGET_NAME)
    set(COMPILE_OPTS "")

    # Warnings
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND COMPILE_OPTS -Wall -Wextra)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND COMPILE_OPTS -Wall -Wextra)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND COMPILE_OPTS /W4)
    endif()

    # Address sanitizer
    if(rawnnx_ENABLE_ASAN)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            list(APPEND COMPILE_OPTS -fsanitize=address)
            target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(APPEND COMPILE_OPTS -fsanitize=address)
            target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            list(APPEND COMPILE_OPTS /fsanitize=address)
        endif()
    endif()

    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTS})
    if(rawnnx_WARNINGS_AS_ERRORS)
        set_target_properties(${TARGE_NAMET} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    endif()
endfunction()

