# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

cmake_minimum_required(VERSION 3.25)

macro(default name)
    if(NOT DEFINED "${name}")
        set("${name}" "${ARGN}")
    endif()
endmacro()

default(FORMAT_COMMAND clang-format)
default(
    PATTERNS
    api/*.h
    api/*.hpp
    include/*.h
    include/*.hpp

    src/*.h
    src/*.hpp
    src/*.c
    src/*.cc
    src/*.cpp

    demo/*.h
    demo/*.hpp
    demo/*.c
    demo/*.cc
    demo/*.cpp

    sandbox/*.h
    sandbox/*.hpp
    sandbox/*.c
    sandbox/*.cc
    sandbox/*.cpp
)
default(FIX NO)

set(flag --output-replacements-xml)
set(args OUTPUT_VARIABLE output)
if(FIX)
    set(flag -i)
    set(args "")
endif()

file(GLOB_RECURSE files ${PATTERNS})
set(badly_formatted "")
set(output "")
string(LENGTH "${CMAKE_SOURCE_DIR}/" path_prefix_length)

foreach(file IN LISTS files)
    execute_process(
        COMMAND "${FORMAT_COMMAND}" --style=file "${flag}" "${file}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE result
        ${args}
    )

    if(NOT result EQUAL "0")
        message(FATAL_ERROR "'${file}': formatter returned with ${result}")
    endif()

    if(NOT FIX AND output MATCHES "\n<replacement offset")
        string(SUBSTRING "${file}" "${path_prefix_length}" -1 relative_file)
        list(APPEND badly_formatted "${relative_file}")
    endif()

    set(output "")
endforeach()

if(NOT badly_formatted STREQUAL "")
    list(JOIN badly_formatted "\n" bad_list)
    message("The following files are badly formatted:\n\n${bad_list}\n")
    message(FATAL_ERROR "Run again with FIX=YES to fix these files.")
endif()
