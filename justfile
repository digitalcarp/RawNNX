set dotenv-load

# Populate .env with the following optional environment variables:
#
# DEV_PRESET    - dev config+build preset name
# DEBUG_PRESET  - debug config+build preset name
# STRICT_PRESET - dev-strict config+build preset name
#
# CLANG_TIDY_EXPORT_DIR - Directory containing clang-tidy fixes
dev_preset := env("DEV_PRESET", "dev")
debug_preset := env("DEBUG_PRESET", "dev-debug")
strict_preset := env("STRICT_PRESET", "dev-strict")

clang_tidy_export_dir := env("CLANG_TIDY_EXPORT_DIR", "build/dev/clang-tidy-fixes")

default:
    @just help

help:
    @just --list --unsorted

# Configure dev preset
[group('dev')]
config:
    cmake --preset {{dev_preset}}

# Build dev preset
[group('dev')]
build:
    cmake --build --preset {{dev_preset}}

# Configure dev-strict preset
[group('dev')]
config-strict:
    cmake --preset {{strict_preset}}

# Build dev-strict preset
[group('dev')]
build-strict:
    cmake --build --preset {{strict_preset}}

# Configure dev-debug preset
[group('dev')]
config-debug:
    cmake --preset {{debug_preset}}

# Build dev-debug preset
[group('dev')]
build-debug:
    cmake --build --preset {{debug_preset}}

[group('lint')]
format:
    cmake -DFIX=YES -P cmake/Format.cmake

[group('lint')]
format-py:
    black .

[group('lint')]
format-all:
    @just format
    @just format-py

[group('lint')]
format-check:
    cmake -DFIX=NO -P cmake/Format.cmake
    black --check .

# Apply clang-tidy fixes
[group('lint')]
clang-tidy-apply:
    clang-apply-replacements --format --style=file:.clang-format \
        {{clang_tidy_export_dir}}
