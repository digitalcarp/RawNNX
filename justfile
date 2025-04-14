set dotenv-load

default:
    @just help

help:
    @just --list --unsorted

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
