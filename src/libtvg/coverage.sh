#!/bin/bash
set -e -x
cd "$(dirname "$0")"
make clean

make CC="clang" CFLAGS="-ftest-coverage -fprofile-arcs"
make tests

./tests
./pytvg.py

lcov --directory ./ --capture \
     --gcov-tool "$(pwd)/llvm-gcov.sh" \
     --output-file coverage.info
mkdir -p coverage
genhtml -o coverage coverage.info

make clean
