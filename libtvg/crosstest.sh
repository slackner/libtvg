#!/bin/bash
set -e -x

for CC in gcc clang; do
	make clean
	make CC="$CC" test
	./pytvg.py

	make clean
	make CC="$CC" NOMONGODB=1 test

	make clean
	make CC="$CC" NOMONGODB=1 CFLAGS="-m32" test

	make clean
	make CC="$CC" NOVALGRIND=1 test

	make clean
	make CC="$CC" NOMONGODB=1 NOVALGRIND=1 test

	make clean
	make CC="$CC" NOMONGODB=1 NOVALGRIND=1 CFLAGS="-m32" test
done
