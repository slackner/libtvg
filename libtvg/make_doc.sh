#!/bin/bash
#
# To run this script as a pre-commit hook, just create a symlink
# in the .git/hooks directory, e.g.,
# $ ln -s ../../libtvg/make_doc.sh .git/hooks/pre-commit
#
set -e -x
REPODIR="$(dirname "$(readlink -f "$0")")/.."

cd "$REPODIR/libtvg"
make clean
make

TEMPDIR="$(mktemp -d)"
cp pytvg.py libtvg.so "$TEMPDIR"
cd "$TEMPDIR"

sed -i "/^class c_.*/a \\    def __init__(self, *args, **kwargs):\\n        super().__init__(*args, **kwargs)" pytvg.py
sed -i "s/^@libtvgobject$//g" pytvg.py
# FIXME: Hide internal types.

pydocmd simple pytvg++ > "$REPODIR/PYTVG.md"

cd "$REPODIR"
git add PYTVG.md
rm -rf "$TEMPDIR"
