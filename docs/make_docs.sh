#!/usr/bin/env bash
make clean
rm -r source/reference
sphinx-apidoc -o source/reference ../appletree
rm source/reference/modules.rst
make html #SPHINXOPTS="-W --keep-going -n"
