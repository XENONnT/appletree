#!/bin/bash
# copied from https://github.com/XENONnT/straxen/blob/master/.github/scripts/pre_pyflakes.sh
# Pyflakes does not like the way we do __all__ += []. This simple script
# Changes all the files in appletree to abide by this convention and
# removes the lines that have such a signature.
start="$(pwd)"
echo $start

cd appletree
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd plugins
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd $start
echo "done"
