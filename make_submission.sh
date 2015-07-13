#!/bin/sh
files=`ls | grep -v '^build' | grep -v 'data' |grep -v tmp |grep -v gpucomp-a5`
echo $files
rm -rf gpucomp-a5
mkdir gpucomp-a5
set -x
cp -r $files gpucomp-a5
rm -f gpucomp-a5.zip
zip -r gpucomp-a5.zip gpucomp-a5
rm -r gpucomp-a5
