#!/bin/sh
(find . -name "*.h" ;find . -name "*.cpp" ;find . -name "*.cl")| grep -v 'build/' |grep -v md5| xargs wc -l
