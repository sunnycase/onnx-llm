#!/bin/bash

set -x
clear

build_type=Relase
#build_type=Debug

build=x86_64_build
rm -rf ${build}
mkdir ${build}
pushd ${build}
cmake -DCMAKE_BUILD_TYPE=${build_type}           \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DBUILD_NNCASE_LINUX=1 \
      -DDUMP_PROFILE_INFO=1 \
      ..

make -j8
popd

export PATH=$PATH:`pwd`/3rd_party/nncase/linux_x86_64/bin/

dst=Qwen3-0.6B
./x86_64_build/cli_demo ${dst}/config.json
