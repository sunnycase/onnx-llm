#!/bin/bash
set -x
clear

# set cross build toolchain and k230 sdk(Linux)
export PATH=$PATH:`pwd`/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V2.10.1/bin
build_type=Release
#build_type=Debug

build=riscv64_linux_build
rm -rf ${build}
mkdir ${build}
pushd ${build}
cmake -DCMAKE_BUILD_TYPE=${build_type}           \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/k230.linux.toolchain.cmake \
      -DBUILD_NNCASE_LINUX=1 \
      -DDUMP_PROFILE_INFO=1 \
      ..

make -j
popd

cp ${build}/cli_demo /home/share/nfsroot/k230/k230_llm/onnx-llm/linux_qwen3.0
