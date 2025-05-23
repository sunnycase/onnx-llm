#!/bin/bash
set -x
clear

# set cross build toolchain and k230 sdk(RTOS)
export PATH=$PATH:`pwd`/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin
build_type=Release
#build_type=Debug

build=riscv64_rtos_build
rm -rf ${build}
mkdir ${build}
pushd ${build}
cmake -DCMAKE_BUILD_TYPE=${build_type}           \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/k230.rtos.toolchain.cmake \
      -DBUILD_NNCASE_RTOS=1 \
      -DDUMP_PROFILE_INFO=1 \
      ..

make -j
popd

cp ${build}/cli_demo /home/share/nfsroot/k230/k230_llm/onnx-llm/rtos_qwen3.0
