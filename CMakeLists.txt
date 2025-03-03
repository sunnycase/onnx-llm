cmake_minimum_required(VERSION 3.5)
project(onnx-llm)

# find_package(nncase REQUIRED)

option(BUILD_FOR_ANDROID "Build for android whith mini memory mode." OFF)
option(LLM_SUPPORT_VISION "Llm model support vision input." OFF)
option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)
option(BUILD_JNI "Build JNI for android app." OFF)
option(BUILD_ONNX_RUNTIME "Build on onnx runtime." OFF)
option(BUILD_NNCASE "Build on nncase." ON)

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (BUILD_ONNX_RUNTIME)
set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)
include_directories(${ONNXRUNTIME_PATH}/include)
link_directories(${ONNXRUNTIME_PATH}/lib)
else()
if(CMAKE_CROSSCOMPILING)
set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/riscv64)
link_directories(${CMAKE_SOURCE_DIR}/3rd_party/mmz/riscv64)
else()
set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/x86_64)
endif()
include_directories(3rd_party
                    ${NNCASE_PATH}/include
                    ${NNCASE_PATH}/include/nncase/runtime
                    )
link_directories(${NNCASE_PATH}/lib)
endif()



include_directories(${CMAKE_SOURCE_DIR}/include)


FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(llm STATIC ${SRCS})

if (BUILD_ONNX_RUNTIME)
target_link_libraries(llm PRIVATE onnxruntime)
else()
if(CMAKE_CROSSCOMPILING)
target_link_libraries(llm PUBLIC nncase.rt_modules.k230 Nncase.Runtime.Native functional_k230 mmz)
else()
target_link_libraries(llm PUBLIC Nncase.Runtime.Native)
endif()
endif()
add_executable(cli_demo ${CMAKE_SOURCE_DIR}/demo/cli_demo.cpp)
target_link_libraries(cli_demo llm)