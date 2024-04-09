CMAKE_MINIMUM_REQUIRED(VERSION 3.16.3)
project(pointpillars_multihead_40fps)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

find_package(PCL 1.9 REQUIRED COMPONENTS io common)
find_package(yaml-cpp REQUIRED)
find_package(Threads)
find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED QUIET)
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser libnvonnxparser
  HINTS "/usr/local/lib" "{NVONNXTENSORRT_ROOT}/build")
MESSAGE(STATUS "Find ONNX-TensorRT libs at ${NVONNXPARSER_LIBRARY}")

cuda_add_library(pointpillars_multihead_40fps_cuda STATIC
  pointpillars/postprocess.cu
  pointpillars/preprocess.cu
  pointpillars/scatter.cu
  pointpillars/nms.cu
)
target_include_directories(pointpillars_multihead_40fps_cuda PUBLIC
  pointpillars
  ${CUDA_INCLUDE_DIRS}
)


add_library(pointpillars_multihead_40fps STATIC
  pointpillars/pointpillars.cc
)
add_dependencies(pointpillars_multihead_40fps
  pointpillars_multihead_40fps_cuda
)
target_include_directories(pointpillars_multihead_40fps PUBLIC
  pointpillars
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(pointpillars_multihead_40fps
  pointpillars_multihead_40fps_cuda
  yaml-cpp
)

add_executable(binary_demo app/binary_demo.cc)
target_include_directories(binary_demo PUBLIC
  pointpillars
  ${PCL_INCLUDE_DIRS}
)
add_dependencies(binary_demo
  pointpillars_multihead_40fps
  pointpillars_multihead_40fps_cuda
)
target_link_libraries(binary_demo
  pointpillars_multihead_40fps
  pointpillars_multihead_40fps_cuda
  stdc++fs
  ${PCL_LIBRARIES}
  ${TENSORRT_LIBRARY}
  ${NVONNXPARSER_LIBRARY}
)

find_package(GTest)
if(NOT GTEST_LIBRARY)
    message("Warning: GTest not found! Tests will be disabled.")
endif()

if(GTEST_LIBRARY)
  enable_testing()

  add_executable(pointpillars_multihead_40fps_tests test/pointpillars_test.cc)
  target_include_directories(pointpillars_multihead_40fps_tests PUBLIC
    pointpillars
  )
  add_dependencies(pointpillars_multihead_40fps_tests 
    pointpillars_multihead_40fps
    pointpillars_multihead_40fps_cuda
  )
  target_link_libraries(pointpillars_multihead_40fps_tests
    pointpillars_multihead_40fps
    pointpillars_multihead_40fps_cuda
    ${TENSORRT_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    GTest::gtest_main
  )
  gtest_discover_tests(pointpillars_multihead_40fps_tests)
endif()
