cmake_minimum_required(VERSION 3.10)
project(FusionDeploy)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(ORT_DIR "/home/xushuo/code/test_cmake/onnxruntime-linux-x64-cuda-1.17.1")
include_directories(${ORT_DIR}/include)
link_directories(${ORT_DIR}/lib)
add_executable(main 
    main.cpp
    OrtFusionEngine.cpp
)

target_link_libraries(main 
    ${OpenCV_LIBS} 
    onnxruntime 
    ${CUDA_CUDART_LIBRARY})