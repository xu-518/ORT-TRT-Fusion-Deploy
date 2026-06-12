cmake_minimum_required(VERSION 3.10)
project(TensorRT_Fusion)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release) 
add_compile_options(-Wno-deprecated-declarations)
find_package(OpenCV REQUIRED)
set(TENSORRT_DIR /home/xushuo/code/test_cmake/TensorRT-10.0.1.6)
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
add_executable(main 
    main.cpp
    TrtFusionEngine.cpp
)
target_link_libraries(main 
    ${OpenCV_LIBS}
    nvinfer    
    cudart     
)