cmake_minimum_required(VERSION 3.26)
project(deploy)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(
        /opt/cpp_work/library/TensorRT-8.6.1.6/include
        /usr/local/cuda-11.7/include
        ${OpenCV_INCLUDE_DIRS}
        /opt/cpp_work/library/onnxruntime1.16.0/include
)

add_executable(onnx
        test.cpp
        onnxModel.cpp
        onnxModel.h
        #        test1.cpp
)
add_executable(tensorrt
        test1.cpp
)
add_executable(trt_test
        tensorrt_test.cpp
        tensorrt_test.h
        main.cpp
)

target_link_libraries(onnx
        /usr/local/cuda-11.7/lib64/libcudart.so
        /opt/cpp_work/library/onnxruntime1.16.0/lib/libonnxruntime.so
        /opt/cpp_work/library/onnxruntime1.16.0/lib/libonnxruntime.so.1.16.0
        ${OpenCV_LIBS}
)
target_link_libraries(tensorrt
        /usr/local/cuda-11.7/lib64/libcudart.so
        /opt/cpp_work/library/TensorRT-8.6.1.6/lib/libnvinfer.so
        ${OpenCV_LIBS}
)
target_link_libraries(trt_test
        /usr/local/cuda-11.7/lib64/libcudart.so
        /opt/cpp_work/library/TensorRT-8.6.1.6/lib/libnvinfer.so
        ${OpenCV_LIBS}
)