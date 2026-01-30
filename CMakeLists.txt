cmake_minimum_required(VERSION 3.20)
project(RealTimeDectectVer0.3 LANGUAGES CXX CUDA)
if(MSVC)
    # 使用生成表达式，仅当语言是 CXX (C++) 时才添加 /utf-8
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NVCODEC_SDK_ROOT "E:/Video_Codec_SDK_13.0.37")
set(TRTYOLO_INSTALL_DIR "E:/_PROJECT_TRD_LIB/TensorRT-YOLO_gpu")
set(OpenCV_DIR "E:/opencv-4.12.0-windows/opencv/build" )
set(OpenCV_INCLUDE_DIRS "E:/opencv-4.12.0-windows/opencv/build/include" "E:/opencv-4.12.0-windows/opencv/build/include/opencv2")
set(OpenCV_LIBS "E:/opencv-4.12.0-windows/opencv/build/x64/vc16/lib/opencv_world4120.lib" )
set(FFMPEG_DIR "E:/ffmpeg-master-latest-win64-gpl-shared")
set(CUDA_LIBS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFMPEG_DIR}/include)
include_directories(${TRTYOLO_INSTALL_DIR}/include)
include_directories(include)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")
include_directories(${NVCODEC_SDK_ROOT}/Interface)



set(CMAKE_CUDA_ARCHITECTURES 89)
add_library(gpu_kernels STATIC src/cu_kernels.cu)
set_target_properties(gpu_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)  # 添加
#target_compile_options(gpu_kernels PRIVATE "$$   <   $$<COMPILE_LANGUAGE:CUDA>:-c>")  # 避免 -o 冲突
target_link_libraries(gpu_kernels PRIVATE cuda  cudart)  # 确保依赖

add_executable(RealTimeDectectVer0.3
                src/main.cpp
                src/ConfigLoader.cpp
                src/ResultDrawer.cpp
                src/GpuDecoder.cpp      # <--- 新增：我们封装的解码类
                src/NvDecoder.cpp

                include/NvDecoder.h
                include/NvCodecUtils.h
                include/Logger.h
                include/nvcuvid.h
                include/cu_kernels.h
                include/nvEncodeAPI.h
                include/FFmpegDemuxer.h

                include/GpuDecoder.hpp
                include/ConfigLoader.hpp
                include/ResultDrawer.hpp)

target_link_libraries(RealTimeDectectVer0.3 PRIVATE
    ${TRTYOLO_INSTALL_DIR}/lib/trtyolo.lib
    ${TRTYOLO_INSTALL_DIR}/lib/custom_plugins.lib
    ${OpenCV_LIBS} 

    ${FFMPEG_DIR}/lib/avformat.lib
    ${FFMPEG_DIR}/lib/avcodec.lib
    ${FFMPEG_DIR}/lib/swscale.lib
    ${FFMPEG_DIR}/lib/avutil.lib


    ${CUDA_LIBS}/cuda.lib
    E:/Video_Codec_SDK_13.0.37/Lib/win/x64/nvcuvid.lib
    ${CUDA_LIBS}/cudart.lib
    
    gpu_kernels
)



