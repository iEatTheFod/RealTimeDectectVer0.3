#pragma once
#include <cuda_runtime.h>
#include <cstdint> // 用于 uint8_t

// 这是一个 C++ 函数声明，对应 cu_kernels.cu 中的定义
// 注意：参数类型必须与 .cu 文件完全一致
void LaunchNv12ToBgr(uint8_t* d_src, int src_pitch, uint8_t* d_dst, int dst_pitch, int width, int height, cudaStream_t stream = 0);