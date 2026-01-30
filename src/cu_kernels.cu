#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// 简单的 NV12 转 BGR (Packed) 内核
// 显存中：BGRBGRBGR...
__global__ void Nv12ToBgrKernel(uint8_t* pNv12, int nNv12Pitch, uint8_t* pBgr, int nBgrPitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // NV12 结构: Y 平面全分辨率，UV 平面半分辨率交错 (UVUV...)
        int yIdx = y * nNv12Pitch + x;
        int uvIdx = (y / 2) * nNv12Pitch + (height * nNv12Pitch) + (x / 2) * 2;

        uint8_t Y = pNv12[yIdx];
        uint8_t U = pNv12[uvIdx];
        uint8_t V = pNv12[uvIdx + 1];

        // 整数运算转换公式 (标准 BT.601)
        int c = Y - 16;
        int d = U - 128;
        int e = V - 128;

        int B = (298 * c + 516 * d + 128) >> 8;
        int G = (298 * c - 100 * d - 208 * e + 128) >> 8;
        int R = (298 * c + 409 * e + 128) >> 8;

        // 写入 BGR (注意边界截断)
        int bgrIdx = y * nBgrPitch + x * 3;
        pBgr[bgrIdx + 0] = (B < 0) ? 0 : ((B > 255) ? 255 : B);
        pBgr[bgrIdx + 1] = (G < 0) ? 0 : ((G > 255) ? 255 : G);
        pBgr[bgrIdx + 2] = (R < 0) ? 0 : ((R > 255) ? 255 : R);
    }
}

void LaunchNv12ToBgr(uint8_t* d_src, int src_pitch, uint8_t* d_dst, int dst_pitch, int width, int height, cudaStream_t stream) {
    dim3 threads(32, 32);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    Nv12ToBgrKernel<<<blocks, threads, 0, stream>>>(d_src, src_pitch, d_dst, dst_pitch, width, height);
}