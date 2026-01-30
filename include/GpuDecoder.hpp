#pragma once
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <cuda_runtime.h>

// 引用 SDK 的类
#include "NvDecoder.h"
#include "FFmpegDemuxer.h"

class GpuDecoder {
public:
    GpuDecoder(const std::string& uri, int gpu_id = 0);
    ~GpuDecoder();

    void start();
    void stop();

    // 获取最新一帧的显存指针 (BGR格式)
    // 返回: 是否获取成功
    bool getLatestFrame(uint8_t** pDevPtr, int* pitch, int* width, int* height);

private:
    void decodeLoop();

    std::string uri_;
    int gpu_id_;
    CUcontext cuContext_ = nullptr;
    
    // SDK 组件
    FFmpegDemuxer* demuxer_ = nullptr;
    NvDecoder* decoder_ = nullptr;

    // 线程控制
    std::thread decode_thread_;
    std::atomic<bool> running_{false};

    // 显存缓冲区 (用于存放转换后的 BGR 图像)
    uint8_t* d_bgr_frame_ = nullptr;
    int bgr_pitch_ = 0;
    int frame_width_ = 0;
    int frame_height_ = 0;

    // 简单的同步机制 (只存最新一帧)
    std::mutex mtx_;
    bool has_new_frame_ = false;
    cudaStream_t stream_ = nullptr;
};