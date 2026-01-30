#include "GpuDecoder.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>
#include <chrono>

// 引入我们自己写的 CUDA 内核声明 (在 cu_kernels.cu 中定义)
#include "cu_kernels.h" 

// 引入 CUDA Runtime API 头文件 (这是修复崩溃的关键)
#include <cuda_runtime.h>

// 链接 CUDA 静态库 (Windows MSVC 特有写法)
// 如果你的 CMakeLists.txt 已经正确链接，这两行其实可以省略，但加上更保险
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")

GpuDecoder::GpuDecoder(const std::string& uri, int gpu_id) 
    : uri_(uri), gpu_id_(gpu_id) 
{
    // =========================================================================
    // 【关键修复：解决 Context 冲突】
    // 
    // 问题背景：
    // 1. TensorRT (TRT) 库通常使用 CUDA Runtime API (隐式 Context)。
    // 2. NvDecoder SDK 默认使用 Driver API (需要显式 Context)。
    // 3. 如果我们用 cuCtxCreate 创建一个新 Context，显存就无法在解码器和 TRT 之间互通，
    //    导致 TRT 访问显存时崩溃 (core.cpp error)。
    //
    // 解决方案：
    // 不创建新 Context，而是"借用" Runtime API 已经激活的那个 Context。
    // =========================================================================

    // 1. 设置当前设备
    // 这步操作会初始化 CUDA Runtime，并在该设备上激活一个 Primary Context
    cudaError_t err = cudaSetDevice(gpu_id_);
    if (err != cudaSuccess) {
        std::cerr << "[Error] cudaSetDevice failed for ID " << gpu_id_ << ". Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA Set Device Failed. Check GPU ID.");
    }

    // 2. 获取当前绑定的上下文 (即 Runtime API 刚刚激活的那个)
    CUresult res = cuCtxGetCurrent(&cuContext_);
    
    // 如果获取不到 (极少见)，尝试强制触发 Runtime 初始化
    if (res != CUDA_SUCCESS || cuContext_ == nullptr) {
        cudaFree(0); // 空操作，强制 Runtime 初始化
        cuCtxGetCurrent(&cuContext_);
    }

    if (cuContext_ == nullptr) {
        throw std::runtime_error("Failed to get CUDA Context. GPU Driver or Runtime mismatch?");
    }

    std::cout << "[Info] GPU Decoder bound to CUDA Context: " << cuContext_ << " on GPU " << gpu_id_ << std::endl;

    // 3. 创建 CUDA 流 (用于异步执行颜色转换 kernel)
    cudaStreamCreate(&stream_);
}

GpuDecoder::~GpuDecoder() {
    stop(); // 停止线程

    // 释放 SDK 对象
    if (decoder_) { delete decoder_; decoder_ = nullptr; }
    if (demuxer_) { delete demuxer_; demuxer_ = nullptr; }

    // 释放显存 (使用 Runtime API)
    if (d_bgr_frame_) { cudaFree(d_bgr_frame_); d_bgr_frame_ = nullptr; }

    // 销毁流
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    
    // 注意：不要销毁 cuContext_！
    // 因为它是 Primary Context，归 CUDA Runtime 所有，销毁它会导致后续 TRT 调用崩溃。
}

void GpuDecoder::start() {
    if (running_) return;
    running_ = true;
    decode_thread_ = std::thread(&GpuDecoder::decodeLoop, this);
}

void GpuDecoder::stop() {
    running_ = false;
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }
}

void GpuDecoder::decodeLoop() {
    // 【重要】线程入口必须重新绑定设备和上下文
    cudaSetDevice(gpu_id_);
    cuCtxSetCurrent(cuContext_);

    // 1. 初始化 FFmpeg Demuxer (解封装器)
    // 这里的参数是为了降低 RTSP/UDP 流的延迟
    // 注意：如果你的 FFmpegDemuxer 类构造函数不接受 map，请只传 uri_.c_str()
    /* std::map<std::string, std::string> ffmpeg_opts;
    ffmpeg_opts["fflags"] = "nobuffer";       
    ffmpeg_opts["flags"] = "low_delay";       
    demuxer_ = new FFmpegDemuxer(uri_.c_str(), ffmpeg_opts); 
    */
    demuxer_ = new FFmpegDemuxer(uri_.c_str()); 

    // 2. 初始化 NvDecoder (解码器)
    // FFmpeg2NvCodecId: 将 FFmpeg 的 CodecID 转为 NVIDIA 的 CodecID
    decoder_ = new NvDecoder(cuContext_, true, FFmpeg2NvCodecId(demuxer_->GetVideoCodec()));
    int nVideoBytes = 0;
    uint8_t *pVideo = nullptr;

    std::cout << "[Info] Decoding loop started..." << std::endl;

    while (running_) {
        // 从 FFmpeg 获取一包压缩数据 (H.264/HEVC Packet)
        if (!demuxer_->Demux(&pVideo, &nVideoBytes)) {
            // Demux 失败通常意味着流结束或网络断开
            // 这里简单休眠重试，生产环境建议加重连逻辑
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue; 
        }

        // 送入 GPU 解码 (非阻塞，推入队列)
        int nFramesDecoded = decoder_->Decode(pVideo, nVideoBytes);

        // 如果解码器有输出帧
        if (nFramesDecoded > 0) {
            uint8_t* pNv12Frame = nullptr;
            
            // =========================================================
            // 【实时性优化策略：排空队列 (Drain Loop)】
            // 解码器为了性能可能会缓冲几帧。但在实时检测中，我们不想要缓冲。
            // 我们循环取出所有已经解好的帧，只保留【最后一张】(最新帧)。
            // 这样能把延迟压到最低。
            // =========================================================
            for (int i = 0; i < nFramesDecoded; i++) {
                pNv12Frame = decoder_->GetFrame(); 
            }

            // 确保拿到了有效的显存指针
            if (pNv12Frame != nullptr) {
                // 首次运行时：初始化 BGR 缓冲区
                if (d_bgr_frame_ == nullptr) {
                    frame_width_ = decoder_->GetWidth();
                    frame_height_ = decoder_->GetHeight();
                    
                    // 计算 BGR Pitch (每行字节数 = 宽 * 3)
                    bgr_pitch_ = frame_width_ * 3; 
                    
                    // 分配显存
                    //cudaMalloc(&d_bgr_frame_, bgr_pitch_ * frame_height_);
                    // 【修改为】：
                    // 使用 Managed 内存作为跨线程/跨上下文的桥梁
                    cudaMallocManaged(&d_bgr_frame_, bgr_pitch_ * frame_height_);

                    // 【建议加上】预取到 GPU，保证解码性能
                    // 注意：这里 gpu_id_ 和 stream_ 需要是你类成员变量
                    cudaMemPrefetchAsync(d_bgr_frame_, bgr_pitch_ * frame_height_, gpu_id_, stream_);





                    std::cout << "[Info] Video Info: " << frame_width_ << "x" << frame_height_ << std::endl;
                }

                // 加锁保护，防止推理线程同时读取 
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    
                    // 3. 调用 CUDA Kernel: NV12 (YUV420) -> BGR (Packed)
                    // 这个内核在 cu_kernels.cu 中定义，直接在 GPU 上运行，极快。
                    LaunchNv12ToBgr(
                        pNv12Frame, decoder_->GetDeviceFramePitch(), // 源地址 (NV12)
                        d_bgr_frame_, bgr_pitch_,                    // 目标地址 (BGR)
                        frame_width_, frame_height_,
                        stream_                                      // 指定流
                    );

                    // 等待流执行完毕，确保 kernel 跑完
                    cudaStreamSynchronize(stream_);
                    
                    // 标记："我有新数据了，推理线程快来拿"
                    has_new_frame_ = true;
                }
            }
        }
    }
}

// 主线程/推理线程调用此函数获取数据
bool GpuDecoder::getLatestFrame(uint8_t** pDevPtr, int* pitch, int* width, int* height) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 如果没有新帧，直接返回 false，让推理线程休息一下
    if (!has_new_frame_) return false;

    // 返回显存指针
    // 注意：这个 d_bgr_frame_ 指向的是 GPU 显存！
    // TensorRT 可以直接读取它，但 CPU (如 OpenCV imwrite) 不能直接读。
    *pDevPtr = d_bgr_frame_;
    *pitch = bgr_pitch_;
    *width = frame_width_;
    *height = frame_height_;
    
    // 清除标记，避免重复推理同一帧
    has_new_frame_ = false; 
    return true;
}