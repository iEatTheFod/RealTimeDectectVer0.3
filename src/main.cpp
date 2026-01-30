#include <iostream>
#include <opencv2/opencv.hpp> // 仅用于最后的 imshow (从 GPU 拷回来看效果)
#include "GpuDecoder.hpp"
#include "trtyolo.hpp"
#include "ResultDrawer.hpp" // 复用之前的绘图模块
#include "ConfigLoader.hpp"
#include "Logger.h"
#include <cuda_runtime.h>



#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        return -1; \
    } \
}

// 1. LogLevel 和 INFO 是全局定义的，不需要命名空间前缀。
// 2. 直接使用 SDK 提供的 LoggerFactory 来创建实例，不用自己写子类。
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(INFO);



int main() {
    // 1. 加载配置
    auto config = ConfigLoader::load("E:\\_PROJECT_CPP\\RealTimeDectectVer0.3\\config.ini");

    // 2. 初始化 TensorRT-YOLO
    trtyolo::InferOption option;
    option.setDeviceId(config.gpu_id);
    option.enableSwapRB(); 
    option.enableCudaMem(); // <--- 关键：开启 GPU 内存输入支持！
    
    trtyolo::DetectModel detector(config.engine_path, option);

    // 3. 启动 NvCodec 硬件解码器
    GpuDecoder decoder(config.udp_url, config.gpu_id);
    decoder.start();
    
    std::cout << "[Info] GPU Decoder started. Pure hardware pipeline." << std::endl;

    uint8_t* d_img_ptr = nullptr;
    int pitch = 0, w = 0, h = 0;
    
    // 用于显示的临时 CPU Mat
    cv::Mat cpu_frame;
    
    // 【新增】定义一个专门用于 YOLO 的扁平显存指针
    uint8_t* d_flat_input = nullptr;
    size_t flat_size = 0;

    while (true) {
        // 4. 从 GPU 解码器获取显存指针 (零拷贝)
        if (decoder.getLatestFrame(&d_img_ptr, &pitch, &w, &h)) {

        // 【新增】第一次运行时，分配扁平显存
            size_t needed_size = w * h * 3;
            if (d_flat_input == nullptr || flat_size != needed_size) {
                if (d_flat_input) cudaFree(d_flat_input);
                CHECK_CUDA(cudaMalloc(&d_flat_input, needed_size));
                flat_size = needed_size;
                std::cout << "[Info] Allocated flat GPU buffer: " << w << "x" << h << std::endl;
            }

            auto start = std::chrono::high_resolution_clock::now();  

            CHECK_CUDA(cudaStreamSynchronize(NULL)); // 等待关联完成

            // 【关键修复】使用 cudaMemcpy2D 将“带步长”的显存 整理为 “紧凑”的显存
            // 依然是 DeviceToDevice (纯 GPU 拷贝)，速度极快，但解决了崩溃问题
            CHECK_CUDA(cudaMemcpy2D(d_flat_input,       // 目标：紧凑内存
                                    w * 3,              // 目标步长：宽度*3 (无padding)
                                    d_img_ptr,          // 源：解码器输出
                                    pitch,              // 源步长：解码器提供的 pitch
                                    w * 3,              // 拷贝宽度：有效数据宽度
                                    h,                  // 拷贝高度
                                    cudaMemcpyDeviceToDevice));

            // 5. 构造 Image 对象 (直接传入 Device Pointer)
            // 假设 trtyolo::Image 支持传入显存地址（根据 enableCudaMem）
            trtyolo::Image img(d_flat_input, w, h, 3, w * 3);

            // 6. 推理 (数据全在显存，速度极快)
            auto res = detector.predict(img);

            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();

            // -----------------------------------------------------------
            // 可视化部分 (这步必须拷回 CPU，是唯一的耗时点，但仅用于显示)
            // 实际生产中，你可能只需要结果坐标，不需要回传图像
            // -----------------------------------------------------------
            if (cpu_frame.empty() || cpu_frame.size() != cv::Size(w, h)) {
                cpu_frame = cv::Mat(h, w, CV_8UC3);
            }
            // D2H 拷贝 (Device to Host)
            cudaMemcpy2D(cpu_frame.data, cpu_frame.step, d_img_ptr, pitch, w * 3, h, cudaMemcpyDeviceToHost);

            // 绘图
            ResultDrawer drawer;
            drawer.draw(cpu_frame, res, config.conf_thres);
            
            cv::putText(cpu_frame, "GPU Latency: " + std::to_string(latency) + "ms", 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);
            cv::imshow("NvCodec Low Latency", cpu_frame);
        }
        
        if (cv::waitKey(1) == 27) break;
    }

    decoder.stop();
    return 0;
}