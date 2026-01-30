#pragma once
#include <string>
#include <map>

struct AppConfig {
    std::string udp_url;
    int width;
    int height;
    int fps;
    
    std::string engine_path;
    std::string onnx_path;
    float conf_thres;
    float iou_thres;
    
    int gpu_id;
    bool use_fp16;
};

class ConfigLoader {
public:
    static AppConfig load(const std::string& path);
private:
    static std::string trim(const std::string& str);
};