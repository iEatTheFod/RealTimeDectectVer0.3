#include "ConfigLoader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

// 辅助函数：去除字符串首尾空格
std::string ConfigLoader::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

AppConfig ConfigLoader::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    std::cout<<"[Info] Config file loaded from: " << path << std::endl;
    std::map<std::string, std::string> configMap;
    std::string line;
    
    // 逐行读取并解析 key=value
    while (std::getline(file, line)) {
        // 跳过注释和空行
        std::string cleanLine = trim(line);
        if (cleanLine.empty() || cleanLine[0] == '#') continue;

        size_t delimiterPos = cleanLine.find('=');
        if (delimiterPos != std::string::npos) {
            std::string key = trim(cleanLine.substr(0, delimiterPos));
            std::string value = trim(cleanLine.substr(delimiterPos + 1));
            configMap[key] = value;
        }
    }

    AppConfig config;
    try {
        // 从 map 中提取数据，带默认值或错误检查
        config.udp_url = configMap.at("udp_url");
        config.width = std::stoi(configMap["width"]); // 如果没有会抛出异常
        config.height = std::stoi(configMap["height"]);
        config.fps = std::stoi(configMap["fps"]);

        config.engine_path = configMap.at("engine_path");
        config.onnx_path = configMap["onnx_path"];
        config.conf_thres = std::stof(configMap["conf_thres"]);
        config.iou_thres = std::stof(configMap["iou_thres"]);

        config.gpu_id = std::stoi(configMap["gpu_id"]);
        config.use_fp16 = (std::stoi(configMap["use_fp16"]) != 0);
    } catch (const std::exception& e) {
        throw std::runtime_error("Config parsing error (missing key or wrong format): " + std::string(e.what()));
    }

    return config;
}