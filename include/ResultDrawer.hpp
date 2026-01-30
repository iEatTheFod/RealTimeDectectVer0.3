#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
// 引用库的头文件以识别 DetectRes 结构体
#include "trtyolo.hpp" 

class ResultDrawer {
public:
    ResultDrawer();
    
    /**
     * @brief 绘制检测结果
     * @param img 输入/输出图像
     * @param res 推理结果
     * @param conf_thres 置信度阈值
     */
    void draw(cv::Mat& img, const trtyolo::DetectRes& res, float conf_thres);

private:
    // 辅助：根据类别ID获取固定的颜色
    cv::Scalar getColor(int cls_id);
};