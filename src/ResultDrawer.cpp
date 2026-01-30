#include "ResultDrawer.hpp"
#include <string>

ResultDrawer::ResultDrawer() {}

cv::Scalar ResultDrawer::getColor(int cls_id) {
    // 使用简单的哈希算法生成伪随机颜色，确保同一个类别ID永远是同一个颜色
    int r = (cls_id * 123 + 45) % 255;
    int g = (cls_id * 234 + 99) % 255;
    int b = (cls_id * 345 + 12) % 255;
    return cv::Scalar(b, g, r); // OpenCV 使用 BGR
}

void ResultDrawer::draw(cv::Mat& img, const trtyolo::DetectRes& res, float conf_thres) {
    for (int i = 0; i < res.num; ++i) {
        float score = res.scores[i];
        
        // 再次过滤置信度
        if (score < conf_thres) continue;

        int cls_id = res.classes[i];
        const auto& box = res.boxes[i];

        // 1. 获取坐标
        int x = static_cast<int>(box.left);
        int y = static_cast<int>(box.top);
        int w = static_cast<int>(box.right - box.left);
        int h = static_cast<int>(box.bottom - box.top);
        cv::Rect rect(x, y, w, h);

        // 2. 获取颜色
        cv::Scalar color = getColor(cls_id);

        // 3. 绘制矩形框
        cv::rectangle(img, rect, color, 2);

        // 4. 绘制标签背景和文字
        std::string label = std::to_string(cls_id) + ": " + std::to_string(score).substr(0, 4);
        
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 确保标签不画出屏幕外
        int textY = y - 5;
        if (textY < textSize.height) {
            textY = y + textSize.height + 5;
        }

        // 绘制文字背景条（增加可读性）
        cv::rectangle(img, 
            cv::Point(x, textY - textSize.height - 2), 
            cv::Point(x + textSize.width, textY + baseline - 2), 
            color, 
            cv::FILLED);

        // 绘制白色文字
        cv::putText(img, label, cv::Point(x, textY - 2), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}