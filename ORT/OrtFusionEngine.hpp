#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class OrtFusionEngine {
public:
        OrtFusionEngine(const std::string& model_path, int gpu_id = 0);
    ~OrtFusionEngine() = default;
    cv::Mat infer(const cv::Mat& ir_img, const cv::Mat& vis_img);

private:
    Ort::Env env_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::Session session_{nullptr};

    const int batch_size_ = 1;
    const int channels_ = 1;
    const int height_ = 256;
    const int width_ = 256;
    const int image_area_ = height_ * width_;

    std::vector<int64_t> input_shape_{batch_size_, channels_, height_, width_};

    std::vector<const char*> input_names_{"ir_image", "vis_image"};
    std::vector<const char*> output_names_{"fused_image"};

    std::vector<float> ir_blob_;
    std::vector<float> vis_blob_;
};