#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <memory>
#include <iostream>

// 全局日志器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TrtFusionEngine {
public:
    
    TrtFusionEngine(const std::string& engine_path, int input_w = 256, int input_h = 256);
    

    ~TrtFusionEngine();

        cv::Mat infer(const cv::Mat& ir_src, const cv::Mat& vis_src, 
                  double& pre_ms, float& infer_ms, double& post_ms);

private:

    void preprocess(const cv::Mat& src, float* dst_blob);

    Logger logger_;
    int input_w_;
    int input_h_;
    int data_size_;


    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    cudaStream_t stream_;
    cudaEvent_t start_event_, stop_event_;
    
    float* d_ir_input_ = nullptr;
    float* d_vis_input_ = nullptr;
    float* d_output_ = nullptr;

    std::vector<float> h_ir_input_;
    std::vector<float> h_vis_input_;
    std::vector<float> h_output_;
};