#include "OrtFusionEngine.hpp"
#include <iostream>
#include <algorithm>
OrtFusionEngine::OrtFusionEngine(const std::string& model_path, int gpu_id) 
    : ir_blob_(image_area_), vis_blob_(image_area_) {
       std::cout << "[FusionEngine] 初始化 ONNX Runtime 环境..." << std::endl;
      env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FusionModel");
       memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::SessionOptions session_options; 
    session_options.SetIntraOpNumThreads(1); 
     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = gpu_id;
   
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    std::cout << "[FusionEngine] 正在使用全零张量对 GPU 进行预热 (空跑 5 次)..." << std::endl;
  
    std::fill(ir_blob_.begin(), ir_blob_.end(), 0.0f);
    std::fill(vis_blob_.begin(), vis_blob_.end(), 0.0f);
    
    Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, ir_blob_.data(), ir_blob_.size(), input_shape_.data(), input_shape_.size());
    Ort::Value vis_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, vis_blob_.data(), vis_blob_.size(), input_shape_.data(), input_shape_.size());
    

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(ir_tensor));
    input_tensors.push_back(std::move(vis_tensor));

    for (int w = 0; w < 5; ++w) {
        session_.Run(Ort::RunOptions{nullptr}, 
                     input_names_.data(), input_tensors.data(), input_names_.size(), 
                     output_names_.data(), output_names_.size());
    }
    std::cout << "[FusionEngine] 预热完毕！引擎已达到最大功率就绪状态。" << std::endl;
}

cv::Mat OrtFusionEngine::infer(const cv::Mat& ir_img, const cv::Mat& vis_img) {
    if (ir_img.empty() || vis_img.empty()) {
        throw std::invalid_argument("[FusionEngine] 输入图像为空！");
    }

    cv::Mat ir_f32, vis_f32;
    ir_img.convertTo(ir_f32, CV_32FC1, 1.0 / 255.0);
    vis_img.convertTo(vis_f32, CV_32FC1, 1.0 / 255.0);
    
    cv::resize(ir_f32, ir_f32, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
    cv::resize(vis_f32, vis_f32, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);

    std::memcpy(ir_blob_.data(), ir_f32.data, image_area_ * sizeof(float));
    std::memcpy(vis_blob_.data(), vis_f32.data, image_area_ * sizeof(float));


    Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, ir_blob_.data(), ir_blob_.size(), input_shape_.data(), input_shape_.size());
    Ort::Value vis_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, vis_blob_.data(), vis_blob_.size(), input_shape_.data(), input_shape_.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(ir_tensor));
    input_tensors.push_back(std::move(vis_tensor));


    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), input_tensors.data(), input_names_.size(), 
        output_names_.data(), output_names_.size()
    );

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    cv::Mat fused_mat(height_, width_, CV_8UC1);

    for (int p = 0; p < image_area_; ++p) {
        float val = floatarr[p] * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        fused_mat.data[p] = static_cast<uchar>(val);
    }

    return fused_mat;
}