#include "TrtFusionEngine.hpp"
#include <fstream>
#include <stdexcept>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__ << std::endl; \
            exit(-1); \
        } \
    } while (0)


TrtFusionEngine::TrtFusionEngine(const std::string& engine_path, int input_w, int input_h)
    : input_w_(input_w), input_h_(input_h), data_size_(input_w * input_h) {
    
    // 1. 初始化 CPU 缓存 用于存放推理的输入和输出数据
    h_ir_input_.resize(data_size_);
    h_vis_input_.resize(data_size_);
    h_output_.resize(data_size_);

    // 2. 读取并反序列化 Engine  
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.good()) {
        throw std::runtime_error("无法打开 Engine 文件: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    context_ = engine_->createExecutionContext();

    if (!context_) throw std::runtime_error("ExecutionContext 创建失败！");

    // 3. 分配显存
    //显存分配与绑定
    CHECK_CUDA(cudaMalloc((void**)&d_ir_input_,  data_size_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_vis_input_, data_size_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output_,    data_size_ * sizeof(float)));

    context_->setTensorAddress("ir_image",  d_ir_input_);
    context_->setTensorAddress("vis_image", d_vis_input_);
    context_->setTensorAddress("fused_image", d_output_);

    // 4. 初始化 CUDA 流与事件  用于管理异步任务队列，实现 CPU 与 GPU 流水线重叠
    CHECK_CUDA(cudaStreamCreate(&stream_));
    CHECK_CUDA(cudaEventCreate(&start_event_));
    CHECK_CUDA(cudaEventCreate(&stop_event_));

    // 5. 引擎预热
    std::cout << "[TrtFusionEngine] 正在执行 GPU 预热 (5 次空跑)..." << std::endl;
    for (int w = 0; w < 5; ++w) {
        context_->enqueueV3(stream_);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    std::cout << "[TrtFusionEngine] 初始化与预热完成！" << std::endl;
}

TrtFusionEngine::~TrtFusionEngine() {
    
    CHECK_CUDA(cudaFree(d_ir_input_));
    CHECK_CUDA(cudaFree(d_vis_input_));
    CHECK_CUDA(cudaFree(d_output_));
    CHECK_CUDA(cudaEventDestroy(start_event_));
    CHECK_CUDA(cudaEventDestroy(stop_event_));
    CHECK_CUDA(cudaStreamDestroy(stream_));

    if (context_) delete context_;
    if (engine_)  delete engine_;
    if (runtime_) delete runtime_;
}

//图像预处理函数
void TrtFusionEngine::preprocess(const cv::Mat& src, float* dst_blob) {
   
    cv::Mat resized, float_mat;
    cv::resize(src, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(float_mat, CV_32FC1, 1.0 / 255.0);

    if (!float_mat.isContinuous()) float_mat = float_mat.clone();
  
    std::memcpy(dst_blob, float_mat.data, data_size_ * sizeof(float));
}

cv::Mat TrtFusionEngine::infer(const cv::Mat& ir_src, const cv::Mat& vis_src, 
                               double& pre_ms, float& infer_ms, double& post_ms) {

    cv::TickMeter cpu_timer;

    // 1. 前处理
    cpu_timer.start();                     
    preprocess(ir_src, h_ir_input_.data());
    preprocess(vis_src, h_vis_input_.data());
    cpu_timer.stop();                         
    pre_ms = cpu_timer.getTimeMilli();         
    cpu_timer.reset();                         

    // 2. H2D 拷贝
    CHECK_CUDA(cudaMemcpyAsync(d_ir_input_, h_ir_input_.data(), data_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
   
    CHECK_CUDA(cudaMemcpyAsync(d_vis_input_, h_vis_input_.data(), data_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_));

    // 3. GPU 推理与计时
    CHECK_CUDA(cudaEventRecord(start_event_, stream_));
    context_->enqueueV3(stream_);
    CHECK_CUDA(cudaEventRecord(stop_event_, stream_));

    // 4. D2H 拷贝与同步
   
    CHECK_CUDA(cudaMemcpyAsync(h_output_.data(), d_output_, data_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    CHECK_CUDA(cudaEventElapsedTime(&infer_ms, start_event_, stop_event_));

    // 5. 后处理
    cpu_timer.start();
    cv::Mat fused_result(input_h_, input_w_, CV_8UC1);
    for (int row = 0; row < input_h_; ++row) {
        unsigned char* row_ptr = fused_result.ptr(row);
        for (int col = 0; col < input_w_; ++col) {
            float val = h_output_[row * input_w_ + col] * 255.0f;
            int ival = (int)(val + 0.5f);
            row_ptr[col] = (unsigned char)std::min(std::max(ival, 0), 255);
        }
    }
    cpu_timer.stop();
    post_ms = cpu_timer.getTimeMilli();

    return fused_result;
}