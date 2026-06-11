#include <iostream>  
#include <vector>    
#include <string>   
#include <chrono>   
#include <opencv2/opencv.hpp> 
#include "OrtFusionEngine.hpp"
#include <cuda_runtime.h>     


void print_vram_usage(const std::string& stage_name) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        std::cerr << "获取显存失败: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }
    double free_mb = (double)free_byte / (1024.0 * 1024.0);
    double total_mb = (double)total_byte / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;
    std::cout << "[" << stage_name << "] 显存占用: " 
              << used_mb << " MB / " << total_mb << " MB" << std::endl;
}

int main() {

    std::string model_path = "/home/xushuo/code/test_cmake/test_fusion2-sim.onnx";
    std::string ir_dir = "/home/xushuo/code/test_cmake/ir_images/";
    std::string vis_dir = "/home/xushuo/code/test_cmake/vis_images/";
    std::string out_dir = "/home/xushuo/code/class-ort/fused_output/";


    std::vector<cv::String> ir_paths;
    cv::glob(ir_dir + "*.jpg", ir_paths, false);
    if (ir_paths.empty()) {
        std::cerr << "报错：未能找到红外图片！" << std::endl;
        return -1;
    }

   
    print_vram_usage("模型加载前");

    OrtFusionEngine engine(model_path, 3);

    
    print_vram_usage("模型加载&预热后");

    
    double total_inference_time_ms = 0.0;
    int success_count = 0;

    std::cout << "开始批量处理图片..." << std::endl;


    for (const auto& current_ir_path : ir_paths) {
        size_t last_slash = current_ir_path.find_last_of("/\\");
        std::string ir_filename = current_ir_path.substr(last_slash + 1);

        std::string vis_filename = ir_filename;
        size_t pos = vis_filename.find("IR");
        if (pos != std::string::npos) {
            vis_filename.replace(pos, 2, "VIS");
        }

        cv::Mat ir_mat = cv::imread(current_ir_path, cv::IMREAD_GRAYSCALE);
        cv::Mat vis_color = cv::imread(vis_dir + vis_filename, cv::IMREAD_COLOR);
        
        if (ir_mat.empty() || vis_color.empty()) continue;

   
        std::vector<cv::Mat> bgr_channels;
        cv::split(vis_color, bgr_channels);
        cv::Mat vis_mat = bgr_channels[2]; 

        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat fused_mat = engine.infer(ir_mat, vis_mat);
        
        auto end_time = std::chrono::high_resolution_clock::now();
   
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        total_inference_time_ms += inference_time.count();
        success_count++; 

        std::string out_filename = ir_filename;
        size_t ir_pos = out_filename.find("IR");
        if (ir_pos != std::string::npos) {
            out_filename.erase(ir_pos, 2); 
        }

        cv::imwrite(out_dir + out_filename, fused_mat);

        std::cout << "成功处理: " << ir_filename << " -> 保存为: " << out_filename 
                  << " | 耗时: " << inference_time.count() << " ms" << std::endl;
    }

    if (success_count > 0) {
        double avg_time_ms = total_inference_time_ms / success_count;
        std::cout << "\n================ 核心性能报告 ================\n";
        std::cout << "总处理图片数: " << success_count << " 张\n";
        std::cout << "平均推理延迟 (含前/后处理): " << avg_time_ms << " ms\n";
        std::cout << "等效系统吞吐量 (FPS): " << 1000.0 / avg_time_ms << " 帧/秒\n";
        std::cout << "=============================================\n";
    }

    print_vram_usage("运行结束时");

    return 0;
}