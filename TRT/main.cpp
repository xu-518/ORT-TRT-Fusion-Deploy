#include "TrtFusionEngine.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <opencv2/opencv.hpp> // 确保引入 OpenCV

int main() {
    const std::string engine_path = "/home/xushuo/code/test-TensorRT/test_fusion2_int8-ptq.engine";
    std::string ir_dir  = "/home/xushuo/code/test-TensorRT/ir_images/";
    std::string vis_dir = "/home/xushuo/code/test-TensorRT/vis_images/";
    std::string out_dir = "/home/xushuo/code/class-trt/results/";

    std::system(("mkdir -p " + out_dir).c_str());

    // 🚀 1. 初始化并预热引擎
    TrtFusionEngine engine(engine_path, 256, 256);

    double total_pre = 0.0, total_post = 0.0;
    float total_infer = 0.0f;
    int success_count = 0;

    // 🌟 新增：动态获取红外目录下所有的 jpg 图片路径
    std::vector<cv::String> ir_paths;
    cv::glob(ir_dir + "*.jpg", ir_paths, false);
    int total_pairs = ir_paths.size();

    if (total_pairs == 0) {
        std::cerr << "报错：未找到任何图片，请检查输入路径！" << std::endl;
        return -1;
    }

    // 🚀 2. 基于真实文件的业务循环
    for (int i = 0; i < total_pairs; ++i) {
        std::string ir_path_full = ir_paths[i];
        
        // 提取带后缀的文件名，例如 "IR86.jpg"
        size_t last_slash = ir_path_full.find_last_of("/\\");
        std::string ir_filename = ir_path_full.substr(last_slash + 1);

        // 匹配对应的可见光文件名，例如 "VIS86.jpg"
        std::string vis_filename = ir_filename;
        size_t pos = vis_filename.find("IR");
        if (pos != std::string::npos) {
            vis_filename.replace(pos, 2, "VIS");
        }

        cv::Mat ir_src  = cv::imread(ir_path_full, cv::IMREAD_GRAYSCALE);
        cv::Mat vis_src = cv::imread(vis_dir + vis_filename, cv::IMREAD_GRAYSCALE);

        if (ir_src.empty() || vis_src.empty()) {
            std::cerr << "读取失败跳过: " << ir_filename << std::endl;
            continue;
        }

        double pre_ms = 0, post_ms = 0;
        float infer_ms = 0;

        // 🚀 3. 执行推理
        cv::Mat result = engine.infer(ir_src, vis_src, pre_ms, infer_ms, post_ms);

        std::string out_filename = ir_filename;
        size_t ir_pos = out_filename.find("IR");
        if (ir_pos != std::string::npos) {
            out_filename.erase(ir_pos, 2); 
        }
        
        // 保存图片（此时名字完全跟随原图）
        cv::imwrite(out_dir + out_filename, result);

        total_pre += pre_ms; total_infer += infer_ms; total_post += post_ms;
        success_count++;

        std::cout << "[" << std::setw(2) << success_count << "/" << total_pairs << "] 成功保存 " << out_filename << " | "
                  << "前处理: " << std::fixed << std::setprecision(2) << pre_ms << " ms | "
                  << "GPU推理: " << infer_ms << " ms | "
                  << "总计: " << (pre_ms + infer_ms + post_ms) << " ms" << std::endl;
    }

    // 🚀 4. 最终性能报告
    if (success_count > 0) {
        double avg_pre = total_pre / success_count;
        double avg_infer = total_infer / success_count;
        double avg_post = total_post / success_count;
        double avg_total = avg_pre + avg_infer + avg_post;

        double gpu_throughput = 1000.0 / avg_infer;
        double sys_throughput = 1000.0 / avg_total;

        std::cout << "\n================ 📊 性能统计报告 ================\n";
        std::cout << " 成功处理对数: " << success_count << " / " << total_pairs << "\n";
        std::cout << " 平均前处理耗时: " << std::fixed << std::setprecision(2) << avg_pre << " ms\n";
        std::cout << " 平均纯GPU推理:  " << std::fixed << std::setprecision(2) << avg_infer << " ms\n";
        std::cout << " 平均后处理保存: " << std::fixed << std::setprecision(2) << avg_post << " ms\n";
        std::cout << " 平均单对图像总: " << std::fixed << std::setprecision(2) << avg_total << " ms\n";
        std::cout << " -----------------------------------------------\n";
        std::cout << " 🚀 纯 GPU 理论吞吐量 : " << std::fixed << std::setprecision(2) << gpu_throughput << " FPS (对/秒)\n";
        std::cout << " ⚙️  实际系统总吞吐量 : " << std::fixed << std::setprecision(2) << sys_throughput << " FPS (对/秒)\n";
        std::cout << "=================================================\n";
    }

    return 0;
}