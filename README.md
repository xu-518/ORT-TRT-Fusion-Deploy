# 双模态图像融合：高性能 C++ 部署框架

[![C++](https://img.shields.io/badge/C++-[14/17]-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-GPU-orange.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-High_Performance-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## 📖 项目简介
本项目提供了一个面向生产环境的高性能 C++ 部署框架，专为**红外与可见光双模态图像融合**模型设计。
项目采用彻底的模块化与按后端物理隔离的架构，将底层推理引擎与主控业务逻辑完全解耦。目前已完整实现基于 **ONNX Runtime (ORT)** 和 **TensorRT (TRT)** 的双后端支持，深度适配 CUDA 硬件加速，可高效处理双模态图像的同步输入，并输出单张高质量的语义融合图像。

## ✨ 核心特性
* **双模态处理架构**：原生支持红外与可见光图像的对齐输入，端到端输出特征融合结果。
* **物理隔离与模块化封装**：ORT 与 TRT 后端代码完全物理隔离，互不干涉。推理核心被严格封装在 `OrtFusionEngine` 和 `TrtFusionEngine` 类中，主控程序零侵入。
* **极致性能优化**：TensorRT 后端原生接管设备内存（Device Memory）与 CUDA 流（CUDA Stream）调度，最大化吞吐量。
* **零拷贝内存管理**：针对底层 Tensor 内存分配与映射进行了优化，避免冗余的图像数据拷贝。

## 📁 工程目录结构
```text
.
├── ORT/                            # ONNX Runtime 部署模块
│   ├── CMakeLists.txt              # ORT 专属构建配置
│   ├── main.cpp                    # ORT 业务调度与执行逻辑
│   ├── OrtFusionEngine.hpp         # ORT 融合引擎接口定义
│   └── OrtFusionEngine.cpp         # ORT 融合引擎核心实现
├── TRT/                            # TensorRT 部署模块
│   ├── CMakeLists.txt              # TRT 专属构建配置
│   ├── main.cpp                    # TRT 业务调度与执行逻辑
│   ├── TrtFusionEngine.hpp         # TRT 融合引擎接口定义
│   └── TrtFusionEngine.cpp         # TRT 融合引擎核心实现
└── README.md                       # 项目说明文档
```
## 🛠️ 环境依赖
在编译本项目前，请确保您的系统已安装以下依赖库：
* **CMake** (>= 3.10)
* **OpenCV** (推荐 4.x 版本)
* **ONNX Runtime** (推荐 GPU 版本，已在 v1.14+ 验证)
* **CUDA & cuDNN** (如果需要开启 GPU 加速推理)
## 📊 性能与精度基准测试 (Benchmark)

为了进行严谨的定量评估，我们在统一的硬件测试环境（**NVIDIA GeForce RTX 3090 GPU**）下，将本 C++ 部署框架（ORT / TRT）与原生 PyTorch 基线模型进行了端到端的全面对比。评估分为**推理效率**与**融合质量**两个维度。

### 1. 推理性能与加速比 (Speed & Throughput)
*以 PyTorch (FP32) 为基线，括号内为相对基线的性能提升幅度（延时降幅 / FPS增幅）。*

<div align="center">

| Framework | Precision | Latency (ms) | FPS | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch** (Baseline) | FP32 | 17.62 | 56.76 | 1.00× |
| **ORT (C++)** | FP32 | 10.93 (↑37.97%) | 91.47 (↑61.15%) | 1.61× |
| **TRT (C++)** | FP16 | 2.16 (↑87.74%) | 462.32 (↑714.52%) | 8.16× |
| **TRT (C++)** | INT8 | **1.64 (↑90.69%)** | **609.58 (↑973.96%)** | **10.74×** |

</div>

### 2. 融合图像质量客观指标 (Fusion Quality Metrics)
*涵盖边缘强度(EI)、空间频率(SF)、信息熵(EN)、互信息(MI)、平均梯度(AG)及视觉信息保真度(VIFF)。*

<div align="center">

| Framework | Precision | EI | SF | EN | MI | AG | VIFF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch** (Baseline) | FP32 | 59.044 | 16.242 | 7.087 | 14.173 | 5.687 | 0.409 |
| **ORT (C++)** | FP32 | 59.044 | 16.244 | 7.088 | 14.177 | 5.688 | 0.409 |
| **TRT (C++)** | FP16 | 59.047 | 16.242 | 7.087 | 14.173 | 5.688 | 0.409 |
| **TRT (C++)** | INT8 | 59.121 | 16.252 | 7.086 | 14.171 | 5.697 | 0.408 |

</div>

> **💡 结论分析：**
> 1. **极致加速**：相比于原生 PyTorch，基于 C++ 的 ONNX Runtime 获得了 `1.6×` 的稳定加速；而 TensorRT (FP16) 实现了 `8.16×` 的飞跃，INT8 量化后更是达到了令人惊叹的 `10.74×` 加速，帧率突破 **600 FPS**，完全满足严苛的硬实时（Hard Real-time）需求。
> 2. **精度无损**：在实现量级加速的同时，FP32 (ORT) 与 FP16 (TRT) 的各项融合质量客观指标（如空间频率 SF、互信息 MI 等）与基线模型几乎完全对齐；即使在 INT8 量化下，指标波动也微乎其微（VIFF 仅下降 0.001），真正做到了**速度与精度的完美平衡**。

## 📅 演进路线图
[√] 基于 ONNX Runtime 的 C++ 模块化推理引擎

[√] 接入 TensorRT 后端，最大化发挥 NVIDIA GPU 并行吞吐量

## ✉️ 交流与反馈
如果您在图像配准、多尺度融合网络开发或端侧 C++ 部署方面有任何探讨意向，欢迎提交 Issue。
