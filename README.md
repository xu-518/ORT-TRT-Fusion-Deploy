# 双模态图像融合：高性能 C++ 部署框架

[![C++](https://img.shields.io/badge/C++-[14/17]-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-GPU-orange.svg)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## 📖 项目简介
本项目提供了一个面向生产环境的高性能 C++ 部署框架，专为**红外与可见光双模态图像融合**模型设计。

项目采用彻底的模块化架构，将底层推理引擎与主控业务逻辑完全解耦。目前已完整实现基于 **ONNX Runtime (ORT)** 的后端支持，支持 CUDA 硬件加速，可高效处理双模态图像的同步输入，并输出单张高质量的语义融合图像。

## ✨ 核心特性
* **双模态处理架构**：原生支持红外与可见光图像的对齐输入，端到端输出特征融合结果。
* **极简模块化封装**：推理后端被严格封装在 `OrtFusionEngine` 类中，主控程序 `main.cpp` 零侵入，调用逻辑极度清爽。
* **零拷贝内存管理**：针对底层 Tensor 内存分配与映射进行了优化，避免冗余的图像数据拷贝。
* **跨平台标准构建**：提供标准的 `CMakeLists.txt`，支持在不同操作系统和算力平台上无缝编译部署。

## 📁 工程目录结构
```text
.
├── CMakeLists.txt          # CMake 标准构建配置脚本
├── main.cpp                # 业务主逻辑：读取双模态图像、调度引擎、保存结果
├── OrtFusionEngine.hpp     # ONNX Runtime 融合引擎接口定义
├── OrtFusionEngine.cpp     # ONNX Runtime 融合引擎核心实现
├── .gitignore              # Git 版本控制忽略规则
└── README.md               # 项目说明文档
```
## 🛠️ 环境依赖
在编译本项目前，请确保您的系统已安装以下依赖库：

CMake (>= 3.10)

OpenCV (推荐 4.x 版本)

ONNX Runtime (推荐 GPU 版本，已在 v1.14+ 验证)

CUDA & cuDNN (如果需要开启 GPU 加速推理)

## 🚀 快速开始
1. 准备模型文件
由于 GitHub 文件大小限制，训练好的 .onnx 模型文件未包含在此代码仓库中。

请先自行导出或下载融合模型的 ONNX 文件，并将其重命名为 fusion_model.onnx，放置在项目的根目录下。

2. 编译工程
在终端中进入项目根目录，依次执行以下命令：
```text
mkdir build && cd build
cmake ..
make -j4
```
3. 运行推理

提示：推理完成后的融合图像将被自动保存至项目指定的输出目录中。

## 📅 演进路线图
[√] 基于 ONNX Runtime 的 C++ 模块化推理引擎

[x] 接入 TensorRT 后端，最大化发挥 NVIDIA GPU 并行吞吐量

[x] 探索面向边缘端 NPU（如地平线 BPU）的低精度量化部署方案

## ✉️ 交流与反馈
如果您在图像配准、多尺度融合网络开发或端侧 C++ 部署方面有任何探讨意向，欢迎提交 Issue。
