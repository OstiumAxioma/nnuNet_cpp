# nnuNet_cpp - 3D医学图像分割

基于ONNX Runtime的C++实现的3D医学图像分割程序

## 构建脚本说明

### 1. 主程序构建脚本：`build.bat`
- **用途**：构建主程序可执行文件（`testToothSegmentation.exe`）
- **位置**：项目根目录
- **功能**：
  - 创建构建目录
  - 生成Visual Studio 2022解决方案
  - 编译主程序
  - 从`lib/run`目录复制运行时DLL到可执行文件目录（由于代码库限制目前需要自行找云重拿取完整的版本对应的运行库）

### 2. 静态库构建脚本：`static/build.bat`
- **用途**：构建分割核心DLL库（`UnetOnnxSegDLL.dll`）
- **位置**：`static/`目录
- **功能**：
  - 构建核心分割库
  - 必须在主程序构建之前运行
  - 输出DLL到`lib/`目录

## 项目结构

### 根目录结构
```
nnuNet_cpp/
├── build.bat              # 主程序构建脚本
├── CMakeLists.txt         # 主程序CMake配置
├── CLAUDE.md             # AI助手指令文件
├── README.md             # 本文档
├── src/                  # 主程序源代码
├── header/               # 主程序头文件
├── static/               # 静态库源代码
├── lib/                  # 库文件和依赖项
│   ├── onnxruntime/      # ONNX Runtime库
│   ├── libtorch/         # LibTorch库（PyTorch C++ 2.8.0+cu129）
│   └── CImg/             # 图像处理库
├── model/                # 模型文件（.onnx/.pt）
├── img/                  # 输入医学图像
├── param/                # JSON配置文件
├── result/               # 输出分割结果
└── build/                # 构建输出目录
    └── bin/
        └── Release/      # 可执行文件位置
            └── testToothSegmentation.exe
```

### 静态库结构（`static/`目录）
```
static/
├── build.bat             # DLL构建脚本
├── CMakeLists.txt        # DLL的CMake配置
├── CLAUDE.md            # DLL专用说明
├── header/              # DLL头文件
│   ├── UnetSegAI_API.h
│   ├── UnetInference.h
│   ├── framework.h
│   └── pch.h
├── src/                 # DLL源文件
│   ├── UnetSegAI_API.cpp
│   ├── UnetInference.cpp
│   ├── dllmain.cpp
│   └── pch.cpp
└── build/               # DLL构建输出
```

## 依赖库版本

### ONNX Runtime
- **版本**：1.16.x（根据DLL名称推断）
- **必需的DLL文件**：
  - `onnxruntime.dll`
  - `onnxruntime_providers_shared.dll`
  - `onnxruntime_providers_cuda.dll`
- **获取来源**：[ONNX Runtime发布页面](https://github.com/Microsoft/onnxruntime/releases)

### CUDA运行时
- **版本**：CUDA 12.x
- **必需的DLL文件**：
  - `cudart64_12.dll`
  - `cublas64_12.dll`
  - `cublasLt64_12.dll`
  - `cufft64_11.dll`
  - `curand64_10.dll`
  - `cusolver64_11.dll`
  - `cusparse64_12.dll`
- **获取来源**：[NVIDIA CUDA工具包](https://developer.nvidia.com/cuda-downloads)

### cuDNN
- **版本**：cuDNN 9.x
- **必需的DLL文件**：
  - `cudnn64_9.dll`
  - `cudnn_*64_9.dll`（多个组件）
- **获取来源**：[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

### LibTorch（PyTorch C++）
- **版本**：2.8.0（支持 TorchScript .pt/.pth 模型，CUDA 12.9 支持）
- **获取来源**：
  - CUDA 12.9 版本：https://download.pytorch.org/libtorch/cu129/libtorch-win-shared-with-deps-2.8.0%2Bcu129.zip
  - CUDA 12.6 版本：https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.8.0%2Bcu126.zip
  - CPU 版本：https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip
- **必需的DLL文件（CPU推理）**：
  - 核心库：
    - `c10.dll` - 核心张量库（793KB）
    - `torch.dll` - 主入口加载器（9.5KB）
    - `torch_cpu.dll` - CPU运算实现（126MB）
    - `torch_global_deps.dll` - 全局依赖（9.5KB）
  - 额外必需依赖：
    - `fbgemm.dll` - Facebook GEMM优化矩阵运算（2.5MB）
    - `asmjit.dll` - JIT编译器（500KB）
    - `uv.dll` - libuv异步I/O（350KB）
    - `mkl_intel_thread.1.dll` - Intel MKL数学运算（5MB）
    - `mkl_avx2.1.dll` - Intel MKL AVX2优化（必需）
    - `mkl_def.1.dll` - Intel MKL默认实现（必需）
- **GPU推理额外需要**：
  - `c10_cuda.dll`（345KB）
  - `torch_cuda.dll`（836MB）
  - `caffe2_nvrtc.dll`（CUDA JIT编译）
- **总大小**：CPU版本约145MB，GPU版本约1.3GB
- **重要提示**：
  - 必须下载与系统 CUDA 版本匹配的 LibTorch 版本
  - 使用官方完整包 libtorch-win-shared-with-deps-2.8.0+cu129.zip

### 重要提示
所有运行时DLL必须放置在`lib/run/`目录中。构建脚本会自动将它们复制到可执行文件目录。

## 构建步骤

1. **首先构建静态库**：
   ```cmd
   cd static
   build.bat
   cd ..
   ```

2. **构建主程序**：
   ```cmd
   build.bat
   ```

## 程序运行流程

1. **初始化阶段**
   - 从`img/`目录加载HDR/Analyze格式的医学图像
   - 从`param/`目录加载JSON配置文件
   - 创建分割模型实例
   - 设置`model/`目录中的ONNX模型路径

2. **预处理阶段**
   - 将图像数据转换为浮点型
   - 应用CT标准化或Z-score标准化
   - 根据需要调整体素间距

3. **推理阶段**
   - 将体积分割为重叠的块（96×160×160）
   - 对每个块运行ONNX模型
   - 使用高斯加权合并块结果

4. **后处理阶段**
   - 应用argmax获取分割标签
   - 将坐标变换回原始空间
   - 保存结果到`result/`目录

## 数据位置说明

### 输入图像（`img/`目录）
- **格式**：支持多种格式（Analyze：.hdr + .img文件对；NIfTI：.nii.gz文件）
- **类型**：3D医学图像（CBCT扫描）
- **示例**：
  - `img/Series_5_Acq_2_0000.hdr`（Analyze格式）
  - `img/MNI152NLin6_res-1x1x1_T1w.nii.gz`（NIfTI格式）

### 配置文件（`param/`目录）
- **格式**：JSON配置文件（.json）
- **示例**：
  - `param/checkpoint_best_params.json`
  - `param/116.json`
  - `param/knee.json`
- **内容**：包含模型参数、预处理设置等

### 模型文件（`model/`目录）
- **格式**：ONNX模型文件（.onnx）
- **示例**：
  - `model/116.onnx`
  - `model/kneeseg_test.onnx`
- **注意**：模型应具有5D输入（批次、通道、深度、高度、宽度）

### 输出结果（`result/`目录）
- **格式**：NIfTI格式（.nii.gz文件）
- **类型**：3D标签掩码
- **示例**：`result/finalLabelMask.nii.gz`
- **标签值含义**：
  - 0：背景
  - 1+：不同的解剖结构（取决于模型）

## 部署注意事项

1. 确保`lib/run/`中的所有DLL都存在于可执行文件相同的目录中（Release目录）
2. 程序需要支持CUDA的NVIDIA GPU以获得最佳性能
3. 输入图像必须是受支持的医学图像格式
4. 运行前如果`result/`目录不存在，需要先创建

## 故障排除

1. **缺少DLL错误**：运行`build.bat`将复制所有必需的DLL
2. **CUDA错误**：确保安装了兼容的NVIDIA GPU驱动程序
3. **模型加载错误**：验证ONNX模型的兼容性和输入/输出名称
4. **内存错误**：大体积数据可能需要具有足够RAM的GPU

### LibTorch CUDA 配置问题（重要）

**问题现象**：
- 系统有 CUDA 12.9，GPU 能被检测到
- LibTorch DLL 文件存在（torch_cuda.dll, c10_cuda.dll）
- 但 `torch::cuda::is_available()` 返回 false

**根本原因**：
CMake 配置方法错误。手动配置 LibTorch 路径会遗漏关键的编译标志。

**错误方法**（不要使用）：
```cmake
set(TORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch")
set(TORCH_LIBRARIES "${TORCH_PATH}/lib/torch.lib" ...)
```

**正确方法**（必须使用）：
```cmake
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
```

**关键点**：
- 必须使用 `find_package(Torch REQUIRED)` - 官方推荐方法
- 必须添加 `${TORCH_CXX_FLAGS}` 到编译标志 - 包含所有 CUDA 相关宏
- find_package 会自动处理所有依赖和配置

### 维度映射问题（重要）

**问题现象**：
- PyTorch 模型推理时报错 "Expected size X but got size Y"
- 推理结果的 tile 放置完全混乱
- 输出图像维度错误

**根本原因**：
不同框架使用不同的维度约定，混淆这些约定会导致严重错误。

**完整维度对应表**：

| 维度 | ITK | CImg | JSON patch_size | PyTorch/LibTorch | ONNX Runtime | 说明 |
|------|-----|------|----------------|------------------|--------------|------|
| **数据布局** | [X, Y, Z] | (W, H, D, C) | [D, H, W] | [N, C, D, H, W] | [N, C, D, H, W] | 不同框架的内存布局 |
| **Width (X)** | size[0] | 第1维 | patch_size[2] | 第5维 | 第5维 | 横向尺寸 |
| **Height (Y)** | size[1] | 第2维 | patch_size[1] | 第4维 | 第4维 | 纵向尺寸 |
| **Depth (Z)** | size[2] | 第3维 | patch_size[0] | 第3维 | 第3维 | 深度/层数 |
| **Channels** | - | 第4维 | - | 第2维 | 第2维 | 通道数 |
| **Batch** | - | - | - | 第1维 | 第1维 | 批次大小 |

**具体示例（patch_size = [128, 160, 112]）**：

| 框架 | 实际表示 | 维度顺序说明 |
|------|---------|------------|
| **JSON配置** | `[128, 160, 112]` | [depth, height, width] |
| **CImg创建** | `CImg(112, 160, 128, 1)` | (width, height, depth, channels) |
| **PyTorch Tensor** | `[1, 1, 128, 160, 112]` | [batch, channel, depth, height, width] |
| **ONNX Input** | `[1, 1, 128, 160, 112]` | [batch, channel, depth, height, width] |
| **ITK Image** | `[112, 160, 128]` | [x_size, y_size, z_size] |

**正确的代码示例**：
```cpp
// 创建 CImg patch - 注意维度顺序
CImg<float> input_patch(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1);
// 即: (112, 160, 128, 1) = (width, height, depth, channels)

// 转换为 PyTorch tensor - 使用模型期望的顺序
torch::Tensor input_tensor = torch::from_blob(
    input_patch.data(),
    {1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2]},
    options
);
// 即: [1, 1, 128, 160, 112] = [batch, channel, depth, height, width]
```

**关键教训**：
- 不要盲目复制参考代码的维度映射
- 必须理解自己项目的数据流：ITK → CImg → Preprocessor → Model
- 始终验证每个阶段的实际维度

## API使用说明

静态库提供C风格的API以便集成：

```c
// 创建实例
AI_HANDLE handle = UnetSegAI_CreateObj();

// 从JSON设置配置
UnetSegAI_SetConfigFromJson(handle, jsonContent);

// 设置模型路径
UnetSegAI_SetModelPath(handle, L"path/to/model.onnx");

// 运行推理
AI_DataInfo* input_data = /* 准备输入数据 */;
UnetSegAI_Infer(handle, input_data);

// 获取结果
AI_DataInfo* output_data = /* 准备输出缓冲区 */;
UnetSegAI_GetResult(handle, output_data);

// 清理资源
UnetSegAI_ReleseObj(handle);
```

## VSCode 开发环境配置

### 配置文件说明
项目包含完整的 VSCode 配置，支持 C++ 开发、CMake 构建和调试：

- **`.vscode/c_cpp_properties.json`** - C++ 配置，包含所有必要的包含路径
- **`.vscode/settings.json`** - VSCode 工作区设置
- **`.vscode/tasks.json`** - CMake 构建任务
- **`.vscode/launch.json`** - 调试配置

### 解决 #include 错误
如果遇到 `#include` 错误，请：

1. **重新加载 VSCode 窗口**：按 `Ctrl+Shift+P`，输入 "Developer: Reload Window"
2. **重新配置 IntelliSense**：按 `Ctrl+Shift+P`，输入 "C/C++: Reset IntelliSense Database"
3. **检查 CMake 配置**：确保 CMake 扩展已安装并正确配置

### 推荐的 VSCode 扩展
- **C/C++** - Microsoft 官方 C++ 支持
- **CMake Tools** - CMake 项目支持
- **CMake** - CMake 语法高亮

## JSON配置文件支持

程序现在支持从JSON配置文件读取模型参数，解决了硬编码问题：

- 支持从`checkpoint_best_params.json`等文件读取模型参数
- 自动根据配置文件设置patch_size、num_classes、归一化方法等参数
- 模型加载时验证输入输出尺寸匹配

### 配置文件示例
```json
{
    "patch_size": [96, 160, 160],
    "num_classes": 2,
    "input_channels": 1,
    "target_spacing": [1.0, 0.581, 0.581],
    "normalization": "zscore",
    "use_mirroring": [false, false, false]
}
```