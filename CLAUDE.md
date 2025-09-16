# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image segmentation application using ONNX Runtime for inference with CImg for image processing. The project implements segmentation on 3D medical images.

## Build Commands

To build the project, use the provided batch file:
```bash
build.bat
```

This will:
1. Create a `build` directory
2. Generate Visual Studio 2022 solution files
3. Build the project in Release mode
4. Copy required DLLs to the output directory

The executable will be located at: `build\bin\Release\testToothSegmentation.exe`

## Project Structure

Key directories:
- `src/` - Source code (testToothSegmentation.cpp)
- `header/` - Project headers (DentalCbctSegAI_API.h)
- `lib/` - Libraries and dependencies
  - `onnxruntime/` - ONNX Runtime libraries and headers
  - `CImg/` - CImg image processing library
  - `DentalCbctOnnxSegDLL.dll/lib` - Core segmentation library
- `model/` - ONNX model files
- `img/` - HDR format medical images
- `result/` - Output directory for segmentation results

## Path Configuration

When running from `build\bin\Release\`, paths to resources use relative paths:
- HDR images: `..\\..\\..\\img\\[filename].hdr`
- ONNX models: `..\\..\\..\\model\\[filename].onnx`

## Key APIs

The project uses the DentalCbctSegAI API with these main functions:
- `DentalCbctSegAI_CreateObj()` - Initialize segmentation model
- `DentalCbctSegAI_SetModelPath()` - Set ONNX model path
- `DentalCbctSegAI_SetTileStepRatio()` - Configure tile processing
- `DentalCbctSegAI_Infer()` - Run inference
- `DentalCbctSegAI_GetResult()` - Retrieve segmentation results
- `DentalCbctSegAI_ReleseObj()` - Release resources

Status codes:
- `DentalCbctSegAI_STATUS_SUCCESS` (0) - Success
- `DentalCbctSegAI_STATUS_HANDLE_NULL` (1) - Null handle error
- `DentalCbctSegAI_LOADING_FAIED` (6) - Model loading failed

## Input/Output Format

- Input: Analyze format medical images (.hdr/.img pair)
- Output: Analyze format label mask with segmentation results

## Known Issues

### General Issues
1. Missing .img file: Analyze format requires both .hdr and .img files
2. DLL dependencies: Ensure all ONNX Runtime and DentalCbctOnnxSegDLL DLLs are in the executable directory
3. Path issues: The executable expects to run from build\bin\Release\ with resources in the project root
4. Model path: Must use wide character string (wchar_t*) when calling DentalCbctSegAI_SetModelPath

## Current Development Priority

### JSON配置文件支持已实现
静态库已支持从JSON配置文件读取模型参数，解决了硬编码问题：
- 支持从`checkpoint_best_params.json`读取模型参数
- 自动根据配置文件设置patch_size、num_classes、归一化方法等参数
- 模型加载时验证输入输出尺寸匹配

## Important Notes

- 不要使用模型文件名来分析代码功能（如kneeseg_test仅仅是名称）
- 不要自己运行任何脚本，要求用户手动运行并粘贴结果

## 维度映射关键经验（2025-01-09）

### 数据流
```
ITK [X,Y,Z] → CImg(X,Y,Z) → UnetPreprocessor → CImg(W,H,D) → UnetTorchInference/UnetInference
                ↓                                     ↓
          (width,height,depth)              (width,height,depth)
```

### 完整维度对应表

| 维度 | ITK | CImg | JSON patch_size | PyTorch/LibTorch | ONNX Runtime | 说明 |
|------|-----|------|----------------|------------------|--------------|------|
| **数据布局** | [X, Y, Z] | (W, H, D, C) | [D, H, W] | [N, C, D, H, W] | [N, C, D, H, W] | 不同框架的内存布局 |
| **Width (X)** | size[0] | 第1维 | patch_size[2] | 第5维 | 第5维 | 横向尺寸 |
| **Height (Y)** | size[1] | 第2维 | patch_size[1] | 第4维 | 第4维 | 纵向尺寸 |
| **Depth (Z)** | size[2] | 第3维 | patch_size[0] | 第3维 | 第3维 | 深度/层数 |

### 具体示例（patch_size = [128, 160, 112]）

| 框架 | 实际表示 | 维度顺序说明 |
|------|---------|------------|
| **JSON配置** | `[128, 160, 112]` | [depth, height, width] |
| **CImg创建** | `CImg(112, 160, 128, 1)` | (width, height, depth, channels) |
| **PyTorch Tensor** | `[1, 1, 128, 160, 112]` | [batch, channel, depth, height, width] |
| **ONNX Input** | `[1, 1, 128, 160, 112]` | [batch, channel, depth, height, width] |

### 正确的维度映射代码
```cpp
// UnetTorchInference.cpp 中的正确实现
// 创建 CImg patch
CImg<float> input_patch(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1);
// 即: (112, 160, 128, 1) = (width, height, depth, channels)

// 转换为 tensor
torch::Tensor input_tensor = cimgToTensor(
    input_patch,
    {1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2]},
    device
);
// 即: [1, 1, 128, 160, 112] = [batch, channel, depth, height, width]
```

### Tile混乱问题的根本原因和解决方案

**问题**：盲目照搬参考代码`DentalUnet_cimg_version.cpp`的维度映射，导致模型收到错误维度的输入。

**原因**：
1. 参考代码可能使用不同的预处理器或数据加载方式
2. 我们的数据流是：ITK → CImg → UnetPreprocessor → Model
3. 参考代码的维度映射不适用于我们的数据流

**解决方案**：
1. 理解自己项目的数据流，不要机械复制代码
2. 确保CImg patch尺寸正确：`(patch_size[2], patch_size[1], patch_size[0])`
3. 确保tensor shape与模型期望一致：`[1, 1, patch_size[0], patch_size[1], patch_size[2]]`

**关键教训**：
- 必须理解数据在每个处理阶段的具体格式
- 不同框架（ITK、CImg、PyTorch、ONNX）有不同的维度约定
- 从错误信息（如"Expected size 8 but got size 7"）可以快速定位维度问题

## LibTorch 配置重要经验

### 2025-01-09: 解决 LibTorch CUDA 检测失败问题

**问题描述**：
- LibTorch 2.8.0+cu129 无法检测 CUDA，即使系统有 CUDA 12.9 和所有必要的 DLL 文件
- `torch::cuda::is_available()` 始终返回 false
- 同样的库在独立测试程序中工作正常

**根本原因**：
CMake 配置方法错误。手动指定 LibTorch 路径和库文件会遗漏关键的编译标志和宏定义。

**解决方案**：
必须使用官方推荐的 `find_package(Torch)` 方法：

```cmake
# 正确方法
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 错误方法（不要使用）
# set(TORCH_PATH "...")
# set(TORCH_LIBRARIES "...lib/torch.lib" ...)
```

**关键教训**：
1. `find_package(Torch)` 会自动设置所有必要的编译标志、宏定义和依赖
2. `TORCH_CXX_FLAGS` 包含启用 CUDA 所需的关键编译选项
3. 手动配置容易遗漏重要设置，导致 CUDA 支持无法正确编译
4. 参考成功的示例代码（如 ref/env_test）来配置 CMake

## Bug Fixes and Improvements

### 2025-08-29: 修复浮点精度导致的统计误差

**问题描述**：
- C++版本在计算前景区域（非零区域）的mean和std时，结果与Python版本存在显著差异
- Python: mean=2905.92, std=2677.23
- C++修复前: mean=2789.03, std=2634.68
- 像素数量一致（6814617），但统计值错误导致归一化结果不正确

**根本原因**：
- 使用`float`类型累加大量数值（6百万个像素）时精度不足
- 累加总和约200亿，超出float的7位有效数字精度
- 累积的舍入误差导致最终mean偏低

**解决方案**：
- 将`DentalUnet.cpp`中所有统计计算从`float`改为`double`
- 包括：归一化前统计（754-771行）、mask-based归一化（858-881行）、归一化后统计（902-919行）
- double有15-16位有效数字，足以准确处理大规模累加

**修复效果**：
- mean和std与Python版本完全一致
- 改善了整体的dice score精度

### 2025-08-29: 修复patch_size维度顺序问题

**问题描述**：
- 从硬编码提取的JSON配置文件中patch_size顺序错误
- 错误顺序：[160, 160, 96]导致ONNX运行时报错
- 错误信息：index 2 Got: 160 Expected: 96

**根本原因**：
- 模型期望的维度顺序是[depth, height, width]即[96, 160, 160]
- 硬编码提取时保持了代码中的顺序，但模型实际需要不同的顺序

**解决方案**：
- 将patch_size从[160, 160, 96]调整为[96, 160, 160]
- 同时调整target_spacing从[0.581, 0.581, 1.0]到[1.0, 0.581, 0.581]保持一致性

**注意事项**：
- 不同模型训练时可能使用不同的维度顺序
- 需要根据模型实际要求调整JSON配置