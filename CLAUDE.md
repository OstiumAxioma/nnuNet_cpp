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