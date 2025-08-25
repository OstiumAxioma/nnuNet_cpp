# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dental/medical image segmentation application using ONNX Runtime for inference with CImg for image processing. The project implements tooth segmentation on CBCT (Cone Beam Computed Tomography) 3D medical images.

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
- Output: Analyze format label mask where tooth structures are labeled:
  - For tooth k: pulp (3k), dentin (3k+1), crown/metal (3k+2)
  - Upper teeth: k = 1,2,3...
  - Lower teeth: k = -1,-2,-3...

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