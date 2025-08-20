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
- `model/` - ONNX model files (kneeseg_test.onnx)
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

## Common Issues and Fixes

### Fixed Issues
1. **Vector subscript out of range error in DentalCbctOnnxSegDLL** (Fixed)
   - Location: `static/src/DentalUnet.cpp`, line 204-205
   - Error: "vector subscript out of range" assertion failure
   - Cause: Accessing `config.patch_size[3]` when vector only has 3 elements
   - Fix: Changed to `config.patch_size[i]` to use correct loop index

2. **ONNX Runtime memory management and segmentation fault** (Fixed)
   - Location: `static/src/DentalUnet.cpp`, line 357-358
   - Error: Segmentation fault after "Invalid input name" error
   - Cause: `GetInputNameAllocated()` returns an `AllocatedStringPtr` smart pointer. Using `.get()` directly caused the pointer to be freed when the smart pointer went out of scope
   - Fix: Store the `AllocatedStringPtr` objects to keep them alive throughout the function:
     ```cpp
     Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
     Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
     const char* input_name = input_name_ptr.get();
     const char* output_name = output_name_ptr.get();
     ```
   - This ensures the memory remains valid when used later in `session.Run()`

### General Issues
1. Missing .img file: Analyze format requires both .hdr and .img files
2. DLL dependencies: Ensure all ONNX Runtime and DentalCbctOnnxSegDLL DLLs are in the executable directory
3. Path issues: The executable expects to run from build\bin\Release\ with resources in the project root
4. Model path: Must use wide character string (wchar_t*) when calling DentalCbctSegAI_SetModelPath

## Current Development Priority: 硬编码参数解除

### 关键问题
当前静态库存在严重的硬编码参数问题，导致不同模型无法正确运行。特别是：
- **patch_size硬编码**: {160,160,96} vs 模型期望 {128,128,128}
- **num_classes硬编码**: 3 vs 模型期望 4
- **归一化方法硬编码**: CTNormalization vs 应该使用 ZScoreNormalization

### 需要实现的功能
1. **JSON配置文件支持**: 从`checkpoint_best_params.json`读取模型参数
2. **新API接口**: 添加`DentalCbctSegAI_SetConfigFile()`函数
3. **参数验证**: 模型加载时验证输入输出尺寸匹配

### 详细分析文档
参见 `hardcoded_parameters_analysis.md` 获取完整的参数清单和解决方案。

## Code Analysis Memories

- 不要使用模型文件名来分析代码，kneeseg_test仅仅是名称，并不以意味着它不能用于牙齿分割或其它分割
- 不要自己运行任何脚本，要求用户手动运行并粘贴结果
- **硬编码参数是当前最严重的问题**：必须首先解决参数配置化，然后才能支持多种模型