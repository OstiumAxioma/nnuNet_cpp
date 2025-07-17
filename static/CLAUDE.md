# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the DentalCbctOnnxSegDLL static library.

## Library Overview

This is a dental/medical image segmentation DLL library that provides C API for CBCT (Cone Beam Computed Tomography) tooth segmentation using ONNX Runtime. The library wraps a C++ implementation (DentalUnet class) with a C-style API for easy integration.

## Architecture

### API Layer (C Interface)
- **DentalCbctSegAI_API.h/cpp** - C-style API wrapper that exports DLL functions
- Provides handle-based interface to hide C++ implementation details
- All functions return status codes for error handling

### Core Implementation (C++)
- **DentalUnet.h/cpp** - Main segmentation class using ONNX Runtime
- Implements sliding window inference for 3D volume processing
- Uses CImg for image processing and ONNX Runtime for neural network inference

### Key Components

1. **Data Structure**:
   ```c
   AI_DataInfo - Contains 3D volume data and voxel spacing information
   ```

2. **API Functions**:
   - `DentalCbctSegAI_CreateObj()` - Create segmentation instance
   - `DentalCbctSegAI_SetModelPath()` - Set ONNX model path
   - `DentalCbctSegAI_SetTileStepRatio()` - Configure sliding window overlap
   - `DentalCbctSegAI_Infer()` - Run segmentation inference
   - `DentalCbctSegAI_GetResult()` - Retrieve segmentation mask
   - `DentalCbctSegAI_ReleseObj()` - Release resources

3. **nnUNet Configuration**:
   - Input channels: 1 (grayscale medical images)
   - Output classes: 3 (background, structure 1, structure 2)
   - Patch size: 160x160x96 voxels
   - Default step ratio: 0.75 (25% overlap)
   - Normalization: CT normalization with predefined HU values

## Dependencies

- **ONNX Runtime** - For neural network inference
- **CImg** - For image processing operations
- **CUDA** (optional) - For GPU acceleration

## Build Configuration

When building this as a DLL:
- Export symbols using `__declspec(dllexport)`
- Link against ONNX Runtime and CImg libraries
- Requires C++11 or later

## Important Implementation Details

1. **Model Input**: Expects 3D grayscale medical images (short type)
2. **Preprocessing**: 
   - CT normalization using predefined mean/std values
   - Automatic voxel spacing adjustment
3. **Inference**: 
   - Sliding window approach for large volumes
   - Gaussian weighting for overlapping regions
4. **Output**: Segmentation mask with same dimensions as input

## Error Handling

The API uses status codes:
- `DentalCbctSegAI_STATUS_SUCCESS` (0) - Success
- `DentalCbctSegAI_STATUS_HANDLE_NULL` (1) - Invalid handle
- `DentalCbctSegAI_STATUS_VOLUME_SMALL` (2) - Input volume too small
- `DentalCbctSegAI_STATUS_VOLUME_LARGE` (3) - Input volume too large
- `DentalCbctSegAI_STATUS_CROP_FAIED` (4) - ROI detection failed
- `DentalCbctSegAI_STATUS_FAIED` (5) - General segmentation failure
- `DentalCbctSegAI_LOADING_FAIED` (6) - Model loading failed

## Known Issues

1. String encoding: Headers contain non-UTF8 characters (GBK encoding)
2. The actual model being used (kneeseg_test.onnx) may not match the dental segmentation purpose
3. GPU memory issues may occur with large volumes when using CUDA provider