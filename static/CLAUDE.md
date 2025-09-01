# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the UnetOnnxSegDLL static library.

## Library Overview

This is a medical image segmentation DLL library that provides C API for 3D medical image segmentation using ONNX Runtime and nnUNet architecture. The library wraps a modular C++ implementation with a C-style API for easy integration.

## Architecture (Refactored 2025-09-01)

### Modular Structure
The codebase has been refactored from a monolithic design into a modular architecture:

```
UnetMain (Controller) ~830 lines
    ├── UnetPreprocessor (Preprocessing Module) ~397 lines
    ├── UnetInference (Inference Module) ~422 lines  
    └── UnetPostprocessor (Postprocessing Module) ~145 lines
```

### API Layer (C Interface)
- **UnetSegAI_API.h/cpp** - C-style API wrapper that exports DLL functions
- Provides handle-based interface to hide C++ implementation details
- All functions return status codes for error handling

### Core Modules (C++)

1. **UnetMain.h/cpp** - Main controller and coordinator
   - Manages configuration and workflow
   - Handles I/O operations and file saving
   - Coordinates between different modules

2. **UnetPreprocessor.h/cpp** - Preprocessing module
   - `cropToNonzero()` - Crop to non-zero region
   - `createSegMask()` - Create segmentation mask for normalization
   - `normalizeVolume()` - CT or Z-score normalization
   - `resampleVolume()` - Resample to target spacing
   - `binaryFillHoles3D()` - Fill internal holes in 3D

3. **UnetInference.h/cpp** - Inference module
   - `segModelInfer()` - Complete inference pipeline
   - `slidingWindowInfer()` - Sliding window inference with ONNX Runtime
   - `createGaussianKernel()` - Create 3D Gaussian weights
   - `argmax_spectrum()` - Convert probabilities to labels

4. **UnetPostprocessor.h/cpp** - Postprocessing module
   - `processSegmentationMask()` - Complete postprocessing pipeline
   - `revertTranspose()` - Revert coordinate transpose
   - `restoreOriginalSize()` - Restore to original dimensions

5. **ConfigParser.h/cpp** - JSON configuration parser
   - Parses nnUNet configuration from JSON files
   - Supports model parameters and normalization settings

### Key Components

1. **Data Structure**:
   ```c
   AI_DataInfo - Contains 3D volume data, voxel spacing, and origin information
   ```

2. **API Functions**:
   - `UnetSegAI_CreateObj()` - Create segmentation instance
   - `UnetSegAI_SetModelPath()` - Set ONNX model path
   - `UnetSegAI_SetTileStepRatio()` - Configure sliding window overlap
   - `UnetSegAI_Infer()` - Run segmentation inference
   - `UnetSegAI_GetResult()` - Retrieve segmentation mask
   - `UnetSegAI_SetConfigFromJson()` - Load configuration from JSON
   - `UnetSegAI_ReleseObj()` - Release resources

3. **nnUNet Configuration**:
   - Input channels: Configurable (default 1)
   - Output classes: Configurable (default 3)
   - Patch size: Configurable (e.g., 96x160x160)
   - Step ratio: Configurable (default 0.5)
   - Normalization: CT or Z-score normalization
   - Support for mask-based normalization

## Dependencies

- **ONNX Runtime** - For neural network inference
- **CImg** - For image processing operations
- **ITK** (optional) - For NIfTI file I/O
- **CUDA** (optional) - For GPU acceleration
- **nlohmann/json** - For JSON parsing

## Build Configuration

When building this as a DLL:
- Export symbols using `__declspec(dllexport)` 
- Define `UNETONNXSEGDLL_EXPORTS` macro
- Link against ONNX Runtime and CImg libraries
- Requires C++11 or later

## Processing Pipeline

1. **Preprocessing** (UnetPreprocessor):
   - Transpose to model's expected orientation
   - Crop to non-zero region
   - Normalize (CT or Z-score)
   - Resample to target spacing

2. **Inference** (UnetInference):
   - Pad volume if needed
   - Sliding window inference with overlap
   - Gaussian weighting for overlapping regions
   - Aggregate predictions

3. **Postprocessing** (UnetPostprocessor):
   - Argmax to get class labels
   - Revert transpose to original orientation
   - Restore to original size and position
   - Copy metadata (origin, spacing)

## Error Handling

The API uses status codes:
- `UnetSegAI_STATUS_SUCCESS` (0) - Success
- `UnetSegAI_STATUS_HANDLE_NULL` (1) - Invalid handle
- `UnetSegAI_STATUS_VOLUME_SMALL` (2) - Input volume too small
- `UnetSegAI_STATUS_VOLUME_LARGE` (3) - Input volume too large
- `UnetSegAI_STATUS_CROP_FAIED` (4) - ROI detection failed
- `UnetSegAI_STATUS_FAIED` (5) - General segmentation failure
- `UnetSegAI_LOADING_FAIED` (6) - Model loading failed

## Recent Refactoring (2025-09-01)

### Changes Made:
1. **Renamed all components** from "Dental" prefix to "Unet" prefix
2. **Modularized monolithic code** into separate preprocessing, inference, and postprocessing modules
3. **Improved separation of concerns** with friend class pattern for module access
4. **Removed redundant functions** and consolidated duplicate code
5. **Enhanced precision** by using double for statistical calculations

### Migration Notes:
- All `DentalCbctSegAI_*` functions renamed to `UnetSegAI_*`
- `DentalUnet` class renamed to `UnetMain`
- Library name changed from `DentalCbctOnnxSegDLL` to `UnetOnnxSegDLL`
- Function `initializeOnnxruntimeInstances` renamed to `setOnnxruntimeInstances`

## Known Issues

1. String encoding: Some comments contain non-UTF8 characters (GBK encoding)
2. GPU memory issues may occur with large volumes when using CUDA provider
3. IO operations still in UnetMain, could be further modularized