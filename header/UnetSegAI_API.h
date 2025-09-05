#ifndef _UnetSegAI_API__h
#define _UnetSegAI_API__h


// ////////////////////////////////////////////////////////////////////////////
// File: UnetSegAI_API.h
// Author: ZhangWei
// Description: Medical Image Segmentation API Interface
//
// Create Date: 2025-4-30

// ////////////////////////////////////////////////////////////////////////////

#define UnetSegAI_API  extern "C" __declspec(dllexport)


#define UnetSegAI_STATUS_SUCCESS          0   // Success
#define UnetSegAI_STATUS_HANDLE_NULL      1   // Handle is NULL, please call UnetSegAI_CreateObj() first
#define UnetSegAI_STATUS_VOLUME_SMALL     2   // Input volume too small
#define UnetSegAI_STATUS_VOLUME_LARGE     3   // Input volume too large
#define UnetSegAI_STATUS_CROP_FAIED       4   // ROI detection failed
#define UnetSegAI_STATUS_FAIED            5   // Segmentation failed
#define UnetSegAI_LOADING_FAIED           6   // Failed to load AI model

// --------------------------------------------------------------------
//            Type Definitions
// --------------------------------------------------------------------
typedef unsigned char    AI_UCHAR;
typedef unsigned short   AI_USHORT;
typedef short            AI_SHORT;
typedef int              AI_INT;
typedef float            AI_FLOAT;
typedef void             AI_VOID;
typedef void*            AI_HANDLE;
typedef wchar_t*         AI_STRING;
typedef int              AI_BOOL;


// Data structure for input/output volumes
typedef struct
{
	AI_SHORT    *ptr_Data;      // Data pointer
	AI_INT       Width;         // Image width
	AI_INT       Height;        // Image height
	AI_INT       Depth;         // Image depth (number of slices)
	AI_FLOAT     VoxelSpacing;  // Voxel size (unit: mm) - deprecated, use VoxelSpacingX/Y/Z
	AI_FLOAT     VoxelSpacingX;  // Voxel spacing in X direction (unit: mm)
	AI_FLOAT     VoxelSpacingY;  // Voxel spacing in Y direction (unit: mm)
	AI_FLOAT     VoxelSpacingZ;  // Voxel spacing in Z direction (unit: mm)
	// Original spacing fields (before any resampling)
	// If not set or invalid, the library will use current spacing as original
	AI_FLOAT     OriginalVoxelSpacingX;  // Original voxel spacing in X (unit: mm)
	AI_FLOAT     OriginalVoxelSpacingY;  // Original voxel spacing in Y (unit: mm)
	AI_FLOAT     OriginalVoxelSpacingZ;  // Original voxel spacing in Z (unit: mm)
	// Origin fields for medical image geometry
	AI_FLOAT     OriginX;  // Origin X coordinate
	AI_FLOAT     OriginY;  // Origin Y coordinate
	AI_FLOAT     OriginZ;  // Origin Z coordinate
} AI_DataInfo;


// Create segmentation object
// Initialize the algorithm and required parameters (model, etc.)
// Returns NULL on failure, non-NULL handle on success
UnetSegAI_API AI_HANDLE    UnetSegAI_CreateObj();

// Set segmentation model file path
UnetSegAI_API AI_INT       UnetSegAI_SetModelPath(AI_HANDLE AI_Hdl, AI_STRING fn);

// Set sliding window overlap ratio (0-1, e.g., 0.5 = 50% overlap)
UnetSegAI_API AI_INT       UnetSegAI_SetTileStepRatio(AI_HANDLE AI_Hdl, AI_FLOAT ratio);

// Run segmentation inference on medical image volume
// AI_Hdl: Handle created by UnetSegAI_CreateObj()
// srcData: Input medical image data
UnetSegAI_API AI_INT       UnetSegAI_Infer(AI_HANDLE AI_Hdl, AI_DataInfo *srcData);

// Get segmentation result
// AI_Hdl: Handle created by UnetSegAI_CreateObj()
// dstData: Output segmentation mask
// Label values depend on the model configuration
UnetSegAI_API AI_INT       UnetSegAI_GetResult(AI_HANDLE AI_Hdl, AI_DataInfo *dstData);

// Optional: Set output paths for intermediate results (for debugging)
UnetSegAI_API AI_INT       UnetSegAI_SetOutputPaths(AI_HANDLE AI_Hdl, 
                                                                 AI_STRING preprocessPath, 
                                                                 AI_STRING modelOutputPath, 
                                                                 AI_STRING postprocessPath);

// Additional configuration interfaces
UnetSegAI_API AI_INT       UnetSegAI_SetPatchSize(AI_HANDLE AI_Hdl, AI_INT x, AI_INT y, AI_INT z);
UnetSegAI_API AI_INT       UnetSegAI_SetNumClasses(AI_HANDLE AI_Hdl, AI_INT classes);
UnetSegAI_API AI_INT       UnetSegAI_SetInputChannels(AI_HANDLE AI_Hdl, AI_INT channels);
UnetSegAI_API AI_INT       UnetSegAI_SetTargetSpacing(AI_HANDLE AI_Hdl, AI_FLOAT x, AI_FLOAT y, AI_FLOAT z);
UnetSegAI_API AI_INT       UnetSegAI_SetTransposeSettings(AI_HANDLE AI_Hdl, 
                                                                       AI_INT forward_x, AI_INT forward_y, AI_INT forward_z,
                                                                       AI_INT backward_x, AI_INT backward_y, AI_INT backward_z);
UnetSegAI_API AI_INT       UnetSegAI_SetNormalizationType(AI_HANDLE AI_Hdl, const char* type);
UnetSegAI_API AI_INT       UnetSegAI_SetIntensityProperties(AI_HANDLE AI_Hdl, 
                                                                         AI_FLOAT mean, AI_FLOAT std, 
                                                                         AI_FLOAT min_val, AI_FLOAT max_val,
                                                                         AI_FLOAT percentile_00_5, AI_FLOAT percentile_99_5);
UnetSegAI_API AI_INT       UnetSegAI_SetUseMirroring(AI_HANDLE AI_Hdl, AI_BOOL use_mirroring);

// Load configuration from JSON string
UnetSegAI_API AI_INT       UnetSegAI_SetConfigFromJson(AI_HANDLE AI_Hdl, const char* jsonContent);

// Release resources and free memory
UnetSegAI_API AI_VOID      UnetSegAI_ReleseObj(AI_HANDLE AI_Hdl);

#endif