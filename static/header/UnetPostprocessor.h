#ifndef _UNET_POSTPROCESSOR_H_
#define _UNET_POSTPROCESSOR_H_
#pragma once

#include <chrono>
#include "CImg.h"
#include "UnetSegAI_API.h"  // 包含AI_DataInfo的完整定义

// Forward declarations
class UnetMain;
struct nnUNetConfig;
struct CropBBox;

class UnetPostprocessor {
public:
    // 主后处理函数 - 执行完整的后处理流程
    static AI_INT processSegmentationMask(UnetMain* parent, AI_DataInfo* dstData);
    
    // 恢复到原始尺寸
    static void restoreOriginalSize(const cimg_library::CImg<short>& input,
                                   cimg_library::CImg<short>& output,
                                   const CropBBox& bbox,
                                   int width0, int height0, int depth0);
    
    // 撤销转置操作
    static void revertTranspose(cimg_library::CImg<short>& volume,
                               const char* transpose_backward);
};

#endif // _UNET_POSTPROCESSOR_H_