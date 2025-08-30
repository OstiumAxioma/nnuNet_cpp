#ifndef _UNET_POSTPROCESSOR_H_
#define _UNET_POSTPROCESSOR_H_
#pragma once

#include <chrono>
#include "DentalUnet.h"
#include "CImg.h"

// Forward declaration for DentalUnet class only
class DentalUnet;

class UnetPostprocessor {
public:
    // 主后处理函数
    static AI_INT processSegmentationMask(DentalUnet* parent,
                                         cimg_library::CImg<float>& prob_volume,
                                         AI_DataInfo* dstData);
    
    // Argmax操作 - 从概率图得到分割结果
    static cimg_library::CImg<short> argmaxSpectrum(const cimg_library::CImg<float>& input);
    
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