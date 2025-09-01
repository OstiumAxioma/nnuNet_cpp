#ifndef _UNET_PREPROCESSOR_H_
#define _UNET_PREPROCESSOR_H_
#pragma once

#include <vector>
#include <chrono>
#include "UnetMain.h"
#include "CImg.h"

// Forward declaration for UnetMain class only
class UnetMain;

class UnetPreprocessor {
public:
    // 主预处理函数
    static AI_INT preprocessVolume(UnetMain* parent, 
                                  nnUNetConfig& config, 
                                  cimg_library::CImg<short>& input_volume,
                                  cimg_library::CImg<float>& output_volume);
    
    // 裁剪到非零区域
    static cimg_library::CImg<short> cropToNonzero(const cimg_library::CImg<short>& input, CropBBox& bbox);
    
    // 创建前景/背景掩码
    static cimg_library::CImg<short> createSegMask(const cimg_library::CImg<short>& input);
    
    // CT归一化
    static void CTNormalization(cimg_library::CImg<float>& volume, const nnUNetConfig& config);
    
    // Z-Score归一化
    static void ZScoreNormalization(cimg_library::CImg<float>& volume, 
                                   const cimg_library::CImg<short>& seg_mask,
                                   const nnUNetConfig& config,
                                   double& intensity_mean,
                                   double& intensity_std);
    
    // 重采样
    static void resampleVolume(const cimg_library::CImg<float>& input,
                              cimg_library::CImg<float>& output,
                              const std::vector<int64_t>& output_size);
    
    // 统一的归一化接口
    static AI_INT normalizeVolume(cimg_library::CImg<float>& volume,
                                 const cimg_library::CImg<short>& seg_mask,
                                 const nnUNetConfig& config,
                                 double& intensity_mean,
                                 double& intensity_std);
    
    // 3D binary_fill_holes实现
    static void binaryFillHoles3D(cimg_library::CImg<bool>& mask);
};

#endif // _UNET_PREPROCESSOR_H_