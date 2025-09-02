#ifndef _UNET_INFERENCE_H_
#define _UNET_INFERENCE_H_
#pragma once

#include <vector>
#include <chrono>
#include "CImg.h"
#include "onnxruntime_cxx_api.h"

// Forward declarations
class UnetMain;
struct nnUNetConfig;
typedef int AI_INT;

class UnetInference {
public:
    // 主推理函数 - 执行完整的模型推理流程
    static AI_INT segModelInfer(UnetMain* parent, 
                              nnUNetConfig config, 
                              cimg_library::CImg<short> input_volume);
    
    // 归一化和重采样函数（不包括转置和裁剪）
    static AI_INT normalizeAndResample(UnetMain* parent, 
                                      nnUNetConfig& config, 
                                      cimg_library::CImg<short>& input_volume, 
                                      cimg_library::CImg<float>& output_volume);
    
    // 滑动窗口推理
    static AI_INT slidingWindowInfer(UnetMain* parent,
                                    nnUNetConfig config,
                                    cimg_library::CImg<float> normalized_volume);
    
    // 创建3D高斯核
    static void createGaussianKernel(cimg_library::CImg<float>& gaussisan_weight, 
                                    const std::vector<int64_t>& patch_sizes);
    
    // Argmax操作 - 将概率转换为类别标签
    static cimg_library::CImg<short> argmax_spectrum(const cimg_library::CImg<float>& input);
};

#endif // _UNET_INFERENCE_H_