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
    // 主推理函数 - 滑窗推理
    static AI_INT runSlidingWindow(UnetMain* parent,
                                  const nnUNetConfig& config,
                                  const cimg_library::CImg<float>& input,
                                  cimg_library::CImg<float>& output,
                                  Ort::Env& env,
                                  Ort::SessionOptions& session_options,
                                  bool use_gpu);
    
    // 创建3D高斯核 - 改为public以便访问
    static void createGaussianKernel(cimg_library::CImg<float>& kernel, 
                                    const std::vector<int64_t>& patch_sizes);
    
private:
    // 执行单个patch的推理
    static AI_INT inferPatch(Ort::Session& session,
                            const cimg_library::CImg<float>& patch,
                            cimg_library::CImg<float>& output,
                            const std::vector<int64_t>& input_shape,
                            const char* input_name,
                            const char* output_name);
};

#endif // _UNET_INFERENCE_H_