#include "UnetPreprocessor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>

using namespace std;
using namespace cimg_library;

// 主预处理函数 - 暂时保留在DentalUnet.cpp中，后续迁移
AI_INT UnetPreprocessor::preprocessVolume(UnetMain* parent, 
                                         nnUNetConfig& config, 
                                         CImg<short>& input_volume,
                                         CImg<float>& output_volume)
{
    // TODO: 从DentalUnet::segModelInfer迁移预处理逻辑
    return UnetSegAI_STATUS_SUCCESS;
}

// 裁剪到非零区域 - 从DentalUnet.cpp行453-541迁移
CImg<short> UnetPreprocessor::cropToNonzero(const CImg<short>& input, CropBBox& bbox)
{
    // TODO: 从DentalUnet::crop_to_nonzero迁移代码
    return input;
}

// CT归一化 - 从DentalUnet.cpp行1238-1253迁移
void UnetPreprocessor::CTNormalization(CImg<float>& input_volume, const nnUNetConfig& config)
{
    // TODO: 从DentalUnet::CTNormalization迁移代码
}

// Z-Score归一化
void UnetPreprocessor::ZScoreNormalization(CImg<float>& volume, 
                                          const CImg<short>& seg_mask,
                                          const nnUNetConfig& config,
                                          double& intensity_mean,
                                          double& intensity_std)
{
    // TODO: 从DentalUnet::segModelInfer中提取Z-Score归一化逻辑
}

// 重采样
void UnetPreprocessor::resampleVolume(const CImg<float>& input,
                                     CImg<float>& output,
                                     const std::vector<int64_t>& output_size)
{
    // TODO: 从DentalUnet::segModelInfer中提取重采样逻辑
}