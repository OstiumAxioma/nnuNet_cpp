#include "UnetPostprocessor.h"
#include "UnetMain.h"
#include "UnetInference.h"
#include "UnetSegAI_API.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace cimg_library;

// 主后处理函数 - 执行完整的后处理流程
AI_INT UnetPostprocessor::processSegmentationMask(UnetMain* parent, AI_DataInfo* dstData)
{
    std::cout << "\n======= Post-processing Stage =======" << endl;
    auto postprocess_start = std::chrono::steady_clock::now();
    
    // 获取parent的成员变量
    auto& predicted_output_prob = parent->predicted_output_prob;
    auto& output_seg_mask = parent->output_seg_mask;
    auto& unetConfig = parent->unetConfig;
    auto& crop_bbox = parent->crop_bbox;
    auto& imageMetadata = parent->imageMetadata;
    bool saveIntermediateResults = parent->saveIntermediateResults;
    int Width0 = parent->Width0;
    int Height0 = parent->Height0;
    int Depth0 = parent->Depth0;
    
    // 步骤1：对概率图执行argmax（在转置后的坐标系中）
    output_seg_mask = UnetInference::argmax_spectrum(predicted_output_prob);
    
    // 保存后处理数据（在转置撤销前）
    if (saveIntermediateResults) {
        parent->savePostprocessedData(output_seg_mask, L"postprocessed_segmentation_mask_before_transpose");
    }
    
    // 步骤2：撤销转置（恢复到原始坐标系）
    revertTranspose(output_seg_mask, unetConfig.cimg_transpose_backward);
    
    // 步骤3：恢复到原始尺寸
    CImg<short> final_result;
    restoreOriginalSize(output_seg_mask, final_result, crop_bbox, Width0, Height0, Depth0);
    
    // 步骤4：复制结果到输出数据结构
    long volSize = Width0 * Height0 * Depth0 * sizeof(short);
    std::memcpy(dstData->ptr_Data, final_result.data(), volSize);
    
    // 设置输出数据的维度
    dstData->Width = Width0;
    dstData->Height = Height0;
    dstData->Depth = Depth0;
    
    // 将保存的origin信息传回给调用者
    dstData->OriginX = imageMetadata.origin[0];
    dstData->OriginY = imageMetadata.origin[1];
    dstData->OriginZ = imageMetadata.origin[2];
    
    // 同时确保spacing信息也正确传回
    dstData->VoxelSpacingX = imageMetadata.spacing[0];
    dstData->VoxelSpacingY = imageMetadata.spacing[1];
    dstData->VoxelSpacingZ = imageMetadata.spacing[2];
    
    // 保存最终的后处理结果
    if (saveIntermediateResults) {
        parent->savePostprocessedData(final_result, L"postprocessed_segmentation_mask_final");
        std::cout << "  Post-processed result saved to: result/postprocess/" << endl;
    }
    
    auto postprocess_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> postprocess_elapsed = postprocess_end - postprocess_start;
    std::cout << "Post-processing completed in " << postprocess_elapsed.count() << " seconds" << endl;
    std::cout << "======= Post-processing Complete =======" << endl;
    
    return UnetSegAI_STATUS_SUCCESS;
}

// 恢复到原始尺寸
void UnetPostprocessor::restoreOriginalSize(const CImg<short>& input,
                                           CImg<short>& output,
                                           const CropBBox& bbox,
                                           int width0, int height0, int depth0)
{
    // 检查是否需要恢复裁剪
    if (input.width() != width0 || input.height() != height0 || input.depth() != depth0) {
        
        // 创建原始尺寸的结果mask，初始化为0
        output = CImg<short>(width0, height0, depth0, 1, 0);
        
        // 先检查bbox是否已经初始化
        if (bbox.x_max == -1 || bbox.y_max == -1 || bbox.z_max == -1) {
            
            // 使用fallback逻辑 - 直接复制能复制的部分
            int copy_width = std::min(input.width(), width0);
            int copy_height = std::min(input.height(), height0);
            int copy_depth = std::min(input.depth(), depth0);
            
            for (int z = 0; z < copy_depth; z++) {
                for (int y = 0; y < copy_height; y++) {
                    for (int x = 0; x < copy_width; x++) {
                        output(x, y, z) = input(x, y, z);
                    }
                }
            }
        }
        // 将裁剪后的结果放回到原始位置
        else if (bbox.x_min >= 0 && bbox.x_max < width0 && 
                 bbox.y_min >= 0 && bbox.y_max < height0 &&
                 bbox.z_min >= 0 && bbox.z_max < depth0) {
            
            // 将input的内容复制到output的对应位置
            cimg_forXYZ(input, x, y, z) {
                int orig_x = x + bbox.x_min;
                int orig_y = y + bbox.y_min;
                int orig_z = z + bbox.z_min;
                if (orig_x < width0 && orig_y < height0 && orig_z < depth0) {
                    output(orig_x, orig_y, orig_z) = input(x, y, z);
                }
            }
        } else {
            
            // 如果bbox无效，尝试直接复制能复制的部分
            int copy_width = std::min(input.width(), width0);
            int copy_height = std::min(input.height(), height0);
            int copy_depth = std::min(input.depth(), depth0);
            
            for (int z = 0; z < copy_depth; z++) {
                for (int y = 0; y < copy_height; y++) {
                    for (int x = 0; x < copy_width; x++) {
                        output(x, y, z) = input(x, y, z);
                    }
                }
            }
        }
    } else {
        // 如果尺寸匹配，直接复制
        output = input;
    }
}

// 撤销转置操作
void UnetPostprocessor::revertTranspose(CImg<short>& volume, const char* transpose_backward)
{
    volume.permute_axes(transpose_backward);
}