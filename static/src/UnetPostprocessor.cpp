#include "UnetPostprocessor.h"
#include "UnetIO.h"
#include "UnetSegAI_API.h"
#include <iostream>
#include <cstring>

using namespace std;
using namespace cimg_library;

// 主后处理函数
AI_INT UnetPostprocessor::processSegmentationMask(UnetMain* parent,
                                                 CImg<float>& prob_volume,
                                                 AI_DataInfo* dstData)
{
    std::cout << "\n======= Post-processing Stage =======" << endl;
    auto postprocess_start = std::chrono::steady_clock::now();
    
    // 步骤1：对概率图执行argmax（在转置后的坐标系中）
    CImg<short> output_seg_mask = argmaxSpectrum(prob_volume);
    
    // 保存后处理数据（在转置撤销前）
    if (parent && parent->saveIntermediateResults && !parent->postprocessOutputPath.empty()) {
        // 创建UnetIO需要的ImageMetadata类型
        ::ImageMetadata metadata;
        metadata.origin[0] = parent->imageMetadata.origin[0];
        metadata.origin[1] = parent->imageMetadata.origin[1];
        metadata.origin[2] = parent->imageMetadata.origin[2];
        metadata.spacing[0] = parent->imageMetadata.spacing[0];
        metadata.spacing[1] = parent->imageMetadata.spacing[1];
        metadata.spacing[2] = parent->imageMetadata.spacing[2];
        UnetIO::savePostprocessedData(output_seg_mask, parent->postprocessOutputPath, 
                                      L"postprocessed_segmentation_mask_before_transpose", 
                                      metadata);
    }
    
    // 步骤2：撤销转置（恢复到原始坐标系）
    revertTranspose(output_seg_mask, parent->unetConfig.cimg_transpose_backward);
    
    // 步骤3：检查是否需要恢复裁剪
    if (output_seg_mask.width() != parent->Width0 || 
        output_seg_mask.height() != parent->Height0 || 
        output_seg_mask.depth() != parent->Depth0) {
        
        CImg<short> full_result;
        restoreOriginalSize(output_seg_mask, full_result, parent->crop_bbox, 
                          parent->Width0, parent->Height0, parent->Depth0);
        
        // 复制恢复后的结果
        long volSize = parent->Width0 * parent->Height0 * parent->Depth0 * sizeof(short);
        std::memcpy(dstData->ptr_Data, full_result.data(), volSize);
        
    } else {
        // 如果尺寸匹配，直接复制
        long volSize = parent->Width0 * parent->Height0 * parent->Depth0 * sizeof(short);
        std::memcpy(dstData->ptr_Data, output_seg_mask.data(), volSize);
    }
    
    // 将保存的origin信息传回给调用者
    dstData->OriginX = parent->imageMetadata.origin[0];
    dstData->OriginY = parent->imageMetadata.origin[1];
    dstData->OriginZ = parent->imageMetadata.origin[2];
    
    // 同时确保spacing信息也正确传回
    dstData->VoxelSpacingX = parent->imageMetadata.spacing[0];
    dstData->VoxelSpacingY = parent->imageMetadata.spacing[1];
    dstData->VoxelSpacingZ = parent->imageMetadata.spacing[2];
    
    // 保存最终的后处理结果
    if (parent && parent->saveIntermediateResults && !parent->postprocessOutputPath.empty()) {
        // 需要从dstData创建一个CImg对象来保存
        CImg<short> final_result(dstData->Width, dstData->Height, dstData->Depth, 1);
        std::memcpy(final_result.data(), dstData->ptr_Data, 
                   dstData->Width * dstData->Height * dstData->Depth * sizeof(short));
        // 创建UnetIO需要的ImageMetadata类型
        ::ImageMetadata metadata;
        metadata.origin[0] = parent->imageMetadata.origin[0];
        metadata.origin[1] = parent->imageMetadata.origin[1];
        metadata.origin[2] = parent->imageMetadata.origin[2];
        metadata.spacing[0] = parent->imageMetadata.spacing[0];
        metadata.spacing[1] = parent->imageMetadata.spacing[1];
        metadata.spacing[2] = parent->imageMetadata.spacing[2];
        UnetIO::savePostprocessedData(final_result, parent->postprocessOutputPath, 
                                      L"postprocessed_segmentation_mask_final", 
                                      metadata);
        std::cout << "  Post-processed result saved to: result/postprocess/" << endl;
    }

    auto postprocess_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> postprocess_elapsed = postprocess_end - postprocess_start;
    std::cout << "Post-processing completed in " << postprocess_elapsed.count() << " seconds" << endl;
    std::cout << "======= Post-processing Complete =======" << endl;
    
    return UnetSegAI_STATUS_SUCCESS;
}

// Argmax操作
CImg<short> UnetPostprocessor::argmaxSpectrum(const CImg<float>& input)
{
    if (input.is_empty() || input.spectrum() == 0) {
        throw std::invalid_argument("Input must be a non-empty 4D CImg with spectrum dimension.");
    }

    // 创建结果图像，大小与输入相同，但spectrum维度为1
    CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

    // 遍历每个体素 (x,y,z)
    cimg_forXYZ(input, x, y, z) {
        short max_idx = 0;
        float max_val = input(x, y, z, 0);

        // 遍历spectrum维度
        for (short s = 1; s < input.spectrum(); ++s) {
            const float current_val = input(x, y, z, s);
            if (current_val > max_val) {
                max_val = current_val;
                max_idx = s;
            }
        }
        result(x, y, z) = max_idx; // 存储最大值的索引
    }
    return result;
}

// 恢复到原始尺寸
void UnetPostprocessor::restoreOriginalSize(const CImg<short>& input,
                                           CImg<short>& output,
                                           const CropBBox& bbox,
                                           int width0, int height0, int depth0)
{
    // 创建原始尺寸的结果mask，初始化为0
    output = CImg<short>(width0, height0, depth0, 1, 0);
    
    // 先检查bbox是否已经初始化
    if (bbox.x_max == -1 || bbox.y_max == -1 || bbox.z_max == -1) {
        // 使用fallback逻辑
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
}

// 撤销转置操作
void UnetPostprocessor::revertTranspose(CImg<short>& volume,
                                       const char* transpose_backward)
{
    volume.permute_axes(transpose_backward);
}