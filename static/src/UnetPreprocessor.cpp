#include "UnetPreprocessor.h"
#include "UnetMain.h"
#include "UnetIO.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>
#include <map>
#include <chrono>

using namespace std;
using namespace cimg_library;

// 辅助函数：3D binary_fill_holes实现（匹配scipy.ndimage.binary_fill_holes）
static void binary_fill_holes_3d(CImg<bool>& mask) {
    // 使用flood fill从边界开始，标记所有外部背景
    // 未被标记的背景即为内部孔洞
    
    int width = mask.width();
    int height = mask.height();
    int depth = mask.depth();
    
    // 创建visited标记
    CImg<bool> visited(width, height, depth, 1, false);
    std::queue<std::tuple<int, int, int>> queue;
    
    // 从所有边界的背景点开始flood fill
    // X边界 (x=0 和 x=width-1)
    for (int y = 0; y < height; y++) {
        for (int z = 0; z < depth; z++) {
            if (!mask(0, y, z) && !visited(0, y, z)) {
                queue.push(std::make_tuple(0, y, z));
                visited(0, y, z) = true;
            }
            if (!mask(width-1, y, z) && !visited(width-1, y, z)) {
                queue.push(std::make_tuple(width-1, y, z));
                visited(width-1, y, z) = true;
            }
        }
    }
    
    // Y边界 (y=0 和 y=height-1)
    for (int x = 0; x < width; x++) {
        for (int z = 0; z < depth; z++) {
            if (!mask(x, 0, z) && !visited(x, 0, z)) {
                queue.push(std::make_tuple(x, 0, z));
                visited(x, 0, z) = true;
            }
            if (!mask(x, height-1, z) && !visited(x, height-1, z)) {
                queue.push(std::make_tuple(x, height-1, z));
                visited(x, height-1, z) = true;
            }
        }
    }
    
    // Z边界 (z=0 和 z=depth-1)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (!mask(x, y, 0) && !visited(x, y, 0)) {
                queue.push(std::make_tuple(x, y, 0));
                visited(x, y, 0) = true;
            }
            if (!mask(x, y, depth-1) && !visited(x, y, depth-1)) {
                queue.push(std::make_tuple(x, y, depth-1));
                visited(x, y, depth-1) = true;
            }
        }
    }
    
    // BFS找到所有连接到边界的背景点
    while (!queue.empty()) {
        auto [x, y, z] = queue.front();
        queue.pop();
        
        // 检查6个邻居（3D中的6连通）
        int dx[] = {-1, 1, 0, 0, 0, 0};
        int dy[] = {0, 0, -1, 1, 0, 0};
        int dz[] = {0, 0, 0, 0, -1, 1};
        
        for (int i = 0; i < 6; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];
            
            // 检查边界条件
            if (nx >= 0 && nx < width && 
                ny >= 0 && ny < height && 
                nz >= 0 && nz < depth) {
                // 如果是背景且未访问过
                if (!mask(nx, ny, nz) && !visited(nx, ny, nz)) {
                    queue.push(std::make_tuple(nx, ny, nz));
                    visited(nx, ny, nz) = true;
                }
            }
        }
    }
    
    // 填充所有内部孔洞（未被访问的背景点）
    int filled_count = 0;
    cimg_forXYZ(mask, x, y, z) {
        if (!mask(x, y, z) && !visited(x, y, z)) {
            mask(x, y, z) = true;  // 填充孔洞
            filled_count++;
        }
    }
}

// 主预处理函数 - 执行完整的预处理管道
AI_INT UnetPreprocessor::preprocessVolume(UnetMain* parent, 
                                         nnUNetConfig& config, 
                                         CImg<short>& input_volume,
                                         CImg<float>& output_volume)
{
    std::cout << "\n======= Preprocessing Stage =======" << std::endl;
    auto preprocess_start = std::chrono::steady_clock::now();

    // 步骤1：转置
    input_volume.permute_axes(config.cimg_transpose_forward);
    
    // 更新转置后的spacing
    parent->transposed_input_voxel_spacing.clear();
    parent->transposed_original_voxel_spacing.clear();
    for (int i = 0; i < 3; ++i) {
        parent->transposed_input_voxel_spacing.push_back(parent->input_voxel_spacing[config.transpose_forward[i]]);
        parent->transposed_original_voxel_spacing.push_back(parent->original_voxel_spacing[config.transpose_forward[i]]);
    }
    
    // 步骤2：裁剪到非零区域
    CImg<short> cropped_volume = cropToNonzero(input_volume, parent->crop_bbox, parent->seg_mask);
    
    // 步骤3：计算归一化参数
    // 根据归一化类型决定是否使用JSON配置的值
    if (config.normalization_type == "ZScoreNormalization") {
        // ZScoreNormalization总是动态计算mean和std（MRI等模态）
        if (config.use_mask_for_norm) {
            // 在mask区域动态计算（将在归一化步骤中进行）
            parent->intensity_mean = 0.0;  // 占位值
            parent->intensity_std = 1.0;   // 占位值
        } else {
            // 在整个裁剪后的数据上动态计算
            parent->intensity_mean = cropped_volume.mean();  // CImg::mean()返回double
            double var = cropped_volume.variance();  // CImg::variance()返回double
            parent->intensity_std = std::sqrt(var);
            if (parent->intensity_std < 1e-8) parent->intensity_std = 1e-8;  // 匹配Python的max(std, 1e-8)
        }
    } else if (config.normalization_type == "CTNormalization" || 
               config.normalization_type == "CT" || 
               config.normalization_type == "ct") {
        // CTNormalization使用JSON配置的值（CT等标准化模态）
        parent->intensity_mean = config.mean;
        parent->intensity_std = config.std;
    } else {
        // 默认行为：动态计算
        parent->intensity_mean = cropped_volume.mean();
        double var = cropped_volume.variance();
        parent->intensity_std = std::sqrt(var);
        if (parent->intensity_std < 1e-8) parent->intensity_std = 1e-8;
    }

    // 步骤4：计算输出尺寸
    bool is_volume_scaled = true;  // 始终进行缩放（与Python一致）
    std::vector<int64_t> input_size = { cropped_volume.width(), cropped_volume.height(), cropped_volume.depth() };
    std::vector<int64_t> output_size;
    
    for (int i = 0; i < 3; ++i) {
        float scaled_factor = parent->transposed_original_voxel_spacing[i] / config.voxel_spacing[i];
        int scaled_sz = std::round(input_size[i] * scaled_factor);
        
        if (scaled_sz < config.patch_size[i])
            scaled_sz = config.patch_size[i];

        output_size.push_back(static_cast<int64_t>(scaled_sz));
    }

    // 步骤5：归一化（在原始分辨率上进行）
    CImg<float> normalized_volume;
    normalized_volume.assign(cropped_volume);  // 转换为float
    
    // 保存归一化前的数据
    if (parent->saveIntermediateResults && !parent->preprocessOutputPath.empty()) {
        // 创建UnetIO需要的ImageMetadata类型
        ::ImageMetadata metadata;
        metadata.origin[0] = parent->imageMetadata.origin[0];
        metadata.origin[1] = parent->imageMetadata.origin[1];
        metadata.origin[2] = parent->imageMetadata.origin[2];
        metadata.spacing[0] = parent->imageMetadata.spacing[0];
        metadata.spacing[1] = parent->imageMetadata.spacing[1];
        metadata.spacing[2] = parent->imageMetadata.spacing[2];
        UnetIO::savePreprocessedData(normalized_volume, parent->preprocessOutputPath, L"before_normalization", metadata);
    }
    
    // 执行归一化
    std::map<std::string, int> normalizationOptionsMap = {
        {"CTNormalization",     10},
        {"CT",                  10},
        {"ct",                  10},
        {"CTNorm",              10},
        {"ctnorm",              10},
        {"ZScoreNormalization", 20},
        {"zscore",              20},
        {"z-score",             20},
    };
    auto it = normalizationOptionsMap.find(config.normalization_type);
    int normalization_type = (it != normalizationOptionsMap.end()) ? it->second : 20;

    switch (normalization_type) {
    case 10:
        CTNormalization(normalized_volume, config);
        break;
    case 20:
        ZScoreNormalization(normalized_volume, parent->seg_mask, config, 
                          parent->intensity_mean, parent->intensity_std);
        break;
    default:
        // 默认使用Z-Score归一化
        normalized_volume -= parent->intensity_mean;
        normalized_volume /= parent->intensity_std;
        break;
    }

    // 步骤6：重采样（在归一化后进行）
    if (is_volume_scaled) {
        resampleVolume(normalized_volume, output_volume, output_size);
    } else {
        output_volume.assign(normalized_volume);
    }

    // 保存预处理后的数据
    if (parent->saveIntermediateResults && !parent->preprocessOutputPath.empty()) {
        // 创建UnetIO需要的ImageMetadata类型
        ::ImageMetadata metadata;
        metadata.origin[0] = parent->imageMetadata.origin[0];
        metadata.origin[1] = parent->imageMetadata.origin[1];
        metadata.origin[2] = parent->imageMetadata.origin[2];
        metadata.spacing[0] = parent->imageMetadata.spacing[0];
        metadata.spacing[1] = parent->imageMetadata.spacing[1];
        metadata.spacing[2] = parent->imageMetadata.spacing[2];
        // 保存归一化后但重采样前的数据
        UnetIO::savePreprocessedData(normalized_volume, parent->preprocessOutputPath, 
                                    L"after_normalization_before_resample", metadata);
        // 保存最终的预处理数据
        UnetIO::savePreprocessedData(output_volume, parent->preprocessOutputPath, 
                                    L"preprocessed_normalized_volume", metadata);
    }

    auto preprocess_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> preprocess_elapsed = preprocess_end - preprocess_start;
    std::cout << "Preprocessing completed in " << preprocess_elapsed.count() << " seconds" << std::endl;
    std::cout << "  Preprocessed volume shape: " << output_volume.width() << " x " 
              << output_volume.height() << " x " << output_volume.depth() << std::endl;
    std::cout << "  Mean: " << output_volume.mean() << ", Std: " << std::sqrt(output_volume.variance()) << std::endl;
    std::cout << "======= Preprocessing Complete =======" << std::endl;

    return UnetSegAI_STATUS_SUCCESS;
}

// 裁剪到非零区域 - 与Python版本对齐
CImg<short> UnetPreprocessor::cropToNonzero(const CImg<short>& input, CropBBox& bbox, CImg<short>& seg_mask)
{
    // 找到非零区域的边界
    bbox.x_min = input.width();
    bbox.x_max = -1;
    bbox.y_min = input.height();
    bbox.y_max = -1;
    bbox.z_min = input.depth();
    bbox.z_max = -1;
    
    // 创建非零mask（与Python的nonzero_mask对应）
    CImg<bool> nonzero_mask(input.width(), input.height(), input.depth(), 1, false);
    
    // 扫描整个体积找到非零区域
    cimg_forXYZ(input, x, y, z) {
        if (input(x, y, z) != 0) {
            nonzero_mask(x, y, z) = true;
        }
    }
    
    // 应用binary_fill_holes（与Python的scipy.ndimage.binary_fill_holes一致）
    // 注意：目前暂时禁用以测试是否是fill hole导致的差异
    // binary_fill_holes_3d(nonzero_mask);
    
    // 重新计算bbox（基于填充后的mask）
    cimg_forXYZ(input, x, y, z) {
        if (nonzero_mask(x, y, z)) {
            if (x < bbox.x_min) bbox.x_min = x;
            if (x > bbox.x_max) bbox.x_max = x;
            if (y < bbox.y_min) bbox.y_min = y;
            if (y > bbox.y_max) bbox.y_max = y;
            if (z < bbox.z_min) bbox.z_min = z;
            if (z > bbox.z_max) bbox.z_max = z;
        }
    }
    
    // 如果没有找到非零像素，返回原图像
    if (bbox.x_max == -1) {
        bbox.x_min = 0; bbox.x_max = input.width() - 1;
        bbox.y_min = 0; bbox.y_max = input.height() - 1;
        bbox.z_min = 0; bbox.z_max = input.depth() - 1;
        
        // 创建全为-1的seg_mask（因为全是背景）
        seg_mask = CImg<short>(input.width(), input.height(), input.depth(), 1, -1);
        return input;
    }
    
    // 验证bbox是否合理
    if (bbox.x_min > bbox.x_max || bbox.y_min > bbox.y_max || bbox.z_min > bbox.z_max) {
        // 重置为全图像
        bbox.x_min = 0; bbox.x_max = input.width() - 1;
        bbox.y_min = 0; bbox.y_max = input.height() - 1;
        bbox.z_min = 0; bbox.z_max = input.depth() - 1;
        
        // 创建seg_mask
        seg_mask = CImg<short>(input.width(), input.height(), input.depth(), 1);
        cimg_forXYZ(seg_mask, x, y, z) {
            seg_mask(x, y, z) = (input(x, y, z) != 0) ? 0 : -1;
        }
        return input;
    }
    
    // 执行裁剪
    CImg<short> cropped = input.get_crop(bbox.x_min, bbox.y_min, bbox.z_min, 
                                         bbox.x_max, bbox.y_max, bbox.z_max);
    
    // 裁剪nonzero_mask
    CImg<bool> cropped_mask = nonzero_mask.get_crop(bbox.x_min, bbox.y_min, bbox.z_min,
                                                    bbox.x_max, bbox.y_max, bbox.z_max);
    
    // 创建seg_mask（与Python的seg对应）
    // Python: seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label=-1))
    seg_mask = CImg<short>(cropped.width(), cropped.height(), cropped.depth(), 1);
    cimg_forXYZ(cropped, x, y, z) {
        // 使用填充后的mask：mask区域设为0，背景设为-1（与Python一致）
        seg_mask(x, y, z) = cropped_mask(x, y, z) ? 0 : -1;
    }
    
    return cropped;
}

// CT归一化 - 使用配置的percentile值进行裁剪和标准化
void UnetPreprocessor::CTNormalization(CImg<float>& input_volume, const nnUNetConfig& config)
{
    // 使用percentile值进行裁剪（与Python版本一致）
    double lower_bound = config.percentile_00_5;
    double upper_bound = config.percentile_99_5;
    
    input_volume.cut(lower_bound, upper_bound);

    // 应用z-score标准化（使用double提高精度）
    double mean_hu = config.mean_std_HU[0];
    double std_hu = config.mean_std_HU[1];
    input_volume -= mean_hu;
    input_volume /= std_hu;
}

// Z-Score归一化
void UnetPreprocessor::ZScoreNormalization(CImg<float>& volume, 
                                          const CImg<short>& seg_mask,
                                          const nnUNetConfig& config,
                                          double& intensity_mean,
                                          double& intensity_std)
{
    if (config.use_mask_for_norm && !seg_mask.is_empty()) {
        // 使用seg_mask创建mask（与Python一致：seg >= 0表示非零区域）
        // Python: mask = seg[0] >= 0
        CImg<bool> mask(volume.width(), volume.height(), volume.depth());
        cimg_forXYZ(volume, x, y, z) {
            // seg_mask中：0表示非零区域，-1表示背景
            // 所以seg_mask >= 0就是非零区域
            mask(x, y, z) = (seg_mask(x, y, z) >= 0);
        }
        
        // 在mask区域动态计算mean和std（匹配Python行为）
        // 使用double提高精度，避免累积误差
        double mask_mean = 0.0;
        double mask_std = 0.0;
        int mask_count = 0;
        
        // 计算mask区域的mean
        cimg_forXYZ(volume, x, y, z) {
            if (mask(x, y, z)) {
                mask_mean += volume(x, y, z);
                mask_count++;
            }
        }
        
        if (mask_count > 0) {
            mask_mean /= mask_count;
            
            // 计算mask区域的std
            cimg_forXYZ(volume, x, y, z) {
                if (mask(x, y, z)) {
                    double diff = volume(x, y, z) - mask_mean;
                    mask_std += diff * diff;
                }
            }
            mask_std = std::sqrt(mask_std / mask_count);
            if (mask_std < 1e-8) mask_std = 1e-8;  // 匹配Python的max(std, 1e-8)
            
            // 更新返回的值
            intensity_mean = mask_mean;
            intensity_std = mask_std;
            
            // 只对mask区域进行归一化，背景设为0
            cimg_forXYZ(volume, x, y, z) {
                if (mask(x, y, z)) {
                    volume(x, y, z) = (volume(x, y, z) - mask_mean) / mask_std;
                } else {
                    volume(x, y, z) = 0.0f;
                }
            }
        }
    } else {
        // 传统的全局归一化
        // 使用double提高精度
        intensity_mean = volume.mean();
        double var = volume.variance();
        intensity_std = std::sqrt(var);
        if (intensity_std < 1e-8) intensity_std = 1e-8;
        
        volume -= intensity_mean;
        volume /= intensity_std;
    }
}

// 重采样
void UnetPreprocessor::resampleVolume(const CImg<float>& input,
                                     CImg<float>& output,
                                     const std::vector<int64_t>& output_size)
{
    if (output_size.size() != 3) {
        throw std::runtime_error("Output size must be 3D");
    }
    
    // 使用三次插值（5）而不是线性插值（3）以匹配Python的order=3
    // CImg插值模式: 0=最近邻, 1=线性, 2=移动平均, 3=线性, 5=三次(cubic)
    output = input.get_resize(output_size[0], output_size[1], output_size[2], -100, 5);
}