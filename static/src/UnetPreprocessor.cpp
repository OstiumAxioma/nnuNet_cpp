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
#include "../include/SystemMonitor.h"

using namespace std;
using namespace cimg_library;


// 主预处理函数 - 执行完整的预处理管道
AI_INT UnetPreprocessor::preprocessVolume(UnetMain* parent, 
                                         nnUNetConfig& config, 
                                         CImg<short>& input_volume,
                                         CImg<float>& output_volume)
{
    std::cout << "\n======= Preprocessing Stage =======" << std::endl;
    auto preprocess_start = std::chrono::steady_clock::now();

    // 验证transpose_forward索引的有效性
    for (int i = 0; i < 3; ++i) {
        if (config.transpose_forward[i] < 0 || config.transpose_forward[i] >= 3) {
            std::cerr << "Error: Invalid transpose_forward index: " << config.transpose_forward[i] << std::endl;
            return UnetSegAI_STATUS_FAIED;
        }
    }
    
    // 验证spacing向量的大小
    if (parent->input_voxel_spacing.size() != 3 || parent->original_voxel_spacing.size() != 3) {
        std::cerr << "Error: Spacing vectors not properly initialized" << std::endl;
        std::cerr << "  input_voxel_spacing size: " << parent->input_voxel_spacing.size() << std::endl;
        std::cerr << "  original_voxel_spacing size: " << parent->original_voxel_spacing.size() << std::endl;
        return UnetSegAI_STATUS_FAIED;
    }

    // 步骤1：转置
    input_volume.permute_axes(config.cimg_transpose_forward);
    
    // 更新转置后的spacing（使用临时变量避免部分更新）
    std::vector<float> temp_transposed_input(3);
    std::vector<float> temp_transposed_original(3);
    
    for (int i = 0; i < 3; ++i) {
        int idx = config.transpose_forward[i];
        temp_transposed_input[i] = parent->input_voxel_spacing[idx];
        temp_transposed_original[i] = parent->original_voxel_spacing[idx];
    }
    
    // 原子性更新
    parent->transposed_input_voxel_spacing = temp_transposed_input;
    parent->transposed_original_voxel_spacing = temp_transposed_original;
    
    // 步骤2：裁剪到非零区域
    CImg<short> cropped_volume = cropToNonzero(input_volume, parent->crop_bbox, parent->seg_mask);
    
    // 步骤3：计算归一化参数
    int num_channels = cropped_volume.spectrum();
    parent->intensity_means.assign(num_channels, 0.0);
    parent->intensity_stds.assign(num_channels, 1.0);

    for (int c = 0; c < num_channels; ++c) {
        CImg<short> channel_view = cropped_volume.get_shared_channel(c);
        const std::string& norm_scheme = config.normalization_schemes[c];
        if (norm_scheme == "ZScoreNormalization") {
            if (config.use_mask_for_norm[c]) {
                // 在mask区域动态计算，将在归一化步骤中进行
                // 暂时使用占位值
                parent->intensity_means[c] = 0.0;
                parent->intensity_stds[c] = 1.0;
            } else {
                // 在整个裁剪后的数据上动态计算
                parent->intensity_means[c] = channel_view.mean();
                double var = channel_view.variance();
                parent->intensity_stds[c] = std::sqrt(var);
                if (parent->intensity_stds[c] < 1e-8) parent->intensity_stds[c] = 1e-8;
            }
        } else if (norm_scheme == "CTNormalization" || norm_scheme == "CT" || norm_scheme == "ct") {
            // CTNormalization使用JSON配置的值
            parent->intensity_means[c] = config.means[c];
            parent->intensity_stds[c] = config.stds[c];
        } else {
            // 其他或未指定类型的默认行为：动态计算
            parent->intensity_means[c] = channel_view.mean();
            double var = channel_view.variance();
            parent->intensity_stds[c] = std::sqrt(var);
            if (parent->intensity_stds[c] < 1e-8) parent->intensity_stds[c] = 1e-8;
        }
    }

    // 步骤4：计算输出尺寸
    bool is_volume_scaled = true;  // 始终进行缩放（与Python一致）
    std::vector<int64_t> input_size = { cropped_volume.width(), cropped_volume.height(), cropped_volume.depth() };
    std::vector<int64_t> output_size;

    // ================= START OF MODIFIED SECTION =================
    // 参考Python代码逻辑，增加对2D情况的处理
    // 在nnU-Net中，2D模型的plans.json文件中 "voxel_spacing" 键对应的值会是一个二维数组
    bool is_2d = config.voxel_spacing.size() == 2;

    if (is_2d) {
        // 对于2D情况，我们只对空间维度（通常是X和Y）进行重采样。
        // 第三个维度（通常是Z，即切片维度）保持其原始大小，不进行缩放。
        // 假设转置后的维度顺序是 X, Y, Z

        // 缩放 X 和 Y 维度
        for (int i = 0; i < 2; ++i) {
            float scaled_factor = parent->transposed_original_voxel_spacing[i] / config.voxel_spacing[i];
            int scaled_sz = std::round(input_size[i] * scaled_factor);

            //if (scaled_sz < config.patch_size[i]) {
                //scaled_sz = config.patch_size[i];
            //}
            output_size.push_back(static_cast<int64_t>(scaled_sz));
        }

        // Z 维度的尺寸保持不变
        output_size.push_back(input_size[2]);

    } else {
        // 原始的3D情况处理逻辑：对所有三个维度进行重采样
        for (int i = 0; i < 3; ++i) {
            float scaled_factor = parent->transposed_original_voxel_spacing[i] / config.voxel_spacing[i];
            int scaled_sz = std::round(input_size[i] * scaled_factor);

            //if (scaled_sz < config.patch_size[i])
                //scaled_sz = config.patch_size[i];

            output_size.push_back(static_cast<int64_t>(scaled_sz));
        }
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

    // 执行归一化（逐通道）
    for (int c = 0; c < num_channels; ++c) {
        CImg<float> channel_view = normalized_volume.get_shared_channel(c);
        const std::string& norm_scheme = config.normalization_schemes[c];

        if (norm_scheme == "CTNormalization" || norm_scheme == "CT" || norm_scheme == "ct") {
            CTNormalization(channel_view, config, c);
        } else if (norm_scheme == "ZScoreNormalization") {
            // ZScoreNormalization现在直接修改传入的channel_view
            ZScoreNormalization(channel_view, parent->seg_mask, config, c, parent->intensity_means[c], parent->intensity_stds[c]);
        } else {
            // 默认使用Z-Score归一化（使用已计算好的参数）
            channel_view -= parent->intensity_means[c];
            channel_view /= parent->intensity_stds[c];
        }
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
    
    // 使用统一的输出格式和资源监控
    SystemMonitor::printTimingAndResources("Preprocessing", preprocess_elapsed.count());
    std::cout << "  Preprocessed volume shape: " << output_volume.width() << " x " 
              << output_volume.height() << " x " << output_volume.depth() << std::endl;
    std::cout << "  Mean: " << output_volume.mean() << ", Std: " << std::sqrt(output_volume.variance()) << std::endl;

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
        bool is_voxel_nonzero = false;
        for (int c = 0; c < input.spectrum(); ++c) {
            if (input(x, y, z, c) != 0) {
                is_voxel_nonzero = true;
                break;
            }
        }
        if (is_voxel_nonzero) {
            nonzero_mask(x, y, z) = true;
        }
    }
    // 应用binary_fill_holes（与Python的scipy.ndimage.binary_fill_holes一致）
    // 注意：目前暂时禁用以测试是否是fill hole导致的差异
    // binaryFillHoles3d(nonzero_mask);

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

// CT归一化 - 修改为处理单个通道，并使用channel_index获取参数
void UnetPreprocessor:: CTNormalization(CImg<float>& volume_channel, const nnUNetConfig& config, int channel_index)
{
    // 使用对应通道的percentile值进行裁剪
    double lower_bound = config.percentile_00_5s[channel_index];
    double upper_bound = config.percentile_99_5s[channel_index];
    
    volume_channel.cut(lower_bound, upper_bound);

    // 应用对应通道的z-score标准化
    double mean_hu = config.means[channel_index];
    double std_hu = config.stds[channel_index];
    if (std_hu < 1e-8) std_hu = 1e-8;
    
    volume_channel -= mean_hu;
    volume_channel /= std_hu;
}

// 辅助函数：3D binary_fill_holes实现（匹配scipy.ndimage.binary_fill_holes）
static void binaryFillHoles3d(CImg<bool>& mask) {
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

// Z-Score归一化
// Z-Score归一化 - 修改为处理单个通道
void UnetPreprocessor::ZScoreNormalization(CImg<float>& volume_channel, 
                         const CImg<short>& seg_mask,
                         const nnUNetConfig& config,
                         int channel_index,
                         double& intensity_mean,
                         double& intensity_std)
{
    if (config.use_mask_for_norm[channel_index] && !seg_mask.is_empty()) {
        CImg<bool> mask(volume_channel.width(), volume_channel.height(), volume_channel.depth());
        cimg_forXYZ(volume_channel, x, y, z) {
            mask(x, y, z) = (seg_mask(x, y, z) >= 0);
        }
        
        double mask_mean = 0.0;
        double mask_std_dev = 0.0;
        long long mask_count = 0;
        
        cimg_forXYZ(volume_channel, x, y, z) {
            if (mask(x, y, z)) {
                mask_mean += volume_channel(x, y, z);
                mask_count++;
            }
        }
        
        if (mask_count > 0) {
            mask_mean /= mask_count;
            
            cimg_forXYZ(volume_channel, x, y, z) {
                if (mask(x, y, z)) {
                    double diff = volume_channel(x, y, z) - mask_mean;
                    mask_std_dev += diff * diff;
                }
            }
            mask_std_dev = std::sqrt(mask_std_dev / mask_count);
            if (mask_std_dev < 1e-8) mask_std_dev = 1e-8;
            
            // 更新由外部传入的引用值
            intensity_mean = mask_mean;
            intensity_std = mask_std_dev;
            
            cimg_forXYZ(volume_channel, x, y, z) {
                if (mask(x, y, z)) {
                    volume_channel(x, y, z) = (volume_channel(x, y, z) - mask_mean) / mask_std_dev;
                } else {
                    volume_channel(x, y, z) = 0.0f;
                }
            }
        }
    } else {
        // 传统的全局归一化（但现在只作用于单个通道）
        // 使用已经预先计算好的mean和std
        volume_channel -= intensity_mean;
        volume_channel /= intensity_std;
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