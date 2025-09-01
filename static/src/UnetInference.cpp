#include "UnetInference.h"
#include "UnetMain.h"
#include "UnetPreprocessor.h"
#include "UnetSegAI_API.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstring>

using namespace std;
using namespace cimg_library;

// 主推理函数 - 执行完整的模型推理流程
AI_INT UnetInference::segModelInfer(UnetMain* parent, nnUNetConfig config, CImg<short> input_volume)
{
    // 预处理阶段现在由UnetPreprocessor处理
    CImg<float> preprocessed_volume;
    AI_INT preprocess_status = UnetPreprocessor::preprocessVolume(parent, config, input_volume, preprocessed_volume);
    if (preprocess_status != UnetSegAI_STATUS_SUCCESS) {
        return preprocess_status;
    }

    //调用滑窗推理函数
    std::cout << "\n======= Sliding Window Inference =======" << endl;
    auto inference_start = std::chrono::steady_clock::now();
    try {
        AI_INT is_ok = slidingWindowInfer(parent, config, preprocessed_volume);
        if (is_ok != UnetSegAI_STATUS_SUCCESS) {
            return is_ok;
        }
    } catch (const std::exception& e) {
        return UnetSegAI_STATUS_FAIED;
    } catch (...) {
        return UnetSegAI_STATUS_FAIED;
    }

    auto inference_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> inference_elapsed = inference_end - inference_start;
    std::cout << "Inference completed in " << inference_elapsed.count() << " seconds" << endl;
    std::cout << "======= Inference Complete =======" << endl;

    //如果进行了3D重采样，调整大小
    bool is_volume_scaled = (preprocessed_volume.width() != input_volume.width() || 
                            preprocessed_volume.height() != input_volume.height() || 
                            preprocessed_volume.depth() != input_volume.depth());
    if (is_volume_scaled)
        parent->predicted_output_prob.resize(input_volume.width(), input_volume.height(), input_volume.depth(), config.num_classes, 3);

    // 保存模型输出（概率体）
    if (parent->saveIntermediateResults) {
        parent->saveModelOutput(parent->predicted_output_prob, L"model_output_probability");
        std::cout << "  Model output saved to: result/model_output/" << endl;
    }

    // 生成分割掩码（argmax操作）
    parent->output_seg_mask = UnetInference::argmax_spectrum(parent->predicted_output_prob);
    
    // 保存argmax后的mask
    if (parent->saveIntermediateResults) {
        parent->savePostprocessedData(parent->output_seg_mask, L"after_argmax");
    }
    
    return UnetSegAI_STATUS_SUCCESS;
}

// 滑动窗口推理实现
AI_INT UnetInference::slidingWindowInfer(UnetMain* parent, nnUNetConfig config, CImg<float> normalized_volume)
{
    // 访问parent的成员变量
    auto& env = parent->env;
    auto& session_options = parent->session_options;
    auto& predicted_output_prob = parent->predicted_output_prob;
    bool use_gpu = parent->use_gpu;
    bool saveIntermediateResults = parent->saveIntermediateResults;
    
    // GPU设置
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options{};
        try {
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const Ort::Exception& e) {
        }
    }

    // 创建会话
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 检查模型文件名
    if (config.model_file_name == nullptr) {
        return UnetSegAI_LOADING_FAIED;
    }
    
    //try-catch处理ONNX Runtime异常
    try {
        Ort::Session session(env, config.model_file_name, session_options);
        
        // 使用AllocatedStringPtr来管理内存
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        if (input_shape.size() != 5) {
            throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
        }

        // 验证
        if (config.patch_size.size() != 3) {
            throw std::runtime_error("Patch size should be 3D (depth, height, width)");
        }

        // ONNX张量形状: (batch, channel, depth, height, width)
        std::vector<int64_t> input_tensor_shape = { 1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2] };

        int depth = normalized_volume.depth();
        int width = normalized_volume.width();
        int height = normalized_volume.height();
        
        // Padding步骤（匹配Python的pad_nd_image）
        // 计算需要的padding以确保尺寸可被patch_size整除
        int padded_depth = depth;
        int padded_width = width;
        int padded_height = height;
        
        // 如果尺寸小于patch_size，需要padding到至少patch_size
        if (padded_depth < config.patch_size[0]) {
            padded_depth = config.patch_size[0];
        }
        if (padded_height < config.patch_size[1]) {
            padded_height = config.patch_size[1];
        }
        if (padded_width < config.patch_size[2]) {
            padded_width = config.patch_size[2];
        }
        
        // 计算padding量（居中padding，如果需要奇数padding，则"上"侧多padding 1）
        int pad_depth_before = (padded_depth - depth) / 2;
        int pad_depth_after = padded_depth - depth - pad_depth_before;
        int pad_width_before = (padded_width - width) / 2;
        int pad_width_after = padded_width - width - pad_width_before;
        int pad_height_before = (padded_height - height) / 2;
        int pad_height_after = padded_height - height - pad_height_before;
        
        // 创建padded volume
        CImg<float> padded_volume(padded_width, padded_height, padded_depth, 1, 0.0f);
        
        // 复制原始数据到padded volume的中心
        if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
            cimg_forXYZ(normalized_volume, x, y, z) {
                padded_volume(x + pad_width_before, y + pad_height_before, z + pad_depth_before) = 
                    normalized_volume(x, y, z);
            }
        } else {
            // 如果没有padding，直接使用原始volume
            padded_volume = normalized_volume;
        }
        
        // 使用padded dimensions进行后续计算
        int working_depth = padded_depth;
        int working_width = padded_width;
        int working_height = padded_height;

        // x图像宽度, y图像高度, z图像深度
        float step_size_ratio = config.step_size_ratio;
        
        // 匹配Python的compute_steps_for_sliding_window逻辑
        // 首先计算目标步长
        std::vector<int> steps;
        for (int i = 0; i < 3; i++) {
            if (config.patch_size[i] > 1) {
                steps.push_back(std::max(1, static_cast<int>(config.patch_size[i] * step_size_ratio)));
            } else {
                steps.push_back(1);
            }
        }

        int step_depth = steps[0];
        int step_height = steps[1];
        int step_width = steps[2];

        // 计算滑动窗口的数量
        auto compute_num_blocks = [](int image_size, int patch_size, int step_size) -> int {
            if (patch_size == image_size) {
                return 1;
            } else if (patch_size < image_size) {
                return (int)std::ceil((float)(image_size - patch_size) / step_size) + 1;
            } else {
                return 1;
            }
        };

        int num_blocks_depth = compute_num_blocks(working_depth, config.patch_size[0], step_depth);
        int num_blocks_height = compute_num_blocks(working_height, config.patch_size[1], step_height);
        int num_blocks_width = compute_num_blocks(working_width, config.patch_size[2], step_width);

        // 输出概率体积
        CImg<float> padded_output_prob(working_width, working_height, working_depth, config.num_classes, 0.f);
        CImg<float> count_vol(working_width, working_height, working_depth, 1, 0.f);

        // 创建高斯核（用于重叠区域的权重）
        CImg<float> gaussisan_weight(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
        UnetInference::createGaussianKernel(gaussisan_weight, config.patch_size);

        int patch_count = 0;
        int total_patches = num_blocks_depth * num_blocks_height * num_blocks_width;
        std::cout << "Total patches to process: " << total_patches << endl;

        // 滑动窗口
        for (int d = 0; d < num_blocks_depth; d++) {
            for (int h = 0; h < num_blocks_height; h++) {
                for (int w = 0; w < num_blocks_width; w++) {
                    patch_count++;
                    std::cout << "Processing tile #" << patch_count << "/" << total_patches << "..." << endl;
                    
                    // 计算patch的左上角坐标
                    int lb_z = d * step_depth;
                    int lb_y = h * step_height;
                    int lb_x = w * step_width;
                    
                    // 确保不超出边界
                    if (lb_z + config.patch_size[0] > working_depth)
                        lb_z = working_depth - config.patch_size[0];
                    if (lb_y + config.patch_size[1] > working_height)
                        lb_y = working_height - config.patch_size[1];
                    if (lb_x + config.patch_size[2] > working_width)
                        lb_x = working_width - config.patch_size[2];

                    // 提取patch
                    CImg<float> input_patch = padded_volume.get_crop(
                        lb_x, lb_y, lb_z,
                        lb_x + config.patch_size[2] - 1,
                        lb_y + config.patch_size[1] - 1,
                        lb_z + config.patch_size[0] - 1
                    );

                    // 准备输入张量
                    size_t input_patch_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
                    size_t output_patch_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * config.num_classes * sizeof(float);
                    CImg<float> win_pob(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.f);

                    // 执行推理
                    try {
                        // 创建ONNX Runtime张量
                        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
                        
                        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                            memory_info,
                            input_patch.data(),
                            input_patch_vol_sz / sizeof(float),
                            input_tensor_shape.data(),
                            input_tensor_shape.size()
                        );

                        // 运行推理
                        auto output_tensors = session.Run(
                            Ort::RunOptions{nullptr},
                            &input_name,
                            &input_tensor,
                            1,
                            &output_name,
                            1
                        );

                        // 处理输出张量
                        if (output_tensors.empty()) {
                            return UnetSegAI_STATUS_FAIED;
                        }
                        
                        float* output_data = output_tensors[0].GetTensorMutableData<float>();
                        
                        if (output_data == nullptr) {
                            return UnetSegAI_STATUS_FAIED;
                        }

                        // 复制到CImg
                        std::memcpy(win_pob.data(), output_data, output_patch_vol_sz);
                        output_tensors.clear();

                        // 保存单个tile
                        if (saveIntermediateResults) {
                            parent->saveTile(win_pob, patch_count, lb_x, lb_y, lb_z);
                        }

                        // 累加到输出体积（使用高斯权重）
                        cimg_forXYZC(win_pob, x, y, z, c) {
                            int gx = lb_x + x;
                            int gy = lb_y + y;
                            int gz = lb_z + z;
                            
                            // 写入前验证边界（使用padded dimensions）
                            if (gx < 0 || gx >= working_width || gy < 0 || gy >= working_height || gz < 0 || gz >= working_depth) {
                                return UnetSegAI_STATUS_FAIED;
                            }
                            
                            padded_output_prob(gx, gy, gz, c) += (win_pob(x, y, z, c) * gaussisan_weight(x, y, z));
                        }
                        cimg_forXYZ(gaussisan_weight, x, y, z) {
                            count_vol(lb_x + x, lb_y + y, lb_z + z) += gaussisan_weight(x, y, z);
                        }
                    } catch (const std::exception& e) {
                        return UnetSegAI_STATUS_FAIED;
                    }
                    
                    std::cout << "Tile #" << patch_count << " completed" << endl;
                }
            }
        }

        //归一化
        cimg_forXYZC(padded_output_prob, x, y, z, c) {
            padded_output_prob(x, y, z, c) /= count_vol(x, y, z);
        }
        
        // 从padded结果中提取原始尺寸的输出（移除padding）
        predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.f);
        if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
            cimg_forXYZC(predicted_output_prob, x, y, z, c) {
                predicted_output_prob(x, y, z, c) = padded_output_prob(x + pad_width_before, 
                                                                        y + pad_height_before, 
                                                                        z + pad_depth_before, c);
            }
        } else {
            predicted_output_prob = padded_output_prob;
        }
        
        std::cout << "Sliding window inference is done." << endl;

        return UnetSegAI_STATUS_SUCCESS;
    } catch (const Ort::Exception& e) {
        return UnetSegAI_LOADING_FAIED;
    } catch (const std::exception& e) {
        return UnetSegAI_STATUS_FAIED;
    }
}

// 创建3D高斯核
void UnetInference::createGaussianKernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes)
{
    // 匹配Python版本：sigma_scale = 1/8
    float sigma_scale = 1.0f / 8.0f;
    float value_scaling_factor = 10.0f;

    int64_t depth  = patch_sizes[0];
    int64_t height = patch_sizes[1]; 
    int64_t width  = patch_sizes[2];

    // 计算中心点坐标
    float z_center = (depth - 1)  / 2.0f;
    float y_center = (height - 1) / 2.0f;
    float x_center = (width - 1)  / 2.0f;
    
    // 计算各维度的sigma值
    float z_sigma = depth * sigma_scale;
    float y_sigma = height * sigma_scale;
    float x_sigma = width * sigma_scale;

    // 使用CImg生成3D高斯权重
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        float dz = (z - z_center) / z_sigma;
        float dy = (y - y_center) / y_sigma;
        float dx = (x - x_center) / x_sigma;
        
        float distance_squared = dx * dx + dy * dy + dz * dz;
        float gaussian_value = std::exp(-0.5f * distance_squared) * value_scaling_factor;
        
        gaussisan_weight(x, y, z) = gaussian_value;
    }

    // 找到最小非零值
    float min_non_zero = std::numeric_limits<float>::max();
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        if (gaussisan_weight(x, y, z) > 0 && gaussisan_weight(x, y, z) < min_non_zero) {
            min_non_zero = gaussisan_weight(x, y, z);
        }
    }

    // 如果没有找到非零值，设置一个很小的默认值
    if (min_non_zero == std::numeric_limits<float>::max()) {
        min_non_zero = 1e-6f;
    }

    // 将所有零值替换为最小非零值
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        if (gaussisan_weight(x, y, z) == 0) {
            gaussisan_weight(x, y, z) = min_non_zero;
        }
    }
}

// Argmax操作 - 将概率转换为类别标签
CImg<short> UnetInference::argmax_spectrum(const CImg<float>& input) {
    if (input.is_empty() || input.spectrum() == 0) {
        throw std::invalid_argument("Input must be a non-empty 4D CImg with spectrum dimension.");
    }

    // 创建结果图像，大小与输入相同，但spectrum维度为1
    CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

    // 对每个空间位置，找出最大概率的类别
    cimg_forXYZ(input, x, y, z) {
        float max_value = input(x, y, z, 0);
        short max_channel = 0;
        
        // 遍历所有通道（类别）
        for (int c = 1; c < input.spectrum(); c++) {
            if (input(x, y, z, c) > max_value) {
                max_value = input(x, y, z, c);
                max_channel = c;
            }
        }
        
        result(x, y, z) = max_channel;
    }

    return result;
}