#include "UnetInference.h"
#include "UnetMain.h"
#include "UnetSegAI_API.h"
#include <iostream>
#include <cmath>
#include <limits>

using namespace std;
using namespace cimg_library;

// 主推理函数 - 滑窗推理
AI_INT UnetInference::runSlidingWindow(UnetMain* parent,
                                      const nnUNetConfig& config,
                                      const CImg<float>& input,
                                      CImg<float>& output,
                                      Ort::Env& env,
                                      Ort::SessionOptions& session_options,
                                      bool use_gpu)
{
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const Ort::Exception& e) {
            // GPU不可用时继续使用CPU
        }
    }

    // 创建会话
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 检查模型文件名
    if (config.model_file_name == nullptr) {
        return UnetSegAI_LOADING_FAIED;
    }
    
    try {
        Ort::Session session(env, config.model_file_name, session_options);
        
        // 获取输入输出名称
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        if (input_shape.size() != 5) {
            throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
        }

        // 验证patch_size
        if (config.patch_size.size() != 3) {
            throw std::runtime_error("Patch size should be 3D (depth, height, width)");
        }

        // ONNX张量形状: (batch, channel, depth, height, width)
        std::vector<int64_t> input_tensor_shape = { 1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2] };

        int depth = input.depth();
        int width = input.width();
        int height = input.height();
        
        // Padding步骤
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
        
        // 计算padding量
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
            cimg_forXYZ(input, x, y, z) {
                padded_volume(x + pad_width_before, y + pad_height_before, z + pad_depth_before) = 
                    input(x, y, z);
            }
        } else {
            padded_volume = input;
        }
        
        // 使用padded dimensions进行后续计算
        int working_depth = padded_depth;
        int working_width = padded_width;
        int working_height = padded_height;

        float step_size_ratio = config.step_size_ratio;
        
        // 计算目标步长
        float target_step_x = config.patch_size[2] * step_size_ratio;
        float target_step_y = config.patch_size[1] * step_size_ratio;
        float target_step_z = config.patch_size[0] * step_size_ratio;
        
        // 计算步数
        int X_num_steps = std::max(1, (int)ceil(float(working_width - config.patch_size[2]) / target_step_x) + 1);
        int Y_num_steps = std::max(1, (int)ceil(float(working_height - config.patch_size[1]) / target_step_y) + 1);
        int Z_num_steps = std::max(1, (int)ceil(float(working_depth - config.patch_size[0]) / target_step_z) + 1);
        
        // 计算实际步长
        float actualStepSize[3];
        if (X_num_steps > 1) {
            actualStepSize[0] = float(working_width - config.patch_size[2]) / (X_num_steps - 1);
        } else {
            actualStepSize[0] = 0;
        }
        
        if (Y_num_steps > 1) {
            actualStepSize[1] = float(working_height - config.patch_size[1]) / (Y_num_steps - 1);
        } else {
            actualStepSize[1] = 0;
        }
        
        if (Z_num_steps > 1) {
            actualStepSize[2] = float(working_depth - config.patch_size[0]) / (Z_num_steps - 1);
        } else {
            actualStepSize[2] = 0;
        }

        // 初始化输出概率体
        CImg<float> padded_output_prob = CImg<float>(working_width, working_height, working_depth, config.num_classes, 0.f);
        CImg<float> count_vol = CImg<float>(working_width, working_height, working_depth, 1, 0.f);
        
        CImg<float> win_pob = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.f);
        CImg<float> gaussisan_weight = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
        createGaussianKernel(gaussisan_weight, config.patch_size);

        size_t input_patch_voxel_numel = config.patch_size[0] * config.patch_size[1] * config.patch_size[2];
        size_t output_patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

        // 输出tile总体信息
        int total_tiles = X_num_steps * Y_num_steps * Z_num_steps;
        std::cout << "Total tiles to process: " << total_tiles << endl;
        std::cout << "Tile grid: " << X_num_steps << " x " << Y_num_steps << " x " << Z_num_steps << " (X x Y x Z)" << endl;
        std::cout << "Patch size: " << config.patch_size[2] << " x " << config.patch_size[1] << " x " << config.patch_size[0] << " (W x H x D)" << endl;
        
        // 处理每个patch
        int patch_count = 0;
        for (int sz = 0; sz < Z_num_steps; sz++) {
            int lb_z = (int)std::round(sz * actualStepSize[2]);
            if (lb_z + config.patch_size[0] > working_depth) {
                lb_z = working_depth - config.patch_size[0];
            }
            lb_z = std::max(0, lb_z);
            int ub_z = lb_z + config.patch_size[0] - 1;

            for (int sy = 0; sy < Y_num_steps; sy++) {
                int lb_y = (int)std::round(sy * actualStepSize[1]);
                if (lb_y + config.patch_size[1] > working_height) {
                    lb_y = working_height - config.patch_size[1];
                }
                lb_y = std::max(0, lb_y);
                int ub_y = lb_y + config.patch_size[1] - 1;

                for (int sx = 0; sx < X_num_steps; sx++) {
                    int lb_x = (int)std::round(sx * actualStepSize[0]);
                    if (lb_x + config.patch_size[2] > working_width) {
                        lb_x = working_width - config.patch_size[2];
                    }
                    lb_x = std::max(0, lb_x);
                    int ub_x = lb_x + config.patch_size[2] - 1;

                    patch_count += 1;
                    
                    // 输出当前tile信息
                    std::cout << "\nProcessing tile #" << patch_count << "/" << total_tiles << "..." << endl;
                    std::cout << "  Position: [" << lb_x << "-" << ub_x << ", " 
                              << lb_y << "-" << ub_y << ", " 
                              << lb_z << "-" << ub_z << "]" << endl;

                    // 提取patch
                    CImg<float> input_patch;
                    try {
                        input_patch = padded_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
                        if (input_patch.width() != config.patch_size[2] || 
                            input_patch.height() != config.patch_size[1] || 
                            input_patch.depth() != config.patch_size[0]) {
                            return UnetSegAI_STATUS_FAIED;
                        }
                    } catch (const CImgException& e) {
                        return UnetSegAI_STATUS_FAIED;
                    }

                    // 执行单个patch推理
                    AI_INT status = inferPatch(session, input_patch, win_pob, 
                                              input_tensor_shape, input_name, output_name);
                    if (status != UnetSegAI_STATUS_SUCCESS) {
                        return status;
                    }

                    // 保存单个tile（如果启用了中间结果保存）
                    if (parent && parent->saveIntermediateResults) {
                        parent->saveTile(win_pob, patch_count, lb_x, lb_y, lb_z);
                    }

                    // 累加结果到输出概率体
                    try {
                        cimg_forXYZC(win_pob, x, y, z, c) {
                            int gx = lb_x + x;
                            int gy = lb_y + y;
                            int gz = lb_z + z;
                            
                            if (gx < 0 || gx >= working_width || 
                                gy < 0 || gy >= working_height || 
                                gz < 0 || gz >= working_depth) {
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

        // 归一化
        cimg_forXYZC(padded_output_prob, x, y, z, c) {
            padded_output_prob(x, y, z, c) /= count_vol(x, y, z);
        }
        
        // 从padded结果中提取原始尺寸的输出
        output = CImg<float>(width, height, depth, config.num_classes, 0.f);
        if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
            cimg_forXYZC(output, x, y, z, c) {
                output(x, y, z, c) = padded_output_prob(x + pad_width_before, 
                                                        y + pad_height_before, 
                                                        z + pad_depth_before, c);
            }
        } else {
            output = padded_output_prob;
        }
        
        std::cout << "Sliding window inference is done." << endl;
        return UnetSegAI_STATUS_SUCCESS;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << endl;
        return UnetSegAI_LOADING_FAIED;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << endl;
        return UnetSegAI_STATUS_FAIED;
    }
}

// 创建3D高斯核
void UnetInference::createGaussianKernel(CImg<float>& gaussisan_weight, 
                                        const std::vector<int64_t>& patch_sizes)
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

    // 使用与Python相同的sigma计算方法
    float z_sigma = depth  * sigma_scale;
    float y_sigma = height * sigma_scale;
    float x_sigma = width  * sigma_scale;
    
    float z_part = 0.f;
    float y_part = 0.f;
    float x_part = 0.f;
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        z_part = std::exp(-0.5f * std::pow((z - z_center) / z_sigma, 2));
        y_part = std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2));
        x_part = std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2));
        gaussisan_weight(x, y, z) = z_part * y_part * x_part;
    }

    // 匹配Python的归一化方法：除以max再乘以value_scaling_factor
    float max_val = gaussisan_weight.max();
    if (max_val > 0) {
        gaussisan_weight *= (value_scaling_factor / max_val);
    }
    
    // 处理0值（匹配Python：将0值设置为最小非零值）
    float min_non_zero = std::numeric_limits<float>::max();
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        if (gaussisan_weight(x, y, z) > 0 && gaussisan_weight(x, y, z) < min_non_zero) {
            min_non_zero = gaussisan_weight(x, y, z);
        }
    }
    cimg_forXYZ(gaussisan_weight, x, y, z) {
        if (gaussisan_weight(x, y, z) == 0) {
            gaussisan_weight(x, y, z) = min_non_zero;
        }
    }
}

// 执行单个patch的推理
AI_INT UnetInference::inferPatch(Ort::Session& session,
                                const CImg<float>& patch,
                                CImg<float>& output,
                                const std::vector<int64_t>& input_shape,
                                const char* input_name,
                                const char* output_name)
{
    try {
        // 获取输入数据指针
        float* input_data_ptr = const_cast<float*>(patch.data());
        size_t input_patch_voxel_numel = input_shape[2] * input_shape[3] * input_shape[4];
        
        // 验证输入数据
        if (input_data_ptr == nullptr) {
            return UnetSegAI_STATUS_FAIED;
        }

        // 创建ONNX内存信息和输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
            input_data_ptr,
            input_patch_voxel_numel,
            input_shape.data(),
            input_shape.size());

        // 执行推理
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
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

        // 计算输出大小
        size_t output_patch_vol_sz = output.width() * output.height() * output.depth() * output.spectrum() * sizeof(float);
        
        // 复制到输出CImg
        std::memcpy(output.data(), output_data, output_patch_vol_sz);
        
        return UnetSegAI_STATUS_SUCCESS;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error in inferPatch: " << e.what() << endl;
        return UnetSegAI_STATUS_FAIED;
    } catch (const std::exception& e) {
        std::cerr << "Error in inferPatch: " << e.what() << endl;
        return UnetSegAI_STATUS_FAIED;
    } catch (...) {
        std::cerr << "Unknown error in inferPatch" << endl;
        return UnetSegAI_STATUS_FAIED;
    }
}