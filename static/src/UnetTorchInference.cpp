#include "UnetTorchInference.h"
#include "UnetMain.h"
#include "UnetIO.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdlib>  // For std::getenv
#include "../include/SystemMonitor.h"

// Try to include CUDA headers for version info
#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include <cudnn.h>
#endif

using namespace std;
using namespace cimg_library;

// Create 3D Gaussian kernel for weighted averaging (based on DentalUnet_cimg_version.cpp line 225-254)
torch::Tensor UnetTorchInference::create3DGaussianKernel(const std::vector<int64_t>& window_sizes)
{
    if (window_sizes.size() != 3) {
        throw std::runtime_error("Window sizes must have 3 dimensions");
    }

    std::vector<float> sigmas(3);
    for (int i = 0; i < 3; ++i) {
        sigmas[i] = (window_sizes[i] - 1) / 6.0f;  // Standard deviation calculation
    }
    std::cout << "    Sigmas calculated: [" << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2] << "]" << std::endl;

    // Create 1D Gaussian kernels for each dimension
    std::cout << "    Creating torch tensor options..." << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    std::cout << "    Creating linspace for x dimension..." << std::endl;
    auto x = torch::linspace(-(window_sizes[0] - 1) / 2.0, (window_sizes[0] - 1) / 2.0, window_sizes[0], options);
    std::cout << "    Creating linspace for y dimension..." << std::endl;
    auto y = torch::linspace(-(window_sizes[1] - 1) / 2.0, (window_sizes[1] - 1) / 2.0, window_sizes[1], options);
    std::cout << "    Creating linspace for z dimension..." << std::endl;
    auto z = torch::linspace(-(window_sizes[2] - 1) / 2.0, (window_sizes[2] - 1) / 2.0, window_sizes[2], options);

    // Calculate Gaussian values
    std::cout << "    Calculating Gaussian values..." << std::endl;
    auto gauss_x = torch::exp(-0.5 * x.pow(2) / pow(sigmas[0], 2));
    auto gauss_y = torch::exp(-0.5 * y.pow(2) / pow(sigmas[1], 2));
    auto gauss_z = torch::exp(-0.5 * z.pow(2) / pow(sigmas[2], 2));

    // Normalize each dimension
    std::cout << "    Normalizing dimensions..." << std::endl;
    gauss_x /= gauss_x.sum();
    gauss_y /= gauss_y.sum();
    gauss_z /= gauss_z.sum();

    // Create 3D kernel by outer product
    std::cout << "    Creating 3D kernel by outer product..." << std::endl;
    auto kernel = gauss_x.unsqueeze(-1).unsqueeze(-1)
        * gauss_y.unsqueeze(0).unsqueeze(-1)
        * gauss_z.unsqueeze(0).unsqueeze(0);

    // Normalize by mean to maintain intensity scale
    auto result = kernel / kernel.mean();
    return result;
}

// Convert CImg to Torch Tensor
torch::Tensor UnetTorchInference::cimgToTensor(
    const CImg<float>& img,
    const std::vector<int64_t>& shape,
    torch::Device device)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    
    // Create tensor from CImg data (zero-copy when possible)
    torch::Tensor tensor = torch::from_blob(
        const_cast<float*>(img.data()),
        shape,
        options
    );
    
    // Move to device if needed
    if (device.type() == torch::kCUDA) {
        tensor = tensor.to(device);
    }
    
    return tensor;
}

// Copy Tensor data back to CImg
void UnetTorchInference::tensorToCImg(
    const torch::Tensor& tensor,
    CImg<float>& img)
{
    // Ensure tensor is on CPU
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU);
    
    // Copy data
    size_t num_elements = img.size();
    std::memcpy(img.data(), cpu_tensor.data_ptr<float>(), num_elements * sizeof(float));
}

// Main sliding window inference function
AI_INT UnetTorchInference::runSlidingWindowTorch(
    UnetMain* parent,
    const nnUNetConfig& config,
    const CImg<float>& preprocessed_volume,
    CImg<float>& predicted_output_prob,
    torch::jit::script::Module& model,
    bool use_gpu)
{
    std::cout << "\n======= TorchScript Sliding Window Inference =======" << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    // Check CUDA availability
    bool cuda_available = false;
    int device_count = 0;
    
    try {
        cuda_available = torch::cuda::is_available();
        if (cuda_available) {
            device_count = torch::cuda::device_count();
            std::cout << "CUDA available: " << device_count << " device(s) found" << std::endl;
        } else {
            std::cout << "CUDA not available, using CPU" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error checking CUDA: " << e.what() << std::endl;
        cuda_available = false;
    }
    
    bool actually_use_gpu = use_gpu && cuda_available && (device_count > 0);
    
    if (use_gpu && !actually_use_gpu) {
        std::cout << "WARNING: GPU requested but not available. Falling back to CPU." << std::endl;
    }
    
    torch::Device device(actually_use_gpu ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (actually_use_gpu ? "CUDA" : "CPU") << std::endl;
    
    // Move model to device with error handling
    try {
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error moving model to device: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU" << std::endl;
        device = torch::Device(torch::kCPU);
        model.to(device);
        model.eval();
        actually_use_gpu = false;
    }

    // Get volume dimensions
    int width = preprocessed_volume.width();
    int height = preprocessed_volume.height();
    int depth = preprocessed_volume.depth();
    
    std::cout << "Volume shape (W x H x D): " << width << " x " << height << " x " << depth << std::endl;
    std::cout << "Patch size from JSON (D x H x W): " << config.patch_size[0] << " x " << config.patch_size[1] << " x " << config.patch_size[2] << std::endl;
    std::cout << "Actual patch extraction (W x H x D): " << config.patch_size[2] << " x " << config.patch_size[1] << " x " << config.patch_size[0] << std::endl;

    // Create Gaussian kernel for weighted averaging
    torch::Tensor gaussian_kernel;
    try {
        gaussian_kernel = create3DGaussianKernel(
            {config.patch_size[0], config.patch_size[1], config.patch_size[2]}  // 与模型期望一致: [128, 160, 112]
        );
        gaussian_kernel = gaussian_kernel.to(device);
        std::cout << "Gaussian kernel created and moved to " << (actually_use_gpu ? "CUDA" : "CPU") << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error creating Gaussian kernel: " << e.what() << std::endl;
        std::cerr << "Error details: " << e.msg() << std::endl;
        return UnetSegAI_STATUS_FAIED;
    } catch (const std::exception& e) {
        std::cerr << "Standard error creating Gaussian kernel: " << e.what() << std::endl;
        return UnetSegAI_STATUS_FAIED;
    }
    
    // Calculate step sizes and number of tiles
    float step_size_ratio = config.step_size_ratio;
    std::vector<float> actual_step_size(3);
    std::vector<int> num_steps(3);
    
    // Calculate for each dimension
    // CImg维度: (width, height, depth) = (180, 216, 181)
    // JSON patch_size: [depth, height, width] = [128, 160, 112]
    // 所以对应关系是：
    for (int dim = 0; dim < 3; ++dim) {
        int vol_size, patch_size;
        if (dim == 0) { vol_size = width; patch_size = config.patch_size[2]; }  // CImg width 对应 patch width (112)
        else if (dim == 1) { vol_size = height; patch_size = config.patch_size[1]; }  // CImg height 对应 patch height (160)
        else { vol_size = depth; patch_size = config.patch_size[0]; }  // CImg depth 对应 patch depth (128)
        
        num_steps[dim] = (int)std::ceil(float(vol_size - patch_size) / (patch_size * step_size_ratio)) + 1;
        if (num_steps[dim] > 1) {
            actual_step_size[dim] = float(vol_size - patch_size) / (num_steps[dim] - 1);
        } else {
            actual_step_size[dim] = 999999.0f;
            num_steps[dim] = 1;
        }
    }
    
    int total_patches = num_steps[0] * num_steps[1] * num_steps[2];
    std::cout << "Number of tiles: " << total_patches 
              << " (" << num_steps[0] << " x " << num_steps[1] << " x " << num_steps[2] << ")" << std::endl;

    // Initialize output volume and count map
    predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.0f);
    CImg<float> count_vol(width, height, depth, 1, 0.0f);
    
    // Prepare patch buffers
    // CImg patch应该对应实际的空间尺寸
    // CImg格式: (width, height, depth) = (112, 160, 128)
    CImg<float> input_patch(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.0f);
    CImg<float> weight_patch(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.0f);
    CImg<float> output_patch(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.0f);
    
    // Copy Gaussian weights to CImg
    torch::Tensor gaussian_cpu = gaussian_kernel.to(torch::kCPU);
    // 直接拷贝，因为两者都是连续内存
    std::memcpy(weight_patch.data(), gaussian_cpu.data_ptr<float>(), 
                config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float));
    
    // Sliding window loop
    int patch_count = 0;
    for (int sx = 0; sx < num_steps[0]; sx++) {
        int lb_x = (int)std::round(sx * actual_step_size[0]);
        int ub_x = lb_x + config.patch_size[2] - 1;  // width对应patch_size[2] (112)
        
        for (int sy = 0; sy < num_steps[1]; sy++) {
            int lb_y = (int)std::round(sy * actual_step_size[1]);
            int ub_y = lb_y + config.patch_size[1] - 1;  // height对应patch_size[1] (160)
            
            for (int sz = 0; sz < num_steps[2]; sz++) {
                int lb_z = (int)std::round(sz * actual_step_size[2]);
                int ub_z = lb_z + config.patch_size[0] - 1;  // depth对应patch_size[0] (128)
                
                patch_count++;
                // Show progress for every patch to debug tile placement
                std::cout << "Processing patch " << patch_count << "/" << total_patches 
                          << " at position [" << lb_x << "-" << ub_x << ", " 
                          << lb_y << "-" << ub_y << ", " 
                          << lb_z << "-" << ub_z << "]" << std::endl;
                
                // Extract patch
                input_patch = preprocessed_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
                
                // Convert to tensor with shape [1, 1, depth, height, width]
                // 模型期望: [1, 1, 128, 160, 112]
                // 直接使用JSON中的顺序
                torch::Tensor input_tensor = cimgToTensor(
                    input_patch,
                    {1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2]},
                    device
                );
                
                // Run inference with error handling
                torch::Tensor output_tensor;
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    
                    // 获取推理前的GPU内存状态
                    SystemMonitor::GPUInfo gpu_before = SystemMonitor::getGPUInfo();
                    
                    // 记录tile推理开始时间
                    auto tile_start = std::chrono::steady_clock::now();
                    
                    // Run model inference
                    torch::jit::IValue output = model.forward(inputs);
                    
                    // 记录tile推理结束时间
                    auto tile_end = std::chrono::steady_clock::now();
                    std::chrono::duration<double> tile_elapsed = tile_end - tile_start;
                    
                    // 获取推理后的GPU内存状态
                    SystemMonitor::GPUInfo gpu_after = SystemMonitor::getGPUInfo();
                    
                    // 输出tile性能信息
                    std::cout << "  Tile inference time: " << std::fixed << std::setprecision(3) 
                              << tile_elapsed.count() << "s" << std::endl;
                    if (gpu_after.available) {
                        std::cout << "  GPU memory: " << SystemMonitor::formatBytes(gpu_after.usedMemory) 
                                  << " / " << SystemMonitor::formatBytes(gpu_after.totalMemory)
                                  << " (" << std::fixed << std::setprecision(1) 
                                  << gpu_after.memoryUsagePercent << "%)" << std::endl;
                    }
                    
                    // Handle model output - could be a tensor or a list
                    if (output.isTensor()) {
                        // Direct tensor output
                        output_tensor = output.toTensor();
                    } else if (output.isList()) {
                        // Model returns a list (typical for UNet with multiple outputs)
                        auto output_list = output.toList();
                        if (output_list.size() > 0) {
                            // Get the first element (usually the final segmentation output)
                            output_tensor = output_list.get(0).toTensor();
                            if (patch_count == 1) {
                                std::cout << "Model returns a list with " << output_list.size() << " outputs, using the first one" << std::endl;
                            }
                        } else {
                            std::cerr << "Model returned an empty list!" << std::endl;
                            return UnetSegAI_STATUS_FAIED;
                        }
                    } else {
                        std::cerr << "Unexpected model output type (not tensor or list)" << std::endl;
                        return UnetSegAI_STATUS_FAIED;
                    }
                } catch (const c10::Error& e) {
                    std::cerr << "Error during inference at patch " << patch_count << ": " << e.what() << std::endl;
                    return UnetSegAI_STATUS_FAIED;
                } catch (const std::exception& e) {
                    std::cerr << "Unexpected error during inference: " << e.what() << std::endl;
                    return UnetSegAI_STATUS_FAIED;
                }
                
                // Apply Gaussian weighting
                output_tensor = output_tensor.squeeze(0);  // Remove batch dimension [C, D, H, W]
                
                // Apply Gaussian kernel (element-wise multiplication for each channel)
                for (int c = 0; c < config.num_classes; c++) {
                    output_tensor[c] = output_tensor[c] * gaussian_kernel;
                }
                
                // Convert output to CPU
                output_tensor = output_tensor.to(torch::kCPU);
                
                // 参考代码直接memcpy，说明输出的内存布局与CImg匹配
                // 输出tensor: [num_classes, depth, height, width]
                // CImg buffer: [width, height, depth, num_classes]
                long patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
                std::memcpy(output_patch.data(), output_tensor.data_ptr<float>(), patch_vol_sz);

                // 保存单个tile（如果启用了中间结果保存）
                if (parent && parent->saveIntermediateResults && !parent->modelOutputPath.empty()) {
                    UnetIO::saveTile(output_patch, patch_count, lb_x, lb_y, lb_z, parent->modelOutputPath);
                }
                
                // Accumulate weighted predictions
                cimg_forXYZC(output_patch, x, y, z, c) {
                    predicted_output_prob(lb_x + x, lb_y + y, lb_z + z, c) += output_patch(x, y, z, c);
                }
                
                // Accumulate weights
                cimg_forXYZ(weight_patch, x, y, z) {
                    count_vol(lb_x + x, lb_y + y, lb_z + z) += weight_patch(x, y, z);
                }
            }
        }
    }
    
    // Normalize by accumulated weights
    std::cout << "Normalizing predictions..." << std::endl;
    cimg_forXYZC(predicted_output_prob, x, y, z, c) {
        float weight = count_vol(x, y, z);
        if (weight > 1e-8f) {
            predicted_output_prob(x, y, z, c) /= weight;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // 使用统一的输出格式和资源监控
    SystemMonitor::printTimingAndResources("TorchScript Inference", elapsed.count());
    
    return UnetSegAI_STATUS_SUCCESS;
}