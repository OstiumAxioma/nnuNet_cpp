#include "UnetTorchInference.h"
#include "UnetMain.h"
#include <iostream>
#include <cmath>
#include <chrono>

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

    // Create 1D Gaussian kernels for each dimension
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto x = torch::linspace(-(window_sizes[0] - 1) / 2.0, (window_sizes[0] - 1) / 2.0, window_sizes[0], options);
    auto y = torch::linspace(-(window_sizes[1] - 1) / 2.0, (window_sizes[1] - 1) / 2.0, window_sizes[1], options);
    auto z = torch::linspace(-(window_sizes[2] - 1) / 2.0, (window_sizes[2] - 1) / 2.0, window_sizes[2], options);

    // Calculate Gaussian values
    auto gauss_x = torch::exp(-0.5 * x.pow(2) / pow(sigmas[0], 2));
    auto gauss_y = torch::exp(-0.5 * y.pow(2) / pow(sigmas[1], 2));
    auto gauss_z = torch::exp(-0.5 * z.pow(2) / pow(sigmas[2], 2));

    // Normalize each dimension
    gauss_x /= gauss_x.sum();
    gauss_y /= gauss_y.sum();
    gauss_z /= gauss_z.sum();

    // Create 3D kernel by outer product
    auto kernel = gauss_x.unsqueeze(-1).unsqueeze(-1)
        * gauss_y.unsqueeze(0).unsqueeze(-1)
        * gauss_z.unsqueeze(0).unsqueeze(0);

    // Normalize by mean to maintain intensity scale
    return kernel / kernel.mean();
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

    // Set device
    torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (use_gpu ? "CUDA" : "CPU") << std::endl;
    
    // Move model to device
    model.to(device);
    model.eval();

    // Get volume dimensions
    int width = preprocessed_volume.width();
    int height = preprocessed_volume.height();
    int depth = preprocessed_volume.depth();
    
    std::cout << "Volume shape: " << width << " x " << height << " x " << depth << std::endl;
    std::cout << "Patch size: " << config.patch_size[0] << " x " << config.patch_size[1] << " x " << config.patch_size[2] << std::endl;

    // Create Gaussian kernel for weighted averaging
    torch::Tensor gaussian_kernel = create3DGaussianKernel(
        {config.patch_size[2], config.patch_size[1], config.patch_size[0]}  // depth, height, width
    );
    gaussian_kernel = gaussian_kernel.to(device);
    
    // Calculate step sizes and number of tiles
    float step_size_ratio = config.step_size_ratio;
    std::vector<float> actual_step_size(3);
    std::vector<int> num_steps(3);
    
    // Calculate for each dimension
    for (int dim = 0; dim < 3; ++dim) {
        int vol_size, patch_size;
        if (dim == 0) { vol_size = width; patch_size = config.patch_size[0]; }
        else if (dim == 1) { vol_size = height; patch_size = config.patch_size[1]; }
        else { vol_size = depth; patch_size = config.patch_size[2]; }
        
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
    CImg<float> input_patch(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.0f);
    CImg<float> weight_patch(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.0f);
    CImg<float> output_patch(config.patch_size[0], config.patch_size[1], config.patch_size[2], config.num_classes, 0.0f);
    
    // Copy Gaussian weights to CImg
    torch::Tensor gaussian_cpu = gaussian_kernel.to(torch::kCPU);
    std::memcpy(weight_patch.data(), gaussian_cpu.data_ptr<float>(), 
                config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float));
    
    // Sliding window loop
    int patch_count = 0;
    for (int sx = 0; sx < num_steps[0]; sx++) {
        int lb_x = (int)std::round(sx * actual_step_size[0]);
        int ub_x = lb_x + config.patch_size[0] - 1;
        
        for (int sy = 0; sy < num_steps[1]; sy++) {
            int lb_y = (int)std::round(sy * actual_step_size[1]);
            int ub_y = lb_y + config.patch_size[1] - 1;
            
            for (int sz = 0; sz < num_steps[2]; sz++) {
                int lb_z = (int)std::round(sz * actual_step_size[2]);
                int ub_z = lb_z + config.patch_size[2] - 1;
                
                patch_count++;
                if (patch_count % 10 == 0 || patch_count == total_patches) {
                    std::cout << "Processing patch " << patch_count << "/" << total_patches << std::endl;
                }
                
                // Extract patch
                input_patch = preprocessed_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
                
                // Convert to tensor with shape [1, 1, depth, height, width]
                torch::Tensor input_tensor = cimgToTensor(
                    input_patch,
                    {1, 1, config.patch_size[2], config.patch_size[1], config.patch_size[0]},
                    device
                );
                
                // Run inference
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                torch::Tensor output_tensor = model.forward(inputs).toTensor();
                
                // Apply Gaussian weighting
                output_tensor = output_tensor.squeeze(0);  // Remove batch dimension
                output_tensor = output_tensor * gaussian_kernel;
                
                // Copy back to CPU and then to CImg
                output_tensor = output_tensor.to(torch::kCPU);
                long patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
                std::memcpy(output_patch.data(), output_tensor.data_ptr<float>(), patch_vol_sz);
                
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
    std::cout << "TorchScript inference completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "======= TorchScript Inference Complete =======" << std::endl;
    
    return UnetSegAI_STATUS_SUCCESS;
}