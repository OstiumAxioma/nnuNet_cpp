#include "UnetTorchInference.h"
#include "UnetMain.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstdlib>  // For std::getenv

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
    std::cout << "    Entering create3DGaussianKernel" << std::endl;
    if (window_sizes.size() != 3) {
        throw std::runtime_error("Window sizes must have 3 dimensions");
    }
    std::cout << "    Window sizes: [" << window_sizes[0] << ", " << window_sizes[1] << ", " << window_sizes[2] << "]" << std::endl;

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
    std::cout << "    Normalizing kernel by mean..." << std::endl;
    auto result = kernel / kernel.mean();
    std::cout << "    Kernel creation completed successfully" << std::endl;
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

    // Check CUDA availability and set device with detailed diagnostics
    std::cout << "Checking CUDA availability..." << std::endl;
    
    // Show LibTorch version
    std::cout << "  LibTorch version: " << TORCH_VERSION << std::endl;
    
    // Try to show more build information
    std::cout << "  LibTorch build configuration:" << std::endl;
    #ifdef _GLIBCXX_USE_CXX11_ABI
        std::cout << "    CXX11 ABI: " << _GLIBCXX_USE_CXX11_ABI << std::endl;
    #endif
    #ifdef TORCH_VERSION_MAJOR
        std::cout << "    Version Major: " << TORCH_VERSION_MAJOR << std::endl;
    #endif
    #ifdef TORCH_VERSION_MINOR
        std::cout << "    Version Minor: " << TORCH_VERSION_MINOR << std::endl;
    #endif
    #ifdef TORCH_VERSION_PATCH
        std::cout << "    Version Patch: " << TORCH_VERSION_PATCH << std::endl;
    #endif
    
    // Get CUDA runtime version using CUDA API
    #ifdef USE_CUDA
        int runtimeVersion = 0;
        cudaError_t result = cudaRuntimeGetVersion(&runtimeVersion);
        if (result == cudaSuccess) {
            std::cout << "  CUDA Runtime version: " << (runtimeVersion / 1000) << "." << ((runtimeVersion % 1000) / 10) << std::endl;
        } else {
            std::cout << "  CUDA Runtime version: Unable to get (error: " << cudaGetErrorString(result) << ")" << std::endl;
        }
        
        // Get CUDA driver version
        int driverVersion = 0;
        result = cudaDriverGetVersion(&driverVersion);
        if (result == cudaSuccess) {
            std::cout << "  CUDA Driver version: " << (driverVersion / 1000) << "." << ((driverVersion % 1000) / 10) << std::endl;
        }
        
        // Get cuDNN version
        std::cout << "  cuDNN version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
    #else
        std::cout << "  CUDA support not compiled in (USE_CUDA not defined)" << std::endl;
    #endif
    
    // Show CUDA compilation version from LibTorch (if available)
    #ifdef CUDA_VERSION
        std::cout << "  LibTorch compiled with CUDA: " << (CUDA_VERSION / 1000) << "." << ((CUDA_VERSION % 1000) / 10) << std::endl;
    #endif
    
    // Runtime check for CUDA
    bool cuda_available = false;
    int device_count = 0;
    
    // Add environment variable check for CUDA
    std::cout << "\n  Checking environment variables:" << std::endl;
    
    #ifdef _WIN32
        // Use safer Windows version
        char* cuda_path_buf = nullptr;
        size_t cuda_path_size = 0;
        if (_dupenv_s(&cuda_path_buf, &cuda_path_size, "CUDA_PATH") == 0 && cuda_path_buf != nullptr) {
            std::cout << "    CUDA_PATH: " << cuda_path_buf << std::endl;
            free(cuda_path_buf);
        } else {
            std::cout << "    CUDA_PATH: Not set" << std::endl;
        }
        
        char* path_buf = nullptr;
        size_t path_size = 0;
        if (_dupenv_s(&path_buf, &path_size, "PATH") == 0 && path_buf != nullptr) {
            std::string path_str(path_buf);
            if (path_str.find("CUDA") != std::string::npos) {
                std::cout << "    PATH contains CUDA directories: YES" << std::endl;
            } else {
                std::cout << "    PATH contains CUDA directories: NO" << std::endl;
            }
            free(path_buf);
        }
    #else
        const char* cuda_path = std::getenv("CUDA_PATH");
        if (cuda_path) {
            std::cout << "    CUDA_PATH: " << cuda_path << std::endl;
        } else {
            std::cout << "    CUDA_PATH: Not set" << std::endl;
        }
        
        const char* path_env = std::getenv("PATH");
        if (path_env) {
            std::string path_str(path_env);
            if (path_str.find("CUDA") != std::string::npos) {
                std::cout << "    PATH contains CUDA directories: YES" << std::endl;
            } else {
                std::cout << "    PATH contains CUDA directories: NO" << std::endl;
            }
        }
    #endif
    
    // Try to manually initialize CUDA before checking availability
    #ifdef USE_CUDA
        std::cout << "\n  USE_CUDA is defined - attempting manual CUDA initialization..." << std::endl;
        
        // Try to manually check CUDA device
        int cuda_device_count = 0;
        cudaError_t cuda_result = cudaGetDeviceCount(&cuda_device_count);
        if (cuda_result == cudaSuccess) {
            std::cout << "    cudaGetDeviceCount: " << cuda_device_count << " devices found" << std::endl;
            
            if (cuda_device_count > 0) {
                // Get first device properties
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "    Device 0: " << prop.name << std::endl;
                std::cout << "    Compute capability: " << prop.major << "." << prop.minor << std::endl;
                std::cout << "    Total memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
            }
        } else {
            std::cout << "    cudaGetDeviceCount failed: " << cudaGetErrorString(cuda_result) << std::endl;
            std::cout << "    This usually means CUDA driver is not installed or GPU is not available" << std::endl;
        }
    #else
        std::cout << "\n  WARNING: USE_CUDA is NOT defined during compilation!" << std::endl;
        std::cout << "  This means the code was compiled without CUDA support." << std::endl;
        std::cout << "  Please rebuild with -DUSE_CUDA=ON in CMake." << std::endl;
    #endif
    
    try {
        // First check if LibTorch was built with CUDA support
        std::cout << "\n  Checking if LibTorch was built with CUDA support:" << std::endl;
        
        // Check if LibTorch headers indicate CUDA support
        // Note: Different LibTorch versions use different macros
        bool libtorch_has_cuda = false;
        
        // Check various macros that indicate CUDA support
        #ifdef USE_CUDA
            std::cout << "    USE_CUDA macro is defined in LibTorch headers" << std::endl;
            libtorch_has_cuda = true;
        #endif
        
        #ifdef AT_CUDA_ENABLED
            std::cout << "    AT_CUDA_ENABLED macro is defined" << std::endl;
            libtorch_has_cuda = true;
        #endif
        
        #ifdef TORCH_CUDA_VERSION
            std::cout << "    TORCH_CUDA_VERSION macro is defined" << std::endl;
            libtorch_has_cuda = true;
        #endif
        
        if (!libtorch_has_cuda) {
            std::cout << "    WARNING: No CUDA-related macros found in LibTorch headers" << std::endl;
            std::cout << "    This might be a header configuration issue, checking runtime..." << std::endl;
        }
        
        // Now check LibTorch's CUDA availability
        std::cout << "\n  Checking LibTorch CUDA runtime availability:" << std::endl;
        
        // First try to check if CUDA module is compiled in
        try {
            // This will throw if CUDA is not compiled in
            int dummy_count = torch::cuda::device_count();
            std::cout << "    torch::cuda::device_count() call succeeded (CUDA module present)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "    torch::cuda::device_count() failed: " << e.what() << std::endl;
            std::cout << "    This indicates LibTorch was compiled without CUDA support" << std::endl;
        }
        
        cuda_available = torch::cuda::is_available();
        std::cout << "    torch::cuda::is_available(): " << (cuda_available ? "true" : "false") << std::endl;
        
        // Try to get CUDA runtime version
        if (cuda_available) {
            // Get CUDA version that LibTorch was compiled with
            std::cout << "  CUDA version for LibTorch: ";
            #ifdef TORCH_CUDA_ARCH_LIST
                std::cout << TORCH_CUDA_ARCH_LIST << std::endl;
            #else
                std::cout << "Not available" << std::endl;
            #endif
            
            // Try to get cuDNN status
            std::cout << "  cuDNN available: " << (torch::cuda::cudnn_is_available() ? "YES" : "NO") << std::endl;
            
            // Try to get more version info
            try {
                std::cout << "  Attempting to verify CUDA functionality..." << std::endl;
                
                // Get device count
                int dev_count = torch::cuda::device_count();
                std::cout << "  Number of CUDA devices: " << dev_count << std::endl;
                
                if (dev_count > 0) {
                    // Try to create a test tensor to verify CUDA works
                    std::cout << "  Testing CUDA tensor creation..." << std::endl;
                    auto test_tensor = torch::ones({1}, torch::TensorOptions().device(torch::kCUDA, 0));
                    std::cout << "  CUDA tensor creation: SUCCESS" << std::endl;
                    
                    // Get device info from tensor
                    std::cout << "  Tensor device: " << test_tensor.device() << std::endl;
                    std::cout << "  Tensor is CUDA: " << test_tensor.is_cuda() << std::endl;
                }
            } catch (const c10::Error& e) {
                std::cout << "  Error during CUDA testing: " << e.what() << std::endl;
            }
        }
        
        if (!cuda_available) {
            // Try to understand why CUDA is not available
            std::cout << "  Attempting to diagnose CUDA availability issue..." << std::endl;
            
            // Check if we can query device count even if not available
            try {
                device_count = torch::cuda::device_count();
                std::cout << "  torch::cuda::device_count(): " << device_count << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  Error getting device count: " << e.what() << std::endl;
            }
            
            // Try to force CUDA initialization
            std::cout << "  Attempting to create a CUDA tensor as a test..." << std::endl;
            try {
                auto test = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA, 0));
                std::cout << "  SUCCESS: Created CUDA tensor!" << std::endl;
                cuda_available = true;
                device_count = 1;
            } catch (const c10::Error& e) {
                std::cout << "  FAILED to create CUDA tensor: " << e.what() << std::endl;
                std::cout << "  Error message: " << e.msg() << std::endl;
            }
        } else {
            device_count = torch::cuda::device_count();
            std::cout << "  Number of CUDA devices: " << device_count << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  Exception checking CUDA: " << e.what() << std::endl;
    }
    
    // Continue with detailed diagnostics if CUDA still not available
    if (!cuda_available) {
        std::cout << "\n  Possible reasons for no CUDA support:" << std::endl;
        std::cout << "    1. CUDA version mismatch (System: 12.9, LibTorch may be built for different version)" << std::endl;
        std::cout << "    2. Missing CUDNN libraries" << std::endl;
        std::cout << "    3. Missing CUDA runtime libraries" << std::endl;
        std::cout << "    4. LibTorch was built for a different CUDA version" << std::endl;
        std::cout << "\n  Your system has CUDA 12.9, but LibTorch 2.3.1 typically supports:" << std::endl;
        std::cout << "    - CUDA 11.8 (cu118)" << std::endl;
        std::cout << "    - CUDA 12.1 (cu121)" << std::endl;
        std::cout << "  You may need LibTorch built for CUDA 12.1 to work with CUDA 12.9" << std::endl;
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
    
    std::cout << "Volume shape: " << width << " x " << height << " x " << depth << std::endl;
    std::cout << "Patch size: " << config.patch_size[0] << " x " << config.patch_size[1] << " x " << config.patch_size[2] << std::endl;

    // Create Gaussian kernel for weighted averaging
    std::cout << "Creating Gaussian kernel..." << std::endl;
    torch::Tensor gaussian_kernel;
    try {
        std::cout << "  Kernel dimensions: [" << config.patch_size[2] << ", " << config.patch_size[1] << ", " << config.patch_size[0] << "]" << std::endl;
        gaussian_kernel = create3DGaussianKernel(
            {config.patch_size[2], config.patch_size[1], config.patch_size[0]}  // depth, height, width
        );
        std::cout << "  Gaussian kernel created successfully" << std::endl;
        std::cout << "  Moving kernel to device: " << (actually_use_gpu ? "CUDA" : "CPU") << std::endl;
        gaussian_kernel = gaussian_kernel.to(device);
        std::cout << "  Kernel moved to device successfully" << std::endl;
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
                
                // Run inference with error handling
                torch::Tensor output_tensor;
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    output_tensor = model.forward(inputs).toTensor();
                } catch (const c10::Error& e) {
                    std::cerr << "Error during inference at patch " << patch_count << ": " << e.what() << std::endl;
                    return UnetSegAI_STATUS_FAIED;
                } catch (const std::exception& e) {
                    std::cerr << "Unexpected error during inference: " << e.what() << std::endl;
                    return UnetSegAI_STATUS_FAIED;
                }
                
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