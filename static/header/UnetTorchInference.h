#ifndef UNET_TORCH_INFERENCE_H
#define UNET_TORCH_INFERENCE_H

#include "UnetSegAI_API.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "CImg.h"
#include <vector>
#include <string>

// Forward declarations
class UnetMain;
struct nnUNetConfig;

class UnetTorchInference {
public:
    // Main sliding window inference function for TorchScript models
    static AI_INT runSlidingWindowTorch(
        UnetMain* parent,
        const nnUNetConfig& config,
        const cimg_library::CImg<float>& preprocessed_volume,
        cimg_library::CImg<float>& predicted_output_prob,
        torch::jit::script::Module& model,
        bool use_gpu
    );

private:
    // Create 3D Gaussian kernel for weighted averaging
    static torch::Tensor create3DGaussianKernel(const std::vector<int64_t>& window_sizes);
    
    // Helper function to convert CImg to Torch Tensor
    static torch::Tensor cimgToTensor(
        const cimg_library::CImg<float>& img,
        const std::vector<int64_t>& shape,
        torch::Device device
    );
    
    // Helper function to copy Tensor data back to CImg
    static void tensorToCImg(
        const torch::Tensor& tensor,
        cimg_library::CImg<float>& img
    );
};

#endif // UNET_TORCH_INFERENCE_H