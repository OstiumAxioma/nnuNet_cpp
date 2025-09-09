#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "LibTorch CUDA Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check CUDA availability
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA Available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    // Check cuDNN availability
    bool cudnn_available = torch::cuda::cudnn_is_available();
    std::cout << "cuDNN Available: " << (cudnn_available ? "Yes" : "No") << std::endl;
    
    // Get device count
    int device_count = torch::cuda::device_count();
    std::cout << "Number of CUDA devices: " << device_count << std::endl;
    
    if (cuda_available && device_count > 0) {
        std::cout << "\n--- CUDA Test ---" << std::endl;
        
        // Create a tensor on CPU
        torch::Tensor cpu_tensor = torch::rand({2, 3});
        std::cout << "CPU Tensor:\n" << cpu_tensor << std::endl;
        std::cout << "Device: " << cpu_tensor.device() << std::endl;
        
        // Move tensor to CUDA
        torch::Tensor cuda_tensor = cpu_tensor.cuda();
        std::cout << "\nCUDA Tensor:\n" << cuda_tensor << std::endl;
        std::cout << "Device: " << cuda_tensor.device() << std::endl;
        
        // Perform operation on CUDA
        torch::Tensor result = torch::matmul(cuda_tensor, cuda_tensor.t());
        std::cout << "\nMatrix multiplication result:\n" << result << std::endl;
        
        // Move back to CPU
        torch::Tensor cpu_result = result.cpu();
        std::cout << "\nResult back on CPU:\n" << cpu_result << std::endl;
        
        std::cout << "\nCUDA operations successful!" << std::endl;
    } else {
        std::cout << "\nCUDA not available. Running CPU test only." << std::endl;
        
        // CPU-only test
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << "CPU Tensor:\n" << tensor << std::endl;
        
        torch::Tensor result = torch::matmul(tensor, tensor.t());
        std::cout << "Matrix multiplication result:\n" << result << std::endl;
    }
    
    // Show configuration
    std::cout << "\n========================================" << std::endl;
    std::cout << "Torch Configuration:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << torch::show_config() << std::endl;
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}