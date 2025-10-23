#include "UnetInference.h"
#include "UnetMain.h"
#include "UnetIO.h"
#include "UnetSegAI_API.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <memory>
#include <numeric>
#include <algorithm>
#include <string>
#include "../include/SystemMonitor.h"

using namespace std;
using namespace cimg_library;

#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define UNET_HAS_CUDA_RUNTIME 1
#else
#define UNET_HAS_CUDA_RUNTIME 0
#endif
#else
#define UNET_HAS_CUDA_RUNTIME 0
#endif

namespace UnetDebug {
    // 默认关闭：不打印每瓦片日志，也不在环内调用 cudaMemGetInfo
    constexpr bool kTileDebug = false;
    // 仅在每 N 个瓦片上打印/采样一次（例如 100）。当 kTileDebug=false 时，此值无效
    constexpr int  kTileDebugEveryN = 100;
    inline bool ShouldLogTile(std::size_t tile_idx) noexcept {
        return kTileDebug && (tile_idx % kTileDebugEveryN == 0);
    }
}

struct GPUSample {
    bool    valid = false;
    size_t  usedBytes = 0;
    size_t  totalBytes = 0;
    double  usagePercent = 0.0;
};

// 只有在需要时才真正读取 GPU 信息；默认返回 invalid，避免昂贵查询
static inline GPUSample MaybeSampleGPU(std::size_t tile_idx) {
    GPUSample s;
    if (!UnetDebug::ShouldLogTile(tile_idx)) return s; // 默认不采样
    auto info = SystemMonitor::getGPUInfo();
    // 依据你的 SystemMonitor::GPUInfo 字段名进行映射
    s.valid        = info.available;           // 若你的实现没有 available，可改为 true
    s.usedBytes    = info.usedMemory;
    s.totalBytes   = info.totalMemory;
    s.usagePercent = info.memoryUsagePercent;  
    return s;
}

static size_t SafeElementCount(const std::vector<int64_t>& dims) {
    if (dims.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

#if UNET_HAS_CUDA_RUNTIME
static bool CheckCuda(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "[IoBinding] " << message << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

struct InferenceIoBindingContext {
    bool enabled = false;
    Ort::IoBinding binding;
    Ort::MemoryInfo device_memory;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    size_t input_elements = 0;
    size_t output_elements = 0;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    float* device_input = nullptr;
    float* device_output = nullptr;

    InferenceIoBindingContext(Ort::Session& session,
                              const std::vector<int64_t>& in_shape,
                              const std::vector<int64_t>& out_shape)
        : enabled(false),
          binding(session),
          device_memory("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault),
          input_shape(in_shape),
          output_shape(out_shape) {
        input_elements = SafeElementCount(input_shape);
        output_elements = SafeElementCount(output_shape);
        if (input_elements == 0 || output_elements == 0) {
            std::cerr << "[IoBinding] Invalid tensor shape for CUDA binding." << std::endl;
            return;
        }
        input_bytes = input_elements * sizeof(float);
        output_bytes = output_elements * sizeof(float);

        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes), "cudaMalloc input buffer")) {
            return;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes), "cudaMalloc output buffer")) {
            cudaFree(device_input);
            device_input = nullptr;
            return;
        }
        enabled = true;
    }

    ~InferenceIoBindingContext() {
        if (device_input) {
            cudaFree(device_input);
        }
        if (device_output) {
            cudaFree(device_output);
        }
    }

    bool IsReady() const { return enabled; }

    AI_INT Run(Ort::Session& session,
               const float* host_input,
               cimg_library::CImg<float>& host_output,
               const char* input_name,
               const char* output_name) {
        if (!enabled || host_input == nullptr || host_output.data() == nullptr) {
            return UnetSegAI_STATUS_FAIED;
        }

        if (!CheckCuda(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D")) {
            return UnetSegAI_STATUS_FAIED;
        }

        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();

        Ort::Value input_value = Ort::Value::CreateTensor<float>(
            device_memory, device_input, input_elements, input_shape.data(), input_shape.size());
        Ort::Value output_value = Ort::Value::CreateTensor<float>(
            device_memory, device_output, output_elements, output_shape.data(), output_shape.size());

        binding.BindInput(input_name, input_value);
        binding.BindOutput(output_name, output_value);

        session.Run(Ort::RunOptions{ nullptr }, binding);
        binding.SynchronizeOutputs();

        if (!CheckCuda(cudaMemcpy(host_output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) {
            return UnetSegAI_STATUS_FAIED;
        }

        return UnetSegAI_STATUS_SUCCESS;
    }
};
#else
struct InferenceIoBindingContext {
    InferenceIoBindingContext(Ort::Session&, const std::vector<int64_t>&, const std::vector<int64_t>&) {}
    bool IsReady() const { return false; }
    AI_INT Run(Ort::Session&, const float*, cimg_library::CImg<float>&, const char*, const char*) {
        return UnetSegAI_STATUS_FAIED;
    }
};
#endif

// 主推理函数 - 滑窗推理
AI_INT UnetInference::runSlidingWindow(UnetMain* parent,
                                      const nnUNetConfig& config,
                                      const CImg<float>& input,
                                      CImg<float>& output,
                                      Ort::Session* session,
                                      const std::string& input_name,
                                      const std::string& output_name)
{
    // Session已经在外部初始化，直接使用
    if (session == nullptr) {
        std::cerr << "Error: Session pointer is null" << std::endl;
        return UnetSegAI_LOADING_FAIED;
    }
    
    const char* input_name_cstr = input_name.c_str();
    const char* output_name_cstr = output_name.c_str();
    
    try {
        auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        if (input_shape.size() != 5) {
            throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
        }

        // 验证patch_size
        if (config.patch_size.size() != 3) {
            throw std::runtime_error("Patch size should be 3D (depth, height, width)");
        }
        const int num_channels = config.input_channels;
        
        // 判断是否为2D情况（与UnetPreprocessor.cpp中的逻辑一致）
        bool is_2d = config.voxel_spacing.size() == 2;
        
        // 根据2D/3D情况构建ONNX输入张量的形状
        std::vector<int64_t> input_tensor_shape;
        if (is_2d) {
            // ONNX 2D张量形状: (batch, channel, height, width)
            // config.patch_size for 2D is assumed to be {height, width}
            input_tensor_shape = { 1, (int64_t)num_channels, config.patch_size[0], config.patch_size[1] };
        } else {
            // ONNX 3D张量形状: (batch, channel, depth, height, width)
            // config.patch_size for 3D is assumed to be {depth, height, width}
            input_tensor_shape = { 1, (int64_t)num_channels, config.patch_size[0], config.patch_size[1], config.patch_size[2] };
        }

        std::vector<int64_t> output_tensor_shape;
        if (is_2d) {
            output_tensor_shape = { 1, static_cast<int64_t>(config.num_classes), config.patch_size[0], config.patch_size[1] };
        } else {
            output_tensor_shape = { 1, static_cast<int64_t>(config.num_classes), config.patch_size[0], config.patch_size[1], config.patch_size[2] };
        }

        std::unique_ptr<InferenceIoBindingContext> io_binding_context;
#if UNET_HAS_CUDA_RUNTIME
        bool has_cuda_provider = false;
        try {
            auto providers = Ort::GetAvailableProviders();
            has_cuda_provider = std::find(providers.begin(), providers.end(), std::string("CUDAExecutionProvider")) != providers.end();
        } catch (const std::exception& e) {
            std::cerr << "[IoBinding] Unable to query session providers: " << e.what() << std::endl;
        }

        if (has_cuda_provider) {
            auto candidate = std::make_unique<InferenceIoBindingContext>(*session, input_tensor_shape, output_tensor_shape);
            if (candidate->IsReady()) {
                io_binding_context = std::move(candidate);
            } else {
                std::cerr << "[IoBinding] Failed to initialize CUDA IO binding buffers. Falling back to CPU tensors." << std::endl;
            }
        }
#endif

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
        CImg<float> padded_volume(padded_width, padded_height, padded_depth, num_channels, 0.0f);

        // 复制原始数据到padded volume的中心
        if (pad_depth_before>=0 && pad_width_before>=0 && pad_height_before>=0){
            cimg_forXYZC(input, x, y, z, c) {
                padded_volume(x + pad_width_before, y + pad_height_before, z + pad_depth_before, c) = input(x, y, z, c);
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

        // 输出tile总体信息（环外：保留）
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
                    
                    // —— 瓦片级日志：仅在需要时打印 ——
                    if (UnetDebug::ShouldLogTile(patch_count)) {
                        std::cout << "\nProcessing tile #" << patch_count << "/" << total_tiles << "..." << std::endl;
                        std::cout << "  Position: [" << lb_x << "-" << ub_x << ", " 
                                  << lb_y << "-" << ub_y << ", " 
                                  << lb_z << "-" << ub_z << "]" << std::endl;
                    }

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

                    // —— GPU 采样：仅在需要时采样（避免每瓦片 cuda 查询） ——
                    auto gpu_before = MaybeSampleGPU(patch_count);
                    
                    // 记录tile推理开始时间
                    auto tile_start = std::chrono::steady_clock::now();
                    
                    // 执行单个patch推理
                    AI_INT status = inferPatch(*session, input_patch, win_pob, 
                                              input_tensor_shape, input_name_cstr, output_name_cstr, io_binding_context.get());
                    if (status != UnetSegAI_STATUS_SUCCESS) {
                        return status;
                    }
                    
                    // 记录tile推理结束时间
                    auto tile_end = std::chrono::steady_clock::now();
                    std::chrono::duration<double> tile_elapsed = tile_end - tile_start;
                    
                    // —— GPU 采样：仅在需要时采样 ——
                    auto gpu_after = MaybeSampleGPU(patch_count);
                    
                    // —— 瓦片级性能信息：仅在需要时打印 ——
                    if (UnetDebug::ShouldLogTile(patch_count)) {
                        std::cout << "  Tile inference time: " << std::fixed << std::setprecision(3) 
                                  << tile_elapsed.count() << "s" << std::endl;
                        if (gpu_after.valid) {
                            std::cout << "  GPU memory: " << SystemMonitor::formatBytes(gpu_after.usedBytes) 
                                      << " / " << SystemMonitor::formatBytes(gpu_after.totalBytes)
                                      << " (" << std::fixed << std::setprecision(1) 
                                      << gpu_after.usagePercent << "%)" << std::endl;
                        }
                    }

                    // 保存单个tile（如果启用了中间结果保存）
                    if (parent && parent->saveIntermediateResults && !parent->modelOutputPath.empty()) {
                        UnetIO::saveTile(win_pob, patch_count, lb_x, lb_y, lb_z, parent->modelOutputPath);
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
                    
                    if (UnetDebug::ShouldLogTile(patch_count)) {
                        std::cout << "Tile #" << patch_count << " completed" << std::endl;
                    }
                }
            }
        }

        // 归一化
        cimg_forXYZ(padded_output_prob, x, y, z) {
            const float weight = count_vol(x, y, z);
            if (weight > 1e-6f) {
                cimg_forC(padded_output_prob, c) {
                    padded_output_prob(x, y, z, c) /= weight;
                }
            } else {
                // 没有瓦片覆盖该体素，保持为0并发出调试警告（一次性）
                static bool warned_zero_weight = false;
                if (!warned_zero_weight) {
                    std::cerr << "[SlidingWindow] Warning: encountered voxel with zero accumulation weight. "
                              << "This indicates some region was not covered by any tile." << std::endl;
                    warned_zero_weight = true;
                }
                cimg_forC(padded_output_prob, c) {
                    padded_output_prob(x, y, z, c) = 0.0f;
                }
            }
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
                                const char* output_name,
                                InferenceIoBindingContext* io_context)
{
    try {
        // 获取输入数据指针
        const float* input_data_ptr = patch.data();
        if (input_data_ptr == nullptr) {
            return UnetSegAI_STATUS_FAIED;
        }

        if (io_context && io_context->IsReady()) {
            return io_context->Run(session, input_data_ptr, output, input_name, output_name);
        }

        size_t input_element_count = SafeElementCount(input_shape);
        if (input_element_count == 0) {
            std::cerr << "inferPatch: invalid input tensor shape." << std::endl;
            return UnetSegAI_STATUS_FAIED;
        }
        
        // 创建ONNX内存信息和输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
            const_cast<float*>(input_data_ptr),
            input_element_count,
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
        size_t output_patch_vol_sz = static_cast<size_t>(output.size()) * sizeof(float);
        
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
