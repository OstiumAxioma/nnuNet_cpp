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
    if (session == nullptr) {
        std::cerr << "Error: Session pointer is null" << std::endl;
        return UnetSegAI_LOADING_FAIED;
    }

    const char* input_name_cstr = input_name.c_str();
    const char* output_name_cstr = output_name.c_str();

    try {
        auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const bool is_2d = (config.patch_size.size() == 2);

        if (is_2d) {
            if (input_shape.size() != 4) {
                throw std::runtime_error("Expected 4D input for a 2D model (batch, channels, height, width)");
            }
            if (config.patch_size.size() != 2) {
                throw std::runtime_error("Patch size should be 2D (height, width) for a 2D model");
            }
        } else {
            if (input_shape.size() != 5) {
                throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
            }
            if (config.patch_size.size() != 3) {
                throw std::runtime_error("Patch size should be 3D (depth, height, width)");
            }
        }

        const int num_channels = config.input_channels;
        std::vector<int64_t> input_tensor_shape = is_2d
            ? std::vector<int64_t>{ 1, static_cast<int64_t>(num_channels), config.patch_size[0], config.patch_size[1] }
            : std::vector<int64_t>{ 1, static_cast<int64_t>(num_channels), config.patch_size[0], config.patch_size[1], config.patch_size[2] };

        std::vector<int64_t> output_tensor_shape = is_2d
            ? std::vector<int64_t>{ 1, static_cast<int64_t>(config.num_classes), config.patch_size[0], config.patch_size[1] }
            : std::vector<int64_t>{ 1, static_cast<int64_t>(config.num_classes), config.patch_size[0], config.patch_size[1], config.patch_size[2] };

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

        const int depth = input.depth();
        const int width = input.width();
        const int height = input.height();

        int padded_depth = depth;
        int padded_width = width;
        int padded_height = height;

        if (is_2d) {
            if (padded_height < config.patch_size[0]) padded_height = static_cast<int>(config.patch_size[0]);
            if (padded_width  < config.patch_size[1]) padded_width  = static_cast<int>(config.patch_size[1]);
        } else {
            if (padded_depth  < config.patch_size[0]) padded_depth  = static_cast<int>(config.patch_size[0]);
            if (padded_height < config.patch_size[1]) padded_height = static_cast<int>(config.patch_size[1]);
            if (padded_width  < config.patch_size[2]) padded_width  = static_cast<int>(config.patch_size[2]);
        }

        int pad_depth_before = (padded_depth - depth) / 2;
        int pad_depth_after  = padded_depth - depth - pad_depth_before;
        int pad_width_before = (padded_width - width) / 2;
        int pad_width_after  = padded_width - width - pad_width_before;
        int pad_height_before = (padded_height - height) / 2;
        int pad_height_after  = padded_height - height - pad_height_before;

        CImg<float> padded_volume(padded_width, padded_height, padded_depth, num_channels, 0.0f);
        if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
            cimg_forXYZC(input, x, y, z, c) {
                padded_volume(x + pad_width_before,
                              y + pad_height_before,
                              z + pad_depth_before,
                              c) = input(x, y, z, c);
            }
        } else {
            padded_volume = input;
        }

        int working_depth  = padded_depth;
        int working_width  = padded_width;
        int working_height = padded_height;

        const float step_size_ratio = config.step_size_ratio;

        int X_num_steps = 0;
        int Y_num_steps = 0;
        int Z_num_steps = 0;
        float actualStepSize[3] = { 0.f, 0.f, 0.f };

        if (is_2d) {
            const int64_t patch_h = config.patch_size[0];
            const int64_t patch_w = config.patch_size[1];

            float target_step_y = patch_h * step_size_ratio;
            float target_step_x = patch_w * step_size_ratio;

            Y_num_steps = std::max(1, static_cast<int>(std::ceil(float(working_height - patch_h) / target_step_y) + 1));
            X_num_steps = std::max(1, static_cast<int>(std::ceil(float(working_width - patch_w) / target_step_x) + 1));
            Z_num_steps = working_depth;

            actualStepSize[0] = (X_num_steps > 1) ? float(working_width - patch_w) / (X_num_steps - 1) : 0.f;
            actualStepSize[1] = (Y_num_steps > 1) ? float(working_height - patch_h) / (Y_num_steps - 1) : 0.f;
            actualStepSize[2] = 1.0f;
        } else {
            const int64_t patch_d = config.patch_size[0];
            const int64_t patch_h = config.patch_size[1];
            const int64_t patch_w = config.patch_size[2];

            float target_step_z = patch_d * step_size_ratio;
            float target_step_y = patch_h * step_size_ratio;
            float target_step_x = patch_w * step_size_ratio;

            Z_num_steps = std::max(1, static_cast<int>(std::ceil(float(working_depth - patch_d) / target_step_z) + 1));
            Y_num_steps = std::max(1, static_cast<int>(std::ceil(float(working_height - patch_h) / target_step_y) + 1));
            X_num_steps = std::max(1, static_cast<int>(std::ceil(float(working_width - patch_w) / target_step_x) + 1));

            actualStepSize[0] = (X_num_steps > 1) ? float(working_width - patch_w) / (X_num_steps - 1) : 0.f;
            actualStepSize[1] = (Y_num_steps > 1) ? float(working_height - patch_h) / (Y_num_steps - 1) : 0.f;
            actualStepSize[2] = (Z_num_steps > 1) ? float(working_depth - patch_d) / (Z_num_steps - 1) : 0.f;
        }

        CImg<float> padded_output_prob(working_width, working_height, working_depth, config.num_classes, 0.f);
        CImg<float> count_vol(working_width, working_height, working_depth, 1, 0.f);

        CImg<float> win_pob;
        CImg<float> gaussisan_weight;
        if (is_2d) {
            win_pob.assign(config.patch_size[1], config.patch_size[0], 1, config.num_classes, 0.f);
            gaussisan_weight.assign(config.patch_size[1], config.patch_size[0], 1, 1, 0.f);
        } else {
            win_pob.assign(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.f);
            gaussisan_weight.assign(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
        }
        createGaussianKernel(gaussisan_weight, config.patch_size);

        const int total_tiles = X_num_steps * Y_num_steps * Z_num_steps;
        std::cout << "Total tiles to process: " << total_tiles << std::endl;
        if (is_2d) {
            std::cout << "Tile grid (2D): " << X_num_steps << " x " << Y_num_steps
                      << " per slice, slices: " << Z_num_steps << std::endl;
            std::cout << "Patch size: " << config.patch_size[1] << " x " << config.patch_size[0]
                      << " (W x H)" << std::endl;
        } else {
            std::cout << "Tile grid (3D): " << X_num_steps << " x " << Y_num_steps << " x " << Z_num_steps
                      << " (X x Y x Z)" << std::endl;
            std::cout << "Patch size: " << config.patch_size[2] << " x " << config.patch_size[1]
                      << " x " << config.patch_size[0] << " (W x H x D)" << std::endl;
        }

        int patch_count = 0;
        for (int sz = 0; sz < Z_num_steps; ++sz) {
            int lb_z = 0;
            int ub_z = 0;
            if (is_2d) {
                lb_z = sz;
                ub_z = lb_z;
            } else {
                const int64_t patch_d = config.patch_size[0];
                lb_z = static_cast<int>(std::round(sz * actualStepSize[2]));
                if (lb_z + patch_d > working_depth) {
                    lb_z = working_depth - static_cast<int>(patch_d);
                }
                lb_z = std::max(0, lb_z);
                ub_z = lb_z + static_cast<int>(patch_d) - 1;
            }

            for (int sy = 0; sy < Y_num_steps; ++sy) {
                const int64_t patch_h = is_2d ? config.patch_size[0] : config.patch_size[1];
                int lb_y = static_cast<int>(std::round(sy * actualStepSize[1]));
                if (lb_y + patch_h > working_height) {
                    lb_y = working_height - static_cast<int>(patch_h);
                }
                lb_y = std::max(0, lb_y);
                int ub_y = lb_y + static_cast<int>(patch_h) - 1;

                for (int sx = 0; sx < X_num_steps; ++sx) {
                    const int64_t patch_w = is_2d ? config.patch_size[1] : config.patch_size[2];
                    int lb_x = static_cast<int>(std::round(sx * actualStepSize[0]));
                    if (lb_x + patch_w > working_width) {
                        lb_x = working_width - static_cast<int>(patch_w);
                    }
                    lb_x = std::max(0, lb_x);
                    int ub_x = lb_x + static_cast<int>(patch_w) - 1;

                    patch_count += 1;

                    if (UnetDebug::ShouldLogTile(patch_count)) {
                        std::cout << "\nProcessing tile #" << patch_count << "/" << total_tiles << "..." << std::endl;
                        if (is_2d) {
                            std::cout << "  Position: [" << lb_x << "-" << ub_x << ", "
                                      << lb_y << "-" << ub_y << "] slice " << lb_z << std::endl;
                        } else {
                            std::cout << "  Position: [" << lb_x << "-" << ub_x << ", "
                                      << lb_y << "-" << ub_y << ", "
                                      << lb_z << "-" << ub_z << "]" << std::endl;
                        }
                    }

                    CImg<float> input_patch;
                    try {
                        input_patch = padded_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z);
                    } catch (const CImgException& e) {
                        std::cerr << "Error extracting patch: " << e.what() << std::endl;
                        return UnetSegAI_STATUS_FAIED;
                    }

                    if (is_2d) {
                        if (input_patch.width() != config.patch_size[1] ||
                            input_patch.height() != config.patch_size[0] ||
                            input_patch.depth() != 1) {
                            return UnetSegAI_STATUS_FAIED;
                        }
                    } else {
                        if (input_patch.width() != config.patch_size[2] ||
                            input_patch.height() != config.patch_size[1] ||
                            input_patch.depth() != config.patch_size[0]) {
                            return UnetSegAI_STATUS_FAIED;
                        }
                    }

                    auto gpu_before = MaybeSampleGPU(patch_count);
                    (void)gpu_before;

                    auto tile_start = std::chrono::steady_clock::now();

                    AI_INT status = inferPatch(*session,
                                               input_patch,
                                               win_pob,
                                               input_tensor_shape,
                                               input_name_cstr,
                                               output_name_cstr,
                                               io_binding_context.get());
                    if (status != UnetSegAI_STATUS_SUCCESS) {
                        return status;
                    }

                    auto tile_end = std::chrono::steady_clock::now();
                    std::chrono::duration<double> tile_elapsed = tile_end - tile_start;

                    auto gpu_after = MaybeSampleGPU(patch_count);

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

                    if (parent && parent->saveIntermediateResults && !parent->modelOutputPath.empty()) {
                        UnetIO::saveTile(win_pob, patch_count, lb_x, lb_y, lb_z, parent->modelOutputPath);
                    }

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
                    } catch (const std::exception&) {
                        return UnetSegAI_STATUS_FAIED;
                    }

                    if (UnetDebug::ShouldLogTile(patch_count)) {
                        std::cout << "Tile #" << patch_count << " completed" << std::endl;
                    }
                }
            }
        }

        cimg_forXYZ(padded_output_prob, x, y, z) {
            const float weight = count_vol(x, y, z);
            if (weight > 1e-6f) {
                cimg_forC(padded_output_prob, c) {
                    padded_output_prob(x, y, z, c) /= weight;
                }
            } else {
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

        output = CImg<float>(width, height, depth, config.num_classes, 0.f);
        if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
            cimg_forXYZC(output, x, y, z, c) {
                output(x, y, z, c) = padded_output_prob(x + pad_width_before,
                                                        y + pad_height_before,
                                                        z + pad_depth_before,
                                                        c);
            }
        } else {
            output = padded_output_prob;
        }

        const float min_weight = count_vol.min();
        const float max_weight = count_vol.max();
        std::cout << "Sliding window accumulation weight range: [" << min_weight << ", " << max_weight << "]" << std::endl;
        std::cout << "Sliding window inference is done." << std::endl;
        return UnetSegAI_STATUS_SUCCESS;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return UnetSegAI_LOADING_FAIED;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return UnetSegAI_STATUS_FAIED;
    }
}

// 创建高斯核（兼容2D和3D）
void UnetInference::createGaussianKernel(CImg<float>& gaussisan_weight,
                                        const std::vector<int64_t>& patch_sizes)
{
    const bool is_2d = (patch_sizes.size() == 2);
    float sigma_scale = 1.0f / 8.0f;
    float value_scaling_factor = 10.0f;

    if (is_2d) {
        int64_t height = patch_sizes[0];
        int64_t width  = patch_sizes[1];

        float y_center = (height - 1) / 2.0f;
        float x_center = (width - 1)  / 2.0f;

        float y_sigma = height * sigma_scale;
        float x_sigma = width  * sigma_scale;

        cimg_forXY(gaussisan_weight, x, y) {
            float y_part = std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2));
            float x_part = std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2));
            gaussisan_weight(x, y, 0) = y_part * x_part;
        }
    } else {
        int64_t depth  = patch_sizes[0];
        int64_t height = patch_sizes[1];
        int64_t width  = patch_sizes[2];

        float z_center = (depth - 1)  / 2.0f;
        float y_center = (height - 1) / 2.0f;
        float x_center = (width - 1)  / 2.0f;

        float z_sigma = depth  * sigma_scale;
        float y_sigma = height * sigma_scale;
        float x_sigma = width  * sigma_scale;

        cimg_forXYZ(gaussisan_weight, x, y, z) {
            float z_part = std::exp(-0.5f * std::pow((z - z_center) / z_sigma, 2));
            float y_part = std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2));
            float x_part = std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2));
            gaussisan_weight(x, y, z) = z_part * y_part * x_part;
        }
    }

    float max_val = gaussisan_weight.max();
    if (max_val > 0) {
        gaussisan_weight *= (value_scaling_factor / max_val);
    }

    float min_non_zero = std::numeric_limits<float>::max();
    if (is_2d) {
        cimg_forXY(gaussisan_weight, x, y) {
            if (gaussisan_weight(x, y, 0) > 0 && gaussisan_weight(x, y, 0) < min_non_zero) {
                min_non_zero = gaussisan_weight(x, y, 0);
            }
        }
        cimg_forXY(gaussisan_weight, x, y) {
            if (gaussisan_weight(x, y, 0) == 0) {
                gaussisan_weight(x, y, 0) = min_non_zero;
            }
        }
    } else {
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
}

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
