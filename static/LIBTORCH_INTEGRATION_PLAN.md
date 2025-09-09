# LibTorch 集成方案文档

## 项目背景
在现有的 nnUNet ONNX Runtime 推理基础上，添加 LibTorch (PyTorch C++) 推理支持，实现双引擎推理能力。

## 设计原则
1. **保持接口一致性**：不修改现有 API，通过文件扩展名自动识别模型类型
2. **代码复用最大化**：复用所有预处理、后处理和滑窗推理逻辑
3. **向后兼容性**：现有 ONNX 功能完全不受影响
4. **统一配置方式**：无需修改 JSON 配置格式

## 实施方案

### 1. CMake 配置修改

**文件**：`static/CMakeLists.txt`

**修改内容**：
```cmake
# 在 find_package(ITK REQUIRED) 之后添加

# 设置 LibTorch 路径
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch231/share/cmake/Torch")

# 查找 Torch
find_package(Torch REQUIRED)

# 修改 target_include_directories，添加 Torch 头文件
target_include_directories(UnetOnnxSegDLL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/CImg
    ${TORCH_INCLUDE_DIRS}  # 新增
)

# 修改 target_link_libraries，添加 Torch 库
target_link_libraries(UnetOnnxSegDLL PRIVATE
    ${ONNXRUNTIME_LIB}
    ${ONNXRUNTIME_PROVIDERS_SHARED_LIB}
    ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}
    ${ITK_LIBRARIES}
    ${TORCH_LIBRARIES}  # 新增
)

# 确保 C++17 标准（LibTorch 需要）
set(CMAKE_CXX_STANDARD 17)
```

### 2. 新增文件

#### A. `header/UnetTorchInference.h`
```cpp
#ifndef UNET_TORCH_INFERENCE_H
#define UNET_TORCH_INFERENCE_H

#include "UnetDef.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "CImg.h"

class UnetMain;
struct nnUNetConfig;

class UnetTorchInference {
public:
    static AI_INT runSlidingWindowTorch(
        UnetMain* parent,
        const nnUNetConfig& config,
        const cimg_library::CImg<float>& preprocessed_volume,
        cimg_library::CImg<float>& predicted_output_prob,
        torch::jit::script::Module& model,
        bool use_gpu
    );

private:
    static torch::Tensor create3DGaussianKernel(const std::vector<int64_t>& window_sizes);
};

#endif
```

#### B. `src/UnetTorchInference.cpp`
实现要点：
- 复用 `UnetInference.cpp` 的滑窗逻辑
- 将 ONNX Runtime 调用替换为 TorchScript 调用
- 使用 `torch::from_blob` 进行 CImg 到 Tensor 的转换
- 实现与 ONNX 版本相同的高斯权重窗口

### 3. 修改现有文件

#### A. `header/UnetMain.h`
```cpp
// 在 include 部分添加
#include <torch/script.h>

// 在 private 成员中添加
private:
    // 模型后端类型
    enum class ModelBackend {
        ONNX,
        TORCH,
        UNKNOWN
    };
    
    ModelBackend model_backend = ModelBackend::UNKNOWN;
    
    // Torch 模型相关
    torch::jit::script::Module torch_model;
    bool torch_model_loaded = false;
    
    // 辅助函数
    ModelBackend detectModelBackend(const wchar_t* model_path);
    std::string wstringToString(const std::wstring& wstr);
```

#### B. `src/UnetMain.cpp`

**修改 `setModelFns` 函数**：
```cpp
void UnetMain::setModelFns(const wchar_t* model_fn) {
    if (model_fn == nullptr) {
        return;
    }
    
    unetConfig.model_file_name = model_fn;
    
    // 检测模型类型
    model_backend = detectModelBackend(model_fn);
    
    // 根据模型类型初始化
    if (model_backend == ModelBackend::ONNX) {
        initializeSession();  // 现有 ONNX 初始化
    } else if (model_backend == ModelBackend::TORCH) {
        initializeTorchModel();  // 新增 Torch 初始化
    }
}
```

**添加新函数**：
```cpp
ModelBackend UnetMain::detectModelBackend(const wchar_t* model_path) {
    std::wstring path(model_path);
    if (path.ends_with(L".onnx")) {
        return ModelBackend::ONNX;
    } else if (path.ends_with(L".pt") || path.ends_with(L".pth")) {
        return ModelBackend::TORCH;
    }
    return ModelBackend::UNKNOWN;
}

AI_INT UnetMain::initializeTorchModel() {
    try {
        torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);
        
        // 转换宽字符路径为窄字符
        std::string model_path = wstringToString(unetConfig.model_file_name);
        
        // 加载模型
        torch_model = torch::jit::load(model_path, device);
        torch_model.eval();
        torch_model_loaded = true;
        
        std::cout << "TorchScript model loaded successfully" << std::endl;
        std::cout << "Using " << (use_gpu ? "CUDA" : "CPU") << " for inference" << std::endl;
        
        return UnetSegAI_STATUS_SUCCESS;
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load TorchScript model: " << e.what() << std::endl;
        return UnetSegAI_LOADING_FAIED;
    }
}
```

**修改 `performInference` 函数**：
```cpp
// 在滑窗推理部分
if (model_backend == ModelBackend::ONNX) {
    // 现有 ONNX 推理
    AI_INT is_ok = UnetInference::runSlidingWindow(
        this, unetConfig, preprocessed_volume, 
        predicted_output_prob, semantic_seg_session_ptr.get(),
        cached_input_name, cached_output_name
    );
    if (is_ok != UnetSegAI_STATUS_SUCCESS) {
        return is_ok;
    }
} else if (model_backend == ModelBackend::TORCH) {
    // Torch 推理
    if (!torch_model_loaded) {
        std::cerr << "TorchScript model not loaded" << std::endl;
        return UnetSegAI_LOADING_FAIED;
    }
    
    AI_INT is_ok = UnetTorchInference::runSlidingWindowTorch(
        this, unetConfig, preprocessed_volume,
        predicted_output_prob, torch_model, use_gpu
    );
    if (is_ok != UnetSegAI_STATUS_SUCCESS) {
        return is_ok;
    }
} else {
    std::cerr << "Unknown model backend" << std::endl;
    return UnetSegAI_LOADING_FAIED;
}
```

### 4. UnetTorchInference 实现要点

基于 `DentalUnet_cimg_version.cpp` 的实现，主要步骤：

1. **创建高斯核**（第225-254行）
2. **滑窗循环**（第328-400行）
3. **数据转换**：
   ```cpp
   torch::Tensor input_patch_tensor = torch::from_blob(
       input_patch.data(), 
       {1, 1, patch_depth, patch_height, patch_width}, 
       torch::TensorOptions().dtype(torch::kFloat32)
   );
   ```
4. **模型推理**：
   ```cpp
   torch::Tensor output = torch_model.forward({input_patch_tensor}).toTensor();
   ```
5. **应用高斯权重并累加**

### 5. 构建和测试

#### 构建步骤：
1. 在 `static` 目录下运行 `build.bat`
2. 确保 libtorch231 的 DLL 被复制到输出目录

#### 测试方法：
```cpp
// 测试 ONNX 模型（现有功能）
UnetSegAI_SetModelPath(handle, L"..\\..\\..\\model\\model.onnx");

// 测试 TorchScript 模型（新功能）
UnetSegAI_SetModelPath(handle, L"..\\..\\..\\model\\model.pt");
```

### 6. 注意事项

1. **内存管理**：
   - Torch Tensor 使用 `resize_(at::IntArrayRef{0})` 释放内存
   - CImg 使用 `.clear()` 释放内存

2. **GPU 使用**：
   - ONNX: 通过 CUDA Provider
   - Torch: 通过 `torch::cuda::is_available()`
   - 两者使用相同的 `use_gpu` 标志

3. **数据格式**：
   - 保持与 ONNX 版本相同的输入输出格式
   - 维度顺序：[batch, channel, depth, height, width]

4. **错误处理**：
   - 捕获 `c10::Error` 异常
   - 返回统一的错误码

### 7. 性能优化建议

1. **批处理**：如果 GPU 内存足够，可以批量处理多个 patch
2. **内存池**：使用 Torch 的内存池减少分配开销
3. **混合精度**：考虑使用 FP16 推理提高速度

## 实施顺序

1. 第一步：修改 CMakeLists.txt，确保能正确链接 LibTorch
2. 第二步：创建 UnetTorchInference.h/cpp 文件
3. 第三步：修改 UnetMain.h 添加必要的成员和声明
4. 第四步：修改 UnetMain.cpp 实现模型检测和切换逻辑
5. 第五步：实现 UnetTorchInference 的滑窗推理
6. 第六步：构建测试

## 风险点

1. **ABI 兼容性**：确保 LibTorch 与编译器 ABI 兼容
2. **依赖冲突**：ONNX Runtime 和 LibTorch 可能有共同依赖
3. **内存使用**：同时加载两种模型可能增加内存压力

## 备份建议

在开始修改前，备份以下文件：
- static/CMakeLists.txt
- static/src/UnetMain.cpp
- static/header/UnetMain.h

## 运行时依赖（重要）

### 必需的 DLL 文件清单

使用 LibTorch 推理时，需要将以下 DLL 文件复制到可执行文件同目录：

#### CPU 推理必需集合
```
# 核心 DLL
c10.dll              (约 793KB)  - 核心张量库
torch.dll            (约 9.5KB)  - 主入口加载器
torch_cpu.dll        (约 126MB)  - CPU 运算实现
torch_global_deps.dll (约 9.5KB) - 全局依赖

# 额外必需依赖（2025-01-05更新）
fbgemm.dll           (约 2.5MB)  - Facebook GEMM 优化矩阵运算库
asmjit.dll           (约 500KB)  - JIT 编译器，用于运行时代码生成
uv.dll               (约 350KB)  - libuv 异步 I/O 库
mkl_intel_thread.1.dll (约 5MB) - Intel MKL 多线程数学运算库
```

#### GPU 推理额外需要
```
c10_cuda.dll         (约 345KB)  - CUDA 张量操作
torch_cuda.dll       (约 836MB)  - CUDA GPU 运算实现
```

#### 可选依赖（根据模型和系统配置）
```
pytorch_jni.dll      - Java 接口（通常不需要）
mkl_*.dll           - Intel MKL 其他组件（如 mkl_core.dll, mkl_intel_ilp64.dll）
*cudnn*.dll         - cuDNN 库（深度学习优化）
libiomp5md.dll      - OpenMP 运行时（多线程支持）
```

### 文件大小考虑

- **CPU 版本（包含所有必需依赖）**：约 145MB
- **CPU + GPU 版本**：约 1.3GB

建议根据实际需求选择：
- 仅 CPU 推理：只复制 CPU 相关 DLL
- 需要 GPU 加速：复制全部 DLL

## 版本信息

- LibTorch 版本：2.3.1
- ONNX Runtime 版本：（现有）
- 开发日期：2025-01-05
- 更新日期：2025-01-05（添加 DLL 依赖说明）