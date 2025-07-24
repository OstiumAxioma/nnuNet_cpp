# nnUNet C++ 实现调试修复总结

本文档记录了在调试 nnUNet C++ 实现过程中遇到并解决的所有问题。

## 1. ONNX Runtime 内存管理错误（程序崩溃）

### 问题描述
- **位置**: `static/src/DentalUnet.cpp`, 第357-358行
- **错误**: 程序在运行时出现 "Invalid input name" 错误后立即发生段错误
- **原因**: `GetInputNameAllocated()` 返回的是 `AllocatedStringPtr` 智能指针，直接使用 `.get()` 获取原始指针后，智能指针离开作用域时会释放内存，导致后续使用时访问已释放的内存

### 解决方案
```cpp
// 错误代码
const char* input_name = session.GetInputNameAllocated(0, allocator).get();
const char* output_name = session.GetOutputNameAllocated(0, allocator).get();

// 修复后的代码
Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
const char* input_name = input_name_ptr.get();
const char* output_name = output_name_ptr.get();
```

通过保存 `AllocatedStringPtr` 对象，确保在整个函数执行期间内存保持有效。

## 2. 向量下标越界错误

### 问题描述
- **位置**: `static/src/DentalUnet.cpp`, 第204-205行
- **错误**: "vector subscript out of range" 断言失败
- **原因**: 访问 `config.patch_size[3]` 时，向量只有3个元素（索引0-2）

### 解决方案
```cpp
// 错误代码
std::cout << "  config.patch_size[0-3]: " << config.patch_size[0] << ", " << config.patch_size[1] << ", " << config.patch_size[2] << ", " << config.patch_size[3] << endl;

// 修复后的代码
std::cout << "  config.patch_size[0-2]: " << config.patch_size[0] << ", " << config.patch_size[1] << ", " << config.patch_size[2] << endl;
```

## 3. Tile 位置计算错误（缩放空间不匹配）

### 问题描述
- **现象**: C++ 生成的 tile bounds 与 Python nnUNet 完全不同，导致分割结果缺失大量区域
- **原因**: C++ 在原始空间计算 tile 位置，而 Python 在缩放后的空间计算

### 调试过程
1. 发现 C++ 输出大量 NaN 值（10.9M vs Python 的 76K）
2. 对比 tile bounds 发现完全不匹配：
   - Python tile 0: [0, 0, 0] 对应 C++ tile 0: [0, 0, 0]
   - Python tile 1: [0, 0, 29] 对应 C++ tile 2: [0, 0, 48]
   - Python tile 2: [4, 0, 0] 对应 C++ tile 1: [0, 0, 0]（重复）
   - Python tile 3: [4, 0, 29] 对应 C++ tile 3: [0, 0, 48]

## 4. 图像缩放未执行问题

### 问题描述
- **现象**: 缩放因子显示为 1,1,1 而不是预期的 ~0.662, 0.662, 0.625
- **原因**: 主程序传递的是归一化后的 spacing（0.581, 0.581, 1.0）而不是原始 spacing（0.385, 0.385, 0.625）

### 解决方案
1. 修改 `AI_DataInfo` 结构，添加原始 spacing 字段：
```cpp
typedef struct {
    // ... 现有字段 ...
    AI_FLOAT OriginalVoxelSpacingX;
    AI_FLOAT OriginalVoxelSpacingY;
    AI_FLOAT OriginalVoxelSpacingZ;
} AI_DataInfo;
```

2. 修改 `DentalUnet` 类，保存原始 spacing：
```cpp
std::vector<float> original_voxel_spacing;
std::vector<float> transposed_original_voxel_spacing;
```

3. 修改缩放计算逻辑，使用原始 spacing：
```cpp
// 使用原始spacing计算缩放因子
scaled_factor = transposed_original_voxel_spacing[i] / config.voxel_spacing[i];
```

## 5. Origin 和 Spacing 混淆导致的尺寸计算错误

### 问题描述
- **现象**: 图像缩放后的尺寸不正确，导致 tile bounds 计算错误
- **根本原因**: 混淆了三种不同的 spacing 概念：
  1. **原始 spacing**（从HDR文件读取的真实物理spacing）: 0.385, 0.385, 0.625
  2. **当前 spacing**（主程序传入的归一化spacing）: 0.581, 0.581, 1.0
  3. **目标 spacing**（模型配置的目标spacing）: 0.581, 0.581, 1.0

### 问题分析
- C++ 代码原本使用当前 spacing 计算缩放因子：`current_spacing / target_spacing`
- 由于当前 spacing 已经是归一化后的值，所以计算结果总是 1.0
- 正确的做法应该使用原始 spacing 计算：`original_spacing / target_spacing`

### 解决方案
通过区分这三种 spacing，确保：
1. 从 HDR 文件读取真实的原始 spacing
2. 将原始 spacing 通过新增的字段传递给静态库
3. 在计算缩放因子时使用原始 spacing 而非当前 spacing

这个修复确保了图像能够正确缩放到目标分辨率，从而使 tile 计算在正确的坐标空间中进行。

## 6. Tile 数量计算错误（生成0个tile）

### 问题描述
- **现象**: 缩放正确执行后（285×240×160 → 189×160×100），但生成的 tile 数量为 0
- **原因**: 当缩放后的维度接近 patch size 时，原有的计算公式会产生 0 个 tile

### 原始错误逻辑
```cpp
int X_num_steps = (int)ceil(float(width - config.patch_size[0]) / (config.patch_size[0] * step_size_ratio));
// 当 width=189, patch_size[0]=160, step_size_ratio=0.5 时：
// X_num_steps = ceil((189-160)/(160*0.5)) = ceil(29/80) = ceil(0.36) = 1
// 但对于 Y 轴，height=160 正好等于 patch_size[1]：
// Y_num_steps = ceil((160-160)/(160*0.5)) = ceil(0/80) = 0
```

### 修复后的逻辑
```cpp
// 使用与Python nnUNet相同的tile计算逻辑
actualStepSize[0] = config.patch_size[0] * step_size_ratio;
actualStepSize[1] = config.patch_size[1] * step_size_ratio;
actualStepSize[2] = config.patch_size[2] * step_size_ratio;

// 确保至少有1个tile
int X_num_steps = std::max(1, (int)ceil(float(width - config.patch_size[0]) / actualStepSize[0]) + 1);
int Y_num_steps = std::max(1, (int)ceil(float(height - config.patch_size[1]) / actualStepSize[1]) + 1);
int Z_num_steps = std::max(1, (int)ceil(float(depth - config.patch_size[2]) / actualStepSize[2]) + 1);

// 当维度小于patch size时，调整步数为1
if (width <= config.patch_size[0]) X_num_steps = 1;
if (height <= config.patch_size[1]) Y_num_steps = 1;
if (depth <= config.patch_size[2]) Z_num_steps = 1;
```

## 7. Tile 边界越界保护

### 问题描述
- **现象**: 当 tile 边界超出图像范围时会导致错误
- **解决方案**: 添加边界检查和调整逻辑

```cpp
// 确保不超出边界
if (lb_z + config.patch_size[2] > depth) {
    lb_z = depth - config.patch_size[2];
}
lb_z = std::max(0, lb_z);
```

## 8. 其他重要修复

### 8.1 中间结果保存功能
- 添加了保存预处理、模型输出和后处理结果的功能
- 实现了 `SetOutputPaths` API 函数
- 添加了 HDR 和 raw 格式的保存功能

### 8.2 调试信息增强
- 添加了详细的缩放计算日志
- 添加了 tile 计算过程的调试输出
- 添加了每个 tile 的边界信息输出

## 总结

通过以上修复，成功解决了 C++ nnUNet 实现的关键问题：

1. **内存管理错误** - 确保了程序的基本运行
2. **数组越界错误** - 避免了运行时崩溃
3. **Spacing 传递问题** - 实现了正确的图像缩放
4. **Tile 计算逻辑** - 确保生成足够的 tile 覆盖整个图像
5. **边界处理** - 避免了越界访问

这些修复使得 C++ 实现能够正确执行 nnUNet 的滑动窗口推理，生成与 Python 版本一致的分割结果。

## 修改的文件列表

1. `D:\Project\nnuNet_cpp\static\src\DentalUnet.cpp`
2. `D:\Project\nnuNet_cpp\static\src\DentalCbctSegAI_API.cpp`
3. `D:\Project\nnuNet_cpp\static\header\DentalUnet.h`
4. `D:\Project\nnuNet_cpp\static\header\DentalCbctSegAI_API.h`
5. `D:\Project\nnuNet_cpp\header\DentalCbctSegAI_API.h`
6. `D:\Project\nnuNet_cpp\src\testToothSegmentation.cpp`