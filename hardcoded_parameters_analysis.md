# 硬编码参数分析与解除计划

## 问题描述

当前静态库DentalCbctOnnxSegDLL中存在大量硬编码参数，导致不同模型无法正确运行。特别是当使用NIfTI格式数据和相应模型时，出现以下错误：

```
[ERROR] ONNX Runtime exception during inference: Got invalid dimensions for input: input_volume for the following indices
 index: 2 Got: 96 Expected: 128
 index: 3 Got: 160 Expected: 128
 index: 4 Got: 160 Expected: 128
```

## 需要解除硬编码的参数清单

### 1. 核心模型参数 (位置: static/src/DentalUnet.cpp:27-44)

| 参数名 | 当前硬编码值 | JSON配置中的对应值 | 说明 |
|--------|-------------|-------------------|------|
| `input_channels` | 1 | 1 | 输入通道数 |
| `num_classes` | 3 | 4 | 输出类别数 **不匹配** |
| `patch_size` | {160, 160, 96} | {128, 128, 128} | 推理块大小 **不匹配** |
| `voxel_spacing` | {0.5810545..., 0.5810545..., 1.0} | {0.699999..., 0.699999..., 0.699999...} | 目标体素间距 **不匹配** |
| `step_size_ratio` | 0.75 | 0.5 (推测) | 滑动窗口重叠比例 |
| `normalization_type` | "CTNormalization" | "ZScoreNormalization" | 归一化方法 **不匹配** |
| `min_max_HU` | {-172.01852..., 1824.9935...} | 无直接对应 | CT值范围 |
| `mean_std_HU` | {274.2257..., 366.0545...} | 无直接对应 | CT值均值和标准差 |

### 2. 数据预处理参数

| 参数名 | 当前硬编码值 | JSON配置中的对应值 | 说明 |
|--------|-------------|-------------------|------|
| `transpose_forward` | {0, 1, 2} | {0, 1, 2} | 数据维度转换 **匹配** |
| `transpose_backward` | {0, 1, 2} | {0, 1, 2} | 数据维度逆转换 **匹配** |
| `use_mirroring` | false | 无直接对应 | 是否使用镜像增强 |

### 3. 强度归一化参数 (来自JSON intensity_properties)

| 参数名 | JSON配置值 | 说明 |
|--------|-----------|------|
| `mean` | 757.3162841796875 | 数据集均值 |
| `std` | 156.12234497070312 | 数据集标准差 |
| `percentile_00_5` | 286.2850341796875 | 0.5%分位数 |
| `percentile_99_5` | 1112.6239013671875 | 99.5%分位数 |
| `min` | 2.7977347373962402 | 最小值 |
| `max` | 2528.466552734375 | 最大值 |

## 错误原因分析

1. **patch_size不匹配**: 硬编码为{160,160,96}，但模型期望{128,128,128}
2. **num_classes不匹配**: 硬编码为3，但模型输出4类
3. **归一化方法不匹配**: 硬编码为CTNormalization，但应该使用ZScoreNormalization
4. **target_spacing不匹配**: 硬编码的voxel_spacing与模型训练时使用的不同

## 解决方案

### 阶段1: 支持JSON配置文件加载
- 为DentalUnet类添加JSON配置文件加载功能
- 从JSON文件中读取所有模型相关参数
- 保持API向后兼容性

### 阶段2: 扩展API接口
- 添加新的API函数支持JSON配置文件路径
- 例如: `DentalCbctSegAI_SetConfigFile(AI_HANDLE handle, const wchar_t* configPath)`

### 阶段3: 自动检测和验证
- 模型加载时自动验证输入输出尺寸
- 提供更详细的错误信息

## 文件位置

### 需要修改的文件:
1. `static/header/DentalUnet.h` - 添加JSON配置加载声明
2. `static/src/DentalUnet.cpp` - 实现JSON配置加载，移除硬编码
3. `header/DentalCbctSegAI_API.h` - 添加新的API函数声明
4. `static/src/DentalCbctSegAI_API.cpp` - 实现新的API函数

### 配置文件:
- `img/checkpoint_best_params.json` - 参考配置格式
- 需要为每个模型创建对应的配置文件

## 优先级

**高优先级 (必须解决)**:
1. patch_size - 直接导致推理失败
2. num_classes - 影响输出结果
3. normalization相关参数 - 影响推理准确性

**中优先级**:
4. voxel_spacing/target_spacing - 影响重采样质量
5. step_size_ratio - 影响推理速度和质量

**低优先级**:
6. transpose相关 - 当前值正确
7. use_mirroring - 功能增强