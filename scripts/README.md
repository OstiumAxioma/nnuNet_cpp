# 结果比较工具使用指南

本目录包含用于比较C++和Python实现结果的工具脚本。

## 文件说明

- `convert_raw_to_npy.py` - 将C++输出的raw格式转换为numpy格式
- `visualize_results.py` - 可视化单个实现的结果
- `compare_results.py` - 比较C++和Python实现的结果
- `convert_all_results.bat` - 批量转换脚本

## 使用流程

### 1. 准备数据

假设你有以下目录结构：
```
D:\Project\nnuNet_cpp\result\              # C++结果
├── preprocess\
│   ├── preprocessed_normalized_volume.raw
│   └── preprocessed_normalized_volume_meta.txt
├── model_output\
│   ├── model_output_probability.raw
│   └── model_output_probability_meta.txt
└── postprocess\
    ├── postprocessed_segmentation_mask.raw
    └── postprocessed_segmentation_mask_meta.txt

D:\Project\python_results\                  # Python结果
├── preprocessed_normalized_volume.npy
├── model_output_probability.npy
└── postprocessed_segmentation_mask.npy
```

### 2. 转换C++结果为numpy格式

```bash
cd scripts

# 转换所有raw文件
python convert_raw_to_npy.py ../result -r

# 或使用批处理脚本
convert_all_results.bat
```

### 3. 比较结果

#### 比较单个文件
```bash
# 比较预处理结果
python compare_results.py files ^
    ../result/preprocess/preprocessed_normalized_volume.npy ^
    ../../python_results/preprocessed_normalized_volume.npy ^
    -o preprocess_comparison.png

# 指定切片
python compare_results.py files ^
    ../result/preprocess/preprocessed_normalized_volume.npy ^
    ../../python_results/preprocessed_normalized_volume.npy ^
    --slice 50
```

#### 比较整个pipeline
```bash
# 比较所有阶段
python compare_results.py pipeline ^
    ../result ^
    ../../python_results ^
    -o comparison_plots

# 这会生成三个比较图：
# - preprocessed_normalized_volume_comparison.png
# - model_output_probability_comparison.png  
# - postprocessed_segmentation_mask_comparison.png
```

## 输出解释

### 比较指标

脚本会输出以下指标：

1. **差异指标**
   - `Max absolute difference`: 最大绝对差值
   - `Mean absolute difference`: 平均绝对差值
   - `RMSE`: 均方根误差
   - `Max/Mean relative difference`: 相对误差

2. **相似性指标**
   - `Correlation coefficient`: 相关系数（越接近1越相似）
   - `Exact match ratio`: 完全相等的元素比例
   - `Close match ratio`: 在容差范围内相等的元素比例

### 可视化图表

每个比较会生成包含6个子图的图表：

1. **第一行**
   - C++结果切片
   - Python结果切片
   - 差异图（红蓝色表示正负差异）

2. **第二行**
   - 值分布直方图
   - 差异分布直方图
   - 散点图（理想情况下点应在y=x线上）

## 示例：判断结果是否一致

```python
# 如果看到类似以下输出，说明结果基本一致：
"""
Difference Metrics:
  Max absolute difference: 1.234567e-06
  Mean absolute difference: 5.678901e-07
  RMSE: 8.901234e-07
  
Similarity Metrics:
  Correlation coefficient: 0.999999
  Exact match ratio: 0.000000      # 浮点数很难完全相等
  Close match ratio (rtol=1e-6): 0.999950
  Close match ratio (rtol=1e-3): 1.000000
"""

# 如果看到较大差异：
"""
Difference Metrics:
  Max absolute difference: 5.432100e+00
  Mean absolute difference: 1.234567e+00
  
Similarity Metrics:
  Correlation coefficient: 0.850000
  Close match ratio (rtol=1e-3): 0.750000
"""
# 这表明两个实现有显著差异，需要检查
```

## 常见问题

1. **形状不匹配**
   - 检查两个实现的数据布局（如 CHW vs HWC）
   - 检查是否有维度顺序差异

2. **数值差异大**
   - 检查归一化方法是否一致
   - 检查数据类型转换是否正确
   - 检查模型权重是否相同

3. **文件找不到**
   - 确保文件名匹配
   - 检查目录结构是否正确