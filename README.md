# nnuNet_cpp - 3D医学图像分割

基于ONNX Runtime的C++实现的3D医学图像分割程序，专门用于牙科/医学CBCT图像分析。

## 构建脚本说明

### 1. 主程序构建脚本：`build.bat`
- **用途**：构建主程序可执行文件（`testToothSegmentation.exe`）
- **位置**：项目根目录
- **功能**：
  - 创建构建目录
  - 生成Visual Studio 2022解决方案
  - 编译主程序
  - 从`lib/run`目录复制运行时DLL到可执行文件目录

### 2. 静态库构建脚本：`static/build.bat`
- **用途**：构建分割核心DLL库（`DentalCbctOnnxSegDLL.dll`）
- **位置**：`static/`目录
- **功能**：
  - 构建核心分割库
  - 必须在主程序构建之前运行
  - 输出DLL到`lib/`目录

## 项目结构

### 根目录结构
```
nnuNet_cpp/
├── build.bat              # 主程序构建脚本
├── CMakeLists.txt         # 主程序CMake配置
├── CLAUDE.md             # AI助手指令文件
├── README.md             # 本文件
├── src/                  # 主程序源代码
├── header/               # 主程序头文件
├── static/               # 静态库源代码
├── lib/                  # 库文件和依赖项
├── model/                # ONNX模型文件
├── img/                  # 输入医学图像
├── result/               # 输出分割结果
└── build/                # 构建输出目录
    └── bin/
        └── Release/      # 可执行文件位置
            └── testToothSegmentation.exe
```

### 静态库结构（`static/`目录）
```
static/
├── build.bat             # DLL构建脚本
├── CMakeLists.txt        # DLL的CMake配置
├── CLAUDE.md            # DLL专用说明
├── header/              # DLL头文件
│   ├── DentalCbctSegAI_API.h
│   ├── DentalUnet.h
│   ├── framework.h
│   └── pch.h
├── src/                 # DLL源文件
│   ├── DentalCbctSegAI_API.cpp
│   ├── DentalUnet.cpp
│   ├── dllmain.cpp
│   └── pch.cpp
└── build/               # DLL构建输出
```

## 依赖库版本

### ONNX Runtime
- **版本**：1.16.x（根据DLL名称推断）
- **必需的DLL文件**：
  - `onnxruntime.dll`
  - `onnxruntime_providers_shared.dll`
  - `onnxruntime_providers_cuda.dll`
- **获取来源**：[ONNX Runtime发布页面](https://github.com/Microsoft/onnxruntime/releases)

### CUDA运行时
- **版本**：CUDA 12.x
- **必需的DLL文件**：
  - `cudart64_12.dll`
  - `cublas64_12.dll`
  - `cublasLt64_12.dll`
  - `cufft64_11.dll`
  - `curand64_10.dll`
  - `cusolver64_11.dll`
  - `cusparse64_12.dll`
- **获取来源**：[NVIDIA CUDA工具包](https://developer.nvidia.com/cuda-downloads)

### cuDNN
- **版本**：cuDNN 9.x
- **必需的DLL文件**：
  - `cudnn64_9.dll`
  - `cudnn_*64_9.dll`（多个组件）
- **获取来源**：[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

### 重要提示
所有运行时DLL必须放置在`lib/run/`目录中。构建脚本会自动将它们复制到可执行文件目录。

## 构建步骤

1. **首先构建静态库**：
   ```cmd
   cd static
   build.bat
   cd ..
   ```

2. **构建主程序**：
   ```cmd
   build.bat
   ```

## 程序运行流程

1. **初始化阶段**
   - 从`img/`目录加载HDR/Analyze格式的医学图像
   - 创建分割模型实例
   - 设置`model/`目录中的ONNX模型路径

2. **预处理阶段**
   - 将图像数据转换为浮点数
   - 应用CT标准化或Z-score标准化
   - 根据需要调整体素间距

3. **推理阶段**
   - 将体积分割为重叠的块（160×160×96）
   - 对每个块运行ONNX模型
   - 使用高斯加权合并块结果

4. **后处理阶段**
   - 应用argmax获取分割标签
   - 将坐标变换回原始空间
   - 保存结果到`result/`目录

## 数据位置说明

### 输入图像（`img/`目录）
- **格式**：Analyze格式（.hdr + .img文件对）
- **类型**：3D医学图像（CBCT扫描）
- **示例**：`img/Series_5_Acq_2_0000.hdr`

### 模型文件（`model/`目录）
- **格式**：ONNX模型文件（.onnx）
- **示例**：`model/kneeseg_test.onnx`
- **注意**：模型应具有5D输入（批次、通道、深度、高度、宽度）

### 输出结果（`result/`目录）
- **格式**：Analyze格式（.hdr + .img文件对）
- **类型**：3D标签掩码
- **示例**：`result/finalLabelMask.hdr`
- **标签值含义**：
  - 0：背景
  - 1+：不同的解剖结构（取决于模型）

## 部署注意事项

1. 确保`lib/run/`中的所有DLL都存在于可执行文件相同的目录中
2. 程序需要支持CUDA的NVIDIA GPU以获得最佳性能
3. 输入图像必须是Analyze格式（需要.hdr和.img两个文件）
4. 运行前如果`result/`目录不存在，需要先创建

## 故障排除

1. **缺少DLL错误**：运行`build.bat`将复制所有必需的DLL
2. **CUDA错误**：确保安装了兼容的NVIDIA GPU驱动程序
3. **模型加载错误**：验证ONNX模型的兼容性和输入/输出名称
4. **内存错误**：大体积数据可能需要具有足够VRAM的GPU

## API使用说明

静态库提供C风格的API以便集成：

```c
// 创建实例
AI_HANDLE handle = DentalCbctSegAI_CreateObj();

// 设置模型路径
DentalCbctSegAI_SetModelPath(handle, L"path/to/model.onnx");

// 运行推理
AI_DataInfo* input_data = /* 准备输入数据 */;
DentalCbctSegAI_Infer(handle, input_data);

// 获取结果
AI_DataInfo* output_data = /* 准备输出缓冲区 */;
DentalCbctSegAI_GetResult(handle, output_data);

// 清理资源
DentalCbctSegAI_ReleseObj(handle);
```