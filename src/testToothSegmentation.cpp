// testToothSegmentation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <memory>
#include <chrono>
#include <windows.h>
#include <exception>
#include <clocale>
#include <cstdlib>
#include <direct.h>  // for _mkdir
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>


// onnx 相关
#include "..\\header\\UnetSegAI_API.h"
#include "..\\lib\\onnxruntime\\include\\onnxruntime_cxx_api.h"
#pragma comment(lib, "..\\\\lib\\\\UnetOnnxSegDLL.lib")

#pragma comment(lib,"..\\\\lib\\\\onnxruntime\\\\lib\\\\onnxruntime.lib")
#pragma comment(lib,"..\\\\lib\\\\onnxruntime\\\\lib\\\\onnxruntime_providers_shared.lib")
#pragma comment(lib,"..\\\\lib\\\\onnxruntime\\\\lib\\\\onnxruntime_providers_cuda.lib")


//CImg用于二进制存储输入输出
#define cimg_display_type 2
#include "..\\lib\\CImg\\CImg.h"

//ITK用于读写医学图像并保留origin信息
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

using namespace std;
using namespace cimg_library;

// testToothSegmentation现在只负责文件读取，JSON解析已移至静态库

// 读取JSON文件内容
string readJsonFile(const string& jsonPath) {
    ifstream file(jsonPath);
    if (!file.is_open()) {
        cout << "Cannot open config file: " << jsonPath << endl;
        return "";
    }
    
    string line, jsonContent;
    while (getline(file, line)) {
        jsonContent += line;
    }
    file.close();
    
    return jsonContent;
}

// 简化的文件选择函数
string selectFileFromDirectory(const string& directory, const string& fileType, const vector<string>& allowedExtensions) {
    vector<string> files;
    
    if (!filesystem::exists(directory)) {
        cout << "Directory " << directory << " does not exist!" << endl;
        return "";
    }
    
    // 遍历目录获取文件
    for (const auto& entry : filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            string filename = entry.path().filename().string();
            string extension = entry.path().extension().string();
            transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // 检查扩展名
            for (const auto& allowedExt : allowedExtensions) {
                string lowerAllowedExt = allowedExt;
                transform(lowerAllowedExt.begin(), lowerAllowedExt.end(), lowerAllowedExt.begin(), ::tolower);
                if (extension == lowerAllowedExt || 
                    (allowedExt == ".nii.gz" && filename.length() > 7 && 
                     filename.substr(filename.length() - 7) == ".nii.gz")) {
                    files.push_back(filename);
                    break;
                }
            }
        }
    }
    
    if (files.empty()) {
        cout << "No " << fileType << " files found!" << endl;
        return "";
    }
    
    // 显示文件列表
    cout << "\n" << fileType << " files:" << endl;
    for (size_t i = 0; i < files.size(); ++i) {
        cout << "[" << (i + 1) << "] " << files[i] << endl;
    }
    
    // 用户选择
    int choice;
    cout << "Select file (1-" << files.size() << "): ";
    cin >> choice;
    
    if (choice < 1 || choice > (int)files.size()) {
        cout << "Invalid selection!" << endl;
        return "";
    }
    
    return directory + "\\\\\\\\" + files[choice - 1];
}

int main()
{
    try {
        cout << "程序开始运行..." << endl;
        
        // 环境检查
        cout << "\n======= Environment Check =======" << endl;
        cout << "Working directory: " << filesystem::current_path() << endl;
        
        // 检查关键目录是否存在
        vector<string> checkDirs = {"..\\\\..\\\\..\\\\param", "..\\\\..\\\\..\\\\model", "..\\\\..\\\\..\\\\img"};
        for (const auto& dir : checkDirs) {
            if (filesystem::exists(dir)) {
                cout << "✓ Directory exists: " << dir << endl;
            } else {
                cout << "✗ Directory missing: " << dir << endl;
            }
        }
        
        // 检查DLL是否存在
        vector<string> checkDLLs = {
            // Core DLLs
            "UnetOnnxSegDLL.dll", 
            "onnxruntime.dll", 
            "onnxruntime_providers_cuda.dll",
            // LibTorch CPU DLLs
            "torch_cpu.dll",      // LibTorch CPU inference
            "c10.dll",            // LibTorch core tensor library
            "torch.dll",          // LibTorch main library
            "torch_global_deps.dll", // LibTorch global dependencies
            // LibTorch CUDA DLLs (for GPU support)
            "torch_cuda.dll",     // LibTorch CUDA operations (required for GPU)
            "c10_cuda.dll",       // CUDA tensor operations (required for GPU)
            // LibTorch additional dependencies
            "fbgemm.dll",         // Facebook GEMM library for optimized matrix operations
            "asmjit.dll",         // JIT compiler for runtime code generation
            "uv.dll",             // libuv for async I/O operations
            "mkl_intel_thread.1.dll",  // Intel MKL for optimized math operations
            // Additional Intel MKL DLLs (required for CPU inference)
            "mkl_avx2.1.dll",     // Intel MKL AVX2 optimizations
            "mkl_def.1.dll"       // Intel MKL default implementations
        };
        for (const auto& dll : checkDLLs) {
            if (filesystem::exists(dll)) {
                cout << "✓ DLL found: " << dll << endl;
            } else {
                cout << "✗ DLL missing: " << dll << " (optional for TorchScript models)" << endl;
            }
        }
        
        //===== 用户选择配置文件 =====
        cout << "\n======= Config File Selection =======" << endl;
        vector<string> configExtensions = {".json"};
        string configPath = selectFileFromDirectory("..\\\\..\\\\..\\\\param", "config", configExtensions);
        if (configPath.empty()) {
            cout << "Failed to select config file!" << endl;
            system("pause");
            return -1;
        }
        
        //===== 用户选择模型文件 =====
        cout << "\n======= Model File Selection =======" << endl;
        vector<string> modelExtensions = {".onnx", ".pt", ".pth"};  // Support both ONNX and TorchScript models
        string modelPath = selectFileFromDirectory("..\\\\..\\\\..\\\\model", "model", modelExtensions);
        if (modelPath.empty()) {
            cout << "Failed to select model file!" << endl;
            system("pause");
            return -1;
        }
        
        //===== 用户选择输入数据文件 =====
        cout << "\n======= Input Data File Selection =======" << endl;
        vector<string> inputExtensions = {".hdr", ".nii", ".nii.gz"};
        string inputHdrPath = selectFileFromDirectory("..\\\\..\\\\..\\\\img", "input data", inputExtensions);
        if (inputHdrPath.empty()) {
            cout << "Failed to select input data file!" << endl;
            system("pause");
            return -1;
        }
        
        //===== 读取JSON配置文件内容 =====
        cout << "\n======= Loading Configuration =======" << endl;
        string jsonContent = readJsonFile(configPath);
        if (jsonContent.empty()) {
            cout << "Failed to read config file!" << endl;
            system("pause");
            return -1;
        }
        cout << "Configuration loaded successfully." << endl;
        
        //===== 使用ITK加载图像 =====
        cout << "\n======= Loading Image =======" << endl;
        cout << "Loading: " << inputHdrPath << endl;
        
        // ITK图像类型定义
        using PixelType = short;
        using ImageType = itk::Image<PixelType, 3>;
        using ReaderType = itk::ImageFileReader<ImageType>;
        
        // 使用ITK读取图像
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName(inputHdrPath);
        
        ImageType::Pointer itkImage;
        reader->Update();
        itkImage = reader->GetOutput();
        cout << "Image loaded successfully" << endl;
        
        // 获取图像元数据
        ImageType::SpacingType spacing = itkImage->GetSpacing();
        ImageType::PointType origin = itkImage->GetOrigin();
        ImageType::DirectionType direction = itkImage->GetDirection();
        ImageType::RegionType region = itkImage->GetLargestPossibleRegion();
        ImageType::SizeType size = region.GetSize();
        
        cout << "Image size: " << size[0] << "x" << size[1] << "x" << size[2] << endl;
        cout << "Spacing: " << spacing[0] << "x" << spacing[1] << "x" << spacing[2] << " mm" << endl;
        
        // 将ITK图像数据复制到CImg
        CImg<short> inputCbctVolume(size[0], size[1], size[2], 1, 0);
        itk::ImageRegionIterator<ImageType> it(itkImage, region);
        short* cimg_data = inputCbctVolume.data();
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            *cimg_data++ = it.Get();
        }

        // 准备API数据结构
        float real_voxel_size[3] = {
            static_cast<float>(spacing[0]),
            static_cast<float>(spacing[1]),
            static_cast<float>(spacing[2])
        };
        
        AI_DataInfo *srcData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
        srcData->Width = size[0];
        srcData->Height = size[1];
        srcData->Depth = size[2];
        srcData->VoxelSpacing = 1.0f;
        srcData->Channels = inputCbctVolume.spectrum();
        srcData->VoxelSpacingX = real_voxel_size[0];
        srcData->VoxelSpacingY = real_voxel_size[1];
        srcData->VoxelSpacingZ = real_voxel_size[2];
        // 需要给OriginalVoxelSpacing单独赋值，否则可能会出现初始化异常问题
        srcData->OriginalVoxelSpacingX = real_voxel_size[0];
        srcData->OriginalVoxelSpacingY = real_voxel_size[1];
        srcData->OriginalVoxelSpacingZ = real_voxel_size[2];
        srcData->OriginX = origin[0];
        srcData->OriginY = origin[1];
        srcData->OriginZ = origin[2];
        srcData->ptr_Data = inputCbctVolume.data();

        //===== 初始化分割模型 =====
        cout << "\n======= Initializing Model =======" << endl;
        cout << "Step 1: Creating model object..." << endl;
        AI_HANDLE AI_Hdl = UnetSegAI_CreateObj();
        if (AI_Hdl == NULL) {
            cout << "ERROR: UnetSegAI_CreateObj() returned NULL!" << endl;
            cout << "This indicates model object creation failed." << endl;
            free(srcData);
            system("pause");
            return -1;
        }
        cout << "Model object created successfully." << endl;
        
        // 配置模型参数
        cout << "Step 2: Loading JSON configuration..." << endl;
        cout << "JSON content length: " << jsonContent.length() << " characters" << endl;
        AI_INT config_status = UnetSegAI_SetConfigFromJson(AI_Hdl, jsonContent.c_str());
        if (config_status != UnetSegAI_STATUS_SUCCESS) {
            cout << "ERROR: UnetSegAI_SetConfigFromJson() failed!" << endl;
            cout << "Status code: " << config_status << endl;
            cout << "JSON config path: " << configPath << endl;
            UnetSegAI_ReleseObj(AI_Hdl);
            free(srcData);
            system("pause");
            return -1;
        }
        cout << "JSON configuration loaded successfully." << endl;
        
        // 设置模型路径
        cout << "Step 3: Setting model path..." << endl;
        cout << "Model path: " << modelPath << endl;
        wstring wModelPath(modelPath.begin(), modelPath.end());
        AI_INT status1 = UnetSegAI_SetModelPath(AI_Hdl, const_cast<wchar_t*>(wModelPath.c_str()));
        if (status1 != UnetSegAI_STATUS_SUCCESS) {
            cout << "ERROR: UnetSegAI_SetModelPath() failed!" << endl;
            cout << "Status code: " << status1 << endl;
            cout << "Model path: " << modelPath << endl;
            
            // 检查模型文件是否存在
            if (!filesystem::exists(modelPath)) {
                cout << "Model file does not exist!" << endl;
            } else {
                cout << "Model file exists but failed to load." << endl;
            }
            
            UnetSegAI_ReleseObj(AI_Hdl);
            free(srcData);
            system("pause");
            return -1;
        }
        cout << "Model path set successfully." << endl;
        
        // 设置TileStepRatio
        cout << "Step 4: Setting tile step ratio..." << endl;
        UnetSegAI_SetTileStepRatio(AI_Hdl, 0.5f);
        cout << "Tile step ratio set to 0.5" << endl;
        
        // 设置中间结果输出路径
        cout << "Step 5: Setting output paths..." << endl;
        wstring preprocessPath = L"..\\..\\..\\result\\preprocess";
        wstring modelOutputPath = L"..\\..\\..\\result\\model_output";
        wstring postprocessPath = L"..\\..\\..\\result\\postprocess";
        
        UnetSegAI_SetOutputPaths(AI_Hdl, 
            const_cast<wchar_t*>(preprocessPath.c_str()),
            const_cast<wchar_t*>(modelOutputPath.c_str()),
            const_cast<wchar_t*>(postprocessPath.c_str()));
        cout << "Output paths set successfully." << endl;
        cout << "Model initialization completed!" << endl;
        
        //===== 执行分割推理 =====
        cout << "\n======= Running Inference =======" << endl;
        auto start = chrono::steady_clock::now();
        
        AI_INT AIWorkStatus = UnetSegAI_Infer(AI_Hdl, srcData);
        
        if (AIWorkStatus != UnetSegAI_STATUS_SUCCESS) {
            cout << "Model inference failed! Status: " << AIWorkStatus << endl;
            UnetSegAI_ReleseObj(AI_Hdl);
            free(srcData);
            return -1;
        }
        
        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Inference completed in " << elapsed.count() << "s" << endl;
        
        //===== 获取结果并保存 =====
        cout << "\n======= Saving Results =======" << endl;
        
        // 准备结果数据结构
        CImg<short> toothLabelMask = CImg<short>(size[0], size[1], size[2], 1, 0);
        AI_DataInfo *toothSegData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
        toothSegData->Width = size[0];
        toothSegData->Height = size[1];
        toothSegData->Depth = size[2];
        toothSegData->VoxelSpacingX = real_voxel_size[0];
        toothSegData->VoxelSpacingY = real_voxel_size[1];
        toothSegData->VoxelSpacingZ = real_voxel_size[2];
        toothSegData->ptr_Data = toothLabelMask.data();
        
        // 获取分割结果
        UnetSegAI_GetResult(AI_Hdl, toothSegData);
        
        // 使用ITK保存结果（NIfTI格式）
        string resultDir = "..\\\\..\\\\..\\\\result";
        _mkdir(resultDir.c_str());
        string resultPath = resultDir + "\\\\finalLabelMask.nii.gz";
        
        using WriterType = itk::ImageFileWriter<ImageType>;
        ImageType::Pointer outputImage = ImageType::New();
        outputImage->SetRegions(region);
        outputImage->SetSpacing(spacing);
        outputImage->SetOrigin(origin);
        outputImage->SetDirection(direction);
        outputImage->Allocate();
        
        // 复制分割结果到ITK图像
        itk::ImageRegionIterator<ImageType> outIt(outputImage, region);
        short* resultPtr = toothLabelMask.data();
        for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt) {
            outIt.Set(*resultPtr++);
        }
        
        // 写入文件
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(resultPath);
        writer->SetInput(outputImage);
        writer->Update();
        
        cout << "Results saved to: " << resultPath << endl;
        
        // 清理资源
        UnetSegAI_ReleseObj(AI_Hdl);
        free(srcData);
        free(toothSegData);
        
        cout << "\nProgram completed successfully!" << endl;
        system("pause");
        return 0;
        
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
        system("pause");
        return -1;
    }
}