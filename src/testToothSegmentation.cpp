// testToothSegmentation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <memory>
#include <chrono>
#include <windows.h>
#include <exception>
#include <csignal>
#include <typeinfo>
#include <clocale>
#include <cstdlib>
#include <direct.h>  // for _mkdir
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>


// onnx 相关
#include "..\header\DentalCbctSegAI_API.h"
#include "..\lib\onnxruntime\include\onnxruntime_cxx_api.h"
#pragma comment(lib, "..\\lib\\DentalCbctOnnxSegDLL.lib")

#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_shared.lib")
#pragma comment(lib,"..\\lib\\onnxruntime\\lib\\onnxruntime_providers_cuda.lib")
//#pragma comment(lib,"..\\utility\\onnxruntime\\lib\\onnxruntime_providers_tensorrt.lib")


/*
// libtorch 231 相关
#include "DentalCbctSegAI_API.h"
#pragma comment(lib, "..\\x64\\Release\\DentalCbctSegDLL.lib")

#pragma comment(lib,"..\\utility\\libtorch231\\lib\\asmjit.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\c10.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\c10_cuda.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\cpuinfo.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\fbgemm.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch_cpu.lib")
#pragma comment(lib,"..\\utility\\libtorch231\\lib\\torch_cuda.lib")
*/


//"/INCLUDE:?warp_size@cuda@at@@YAHXZ"

//113:
//"/INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:?_torch_cuda_cu_linker_symbol_op_cuda@native@at@@YA?AVTensor@2@AEBV32@@Z"

//230
//" /INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:"?ignore_this_library_placeholder@@YAHXZ" "

//CImg用于二进制存储输入输出
#define cimg_display_type 2
#include "..\lib\CImg\CImg.h"

//ITK用于读写医学图像并保留origin信息
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

using namespace std;
using namespace cimg_library;

// 简单的JSON配置结构
struct ModelConfig {
    vector<int> all_labels;
    int num_classes;
    int num_input_channels;
    vector<int> patch_size;
    vector<float> target_spacing;
    vector<int> transpose_forward;
    vector<int> transpose_backward;
    float mean;
    float std;
    float min_val;
    float max_val;
    float percentile_00_5;
    float percentile_99_5;
    string normalization_scheme;
    bool use_mask_for_norm;
    bool use_tta;
};

// 简单的JSON解析函数
bool parseJsonConfig(const string& jsonPath, ModelConfig& config) {
    ifstream file(jsonPath);
    if (!file.is_open()) {
        cout << "Error: Cannot open config file: " << jsonPath << endl;
        return false;
    }
    
    string line;
    string jsonContent;
    while (getline(file, line)) {
        jsonContent += line;
    }
    file.close();
    
    try {
        // 解析num_classes
        size_t pos = jsonContent.find("\"num_classes\":");
        if (pos != string::npos) {
            pos = jsonContent.find(":", pos) + 1;
            size_t end = jsonContent.find(",", pos);
            if (end == string::npos) end = jsonContent.find("}", pos);
            string value = jsonContent.substr(pos, end - pos);
            // 移除空格
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            config.num_classes = stoi(value);
        }
        
        // 解析num_input_channels
        pos = jsonContent.find("\"num_input_channels\":");
        if (pos != string::npos) {
            pos = jsonContent.find(":", pos) + 1;
            size_t end = jsonContent.find(",", pos);
            if (end == string::npos) end = jsonContent.find("}", pos);
            string value = jsonContent.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            config.num_input_channels = stoi(value);
        }
        
        // 解析patch_size数组
        pos = jsonContent.find("\"patch_size\":");
        if (pos != string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            // 解析数组元素
            stringstream ss(arrayStr);
            string item;
            config.patch_size.clear();
            while (getline(ss, item, ',')) {
                item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    config.patch_size.push_back(stoi(item));
                }
            }
        }
        
        // 解析target_spacing数组
        pos = jsonContent.find("\"target_spacing\":");
        if (pos != string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            stringstream ss(arrayStr);
            string item;
            config.target_spacing.clear();
            while (getline(ss, item, ',')) {
                item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    config.target_spacing.push_back(stof(item));
                }
            }
        }
        
        // 解析transpose_forward数组
        pos = jsonContent.find("\"transpose_forward\":");
        if (pos != string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            stringstream ss(arrayStr);
            string item;
            config.transpose_forward.clear();
            while (getline(ss, item, ',')) {
                item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    config.transpose_forward.push_back(stoi(item));
                }
            }
        }
        
        // 解析transpose_backward数组
        pos = jsonContent.find("\"transpose_backward\":");
        if (pos != string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            stringstream ss(arrayStr);
            string item;
            config.transpose_backward.clear();
            while (getline(ss, item, ',')) {
                item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                if (!item.empty()) {
                    config.transpose_backward.push_back(stoi(item));
                }
            }
        }
        
        // 解析intensity_properties中的第一个通道数据
        pos = jsonContent.find("\"intensity_properties\":");
        if (pos != string::npos) {
            pos = jsonContent.find("\"0\":", pos);
            if (pos != string::npos) {
                size_t objStart = jsonContent.find("{", pos);
                size_t objEnd = jsonContent.find("}", objStart);
                string objStr = jsonContent.substr(objStart, objEnd - objStart + 1);
                
                // 解析mean
                size_t meanPos = objStr.find("\"mean\":");
                if (meanPos != string::npos) {
                    meanPos = objStr.find(":", meanPos) + 1;
                    size_t end = objStr.find(",", meanPos);
                    if (end == string::npos) end = objStr.find("}", meanPos);
                    string value = objStr.substr(meanPos, end - meanPos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.mean = stof(value);
                }
                
                // 解析std
                size_t stdPos = objStr.find("\"std\":");
                if (stdPos != string::npos) {
                    stdPos = objStr.find(":", stdPos) + 1;
                    size_t end = objStr.find(",", stdPos);
                    if (end == string::npos) end = objStr.find("}", stdPos);
                    string value = objStr.substr(stdPos, end - stdPos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.std = stof(value);
                }
                
                // 解析min
                size_t minPos = objStr.find("\"min\":");
                if (minPos != string::npos) {
                    minPos = objStr.find(":", minPos) + 1;
                    size_t end = objStr.find(",", minPos);
                    if (end == string::npos) end = objStr.find("}", minPos);
                    string value = objStr.substr(minPos, end - minPos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.min_val = stof(value);
                }
                
                // 解析max
                size_t maxPos = objStr.find("\"max\":");
                if (maxPos != string::npos) {
                    maxPos = objStr.find(":", maxPos) + 1;
                    size_t end = objStr.find(",", maxPos);
                    if (end == string::npos) end = objStr.find("}", maxPos);
                    string value = objStr.substr(maxPos, end - maxPos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.max_val = stof(value);
                }
                
                // 解析percentile_00_5
                size_t p005Pos = objStr.find("\"percentile_00_5\":");
                if (p005Pos != string::npos) {
                    p005Pos = objStr.find(":", p005Pos) + 1;
                    size_t end = objStr.find(",", p005Pos);
                    if (end == string::npos) end = objStr.find("}", p005Pos);
                    string value = objStr.substr(p005Pos, end - p005Pos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.percentile_00_5 = stof(value);
                }
                
                // 解析percentile_99_5
                size_t p995Pos = objStr.find("\"percentile_99_5\":");
                if (p995Pos != string::npos) {
                    p995Pos = objStr.find(":", p995Pos) + 1;
                    size_t end = objStr.find(",", p995Pos);
                    if (end == string::npos) end = objStr.find("}", p995Pos);
                    string value = objStr.substr(p995Pos, end - p995Pos);
                    value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                    config.percentile_99_5 = stof(value);
                }
            }
        }
        
        // 解析normalization_schemes
        pos = jsonContent.find("\"normalization_schemes\":");
        if (pos != string::npos) {
            pos = jsonContent.find("[", pos);
            size_t end = jsonContent.find("]", pos);
            string arrayStr = jsonContent.substr(pos + 1, end - pos - 1);
            
            // 提取第一个scheme
            size_t quoteStart = arrayStr.find("\"");
            if (quoteStart != string::npos) {
                size_t quoteEnd = arrayStr.find("\"", quoteStart + 1);
                config.normalization_scheme = arrayStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
            }
        }
        
        // 解析use_tta
        pos = jsonContent.find("\"use_tta\":");
        if (pos != string::npos) {
            pos = jsonContent.find(":", pos) + 1;
            size_t end = jsonContent.find(",", pos);
            if (end == string::npos) end = jsonContent.find("}", pos);
            string value = jsonContent.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            config.use_tta = (value == "true");
        }
        
        cout << "Successfully parsed config file: " << jsonPath << endl;
        return true;
        
    } catch (const exception& e) {
        cout << "Error parsing JSON config: " << e.what() << endl;
        return false;
    }
}

// 函数：根据扩展名过滤并列出目录中的文件，让用户选择
string selectFileFromDirectory(const string& directory, const string& fileType, const vector<string>& allowedExtensions) {
    vector<string> files;
    
    try {
        // 检查目录是否存在
        if (!filesystem::exists(directory)) {
            cout << "Error: Directory " << directory << " does not exist!" << endl;
            return "";
        }
        
        // 遍历目录获取文件，并根据扩展名过滤
        for (const auto& entry : filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                string filename = entry.path().filename().string();
                string extension = entry.path().extension().string();
                
                // 转换为小写进行比较
                transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                // 检查文件扩展名是否在允许的列表中
                bool isAllowed = false;
                for (const auto& allowedExt : allowedExtensions) {
                    string lowerAllowedExt = allowedExt;
                    transform(lowerAllowedExt.begin(), lowerAllowedExt.end(), lowerAllowedExt.begin(), ::tolower);
                    if (extension == lowerAllowedExt) {
                        isAllowed = true;
                        break;
                    }
                }
                
                // 特殊处理 .nii.gz 文件
                if (!isAllowed && filename.length() > 7) {
                    string lastPart = filename.substr(filename.length() - 7);
                    transform(lastPart.begin(), lastPart.end(), lastPart.begin(), ::tolower);
                    for (const auto& allowedExt : allowedExtensions) {
                        string lowerAllowedExt = allowedExt;
                        transform(lowerAllowedExt.begin(), lowerAllowedExt.end(), lowerAllowedExt.begin(), ::tolower);
                        if (lastPart == lowerAllowedExt) {
                            isAllowed = true;
                            break;
                        }
                    }
                }
                
                if (isAllowed) {
                    files.push_back(filename);
                }
            }
        }
        
        if (files.empty()) {
            cout << "Error: No files found in directory " << directory << "!" << endl;
            return "";
        }
        
        // 显示文件列表
        cout << "\nFound " << fileType << " files in directory " << directory << ":" << endl;
        for (size_t i = 0; i < files.size(); ++i) {
            cout << "[" << (i + 1) << "] " << files[i] << endl;
        }
        
        // 用户选择
        int choice;
        cout << "\nPlease select file number (1-" << files.size() << "): ";
        cin >> choice;
        
        if (choice < 1 || choice > static_cast<int>(files.size())) {
            cout << "Error: Invalid selection!" << endl;
            return "";
        }
        
        string selectedFile = directory + "\\\\" + files[choice - 1];
        cout << "Selected file: " << selectedFile << endl;
        return selectedFile;
        
    } catch (const filesystem::filesystem_error& e) {
        cout << "Filesystem error: " << e.what() << endl;
        return "";
    }
}

// 信号处理函数
void SignalHandler(int signal) {
    std::cerr << "\n===== Program Exception =====" << std::endl;
    std::cerr << "Signal: " << signal << std::endl;
    switch(signal) {
        case SIGSEGV:
            std::cerr << "Segmentation fault (SIGSEGV)" << std::endl;
            break;
        case SIGFPE:
            std::cerr << "Floating point exception (SIGFPE)" << std::endl;
            break;
        case SIGILL:
            std::cerr << "Illegal instruction (SIGILL)" << std::endl;
            break;
        case SIGABRT:
            std::cerr << "Program abort (SIGABRT)" << std::endl;
            break;
    }
    system("pause");
    exit(1);
}

// SEH异常处理器
LONG WINAPI MyUnhandledExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    std::cerr << "\n===== Unhandled SEH Exception =====" << std::endl;
    std::cerr << "Exception Code: 0x" << std::hex << pExceptionInfo->ExceptionRecord->ExceptionCode << std::dec << std::endl;
    
    switch(pExceptionInfo->ExceptionRecord->ExceptionCode) {
        case EXCEPTION_ACCESS_VIOLATION:
            std::cerr << "Access Violation" << std::endl;
            break;
        case EXCEPTION_STACK_OVERFLOW:
            std::cerr << "Stack Overflow" << std::endl;
            break;
        case 0xE06D7363:
            std::cerr << "C++ Exception (0xE06D7363)" << std::endl;
            break;
    }
    
    system("pause");
    return EXCEPTION_EXECUTE_HANDLER;
}

// 包装的函数，用于执行可能抛出SEH异常的代码
int SafeInfer(AI_HANDLE AI_Hdl, AI_DataInfo* srcData, AI_INT& AIWorkStatus) {
    __try {
        AIWorkStatus = DentalCbctSegAI_Infer(AI_Hdl, srcData);
        return 0; // 成功
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        DWORD exceptionCode = GetExceptionCode();
        std::cerr << "\n===== SEH Exception in DentalCbctSegAI_Infer =====" << std::endl;
        std::cerr << "Exception Code: 0x" << std::hex << exceptionCode << std::dec << std::endl;
        
        if (exceptionCode == 0xE06D7363) {
            std::cerr << "DLL内部抛出了C++异常" << std::endl;
            // 实际的错误信息应该在DLL内部的日志中
        }
        
        return -1; // 失败
    }
}

int main()
{	
	// 设置locale以支持中文
	//setlocale(LC_ALL, "chs");
	
	
	// 安装信号处理器
	signal(SIGSEGV, SignalHandler);
	signal(SIGFPE, SignalHandler);
	signal(SIGILL, SignalHandler);
	signal(SIGABRT, SignalHandler);
	
	// 设置SEH异常处理器
	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
	
	try {
		std::cout << "程序开始运行..." << std::endl;
		
		//===== 用户选择配置文件 =====
		cout << "======= Config File Selection =======" << endl;
		vector<string> configExtensions = {".json"};
		string configPath = selectFileFromDirectory("..\\..\\..\\param", "config", configExtensions);
		if (configPath.empty()) {
			cout << "Error: Failed to select config file, program exit!" << endl;
			system("pause");
			return -1;
		}
		
		//===== 用户选择模型文件 =====
		cout << "\n======= Model File Selection =======" << endl;
		vector<string> modelExtensions = {".onnx"};
		string modelPath = selectFileFromDirectory("..\\..\\..\\model", "model", modelExtensions);
		if (modelPath.empty()) {
			cout << "Error: Failed to select model file, program exit!" << endl;
			system("pause");
			return -1;
		}
		
		//===== 用户选择输入数据文件 =====
		cout << "\n======= Input Data File Selection =======" << endl;
		//load raw volume data: 左右左前右后；头脚右左，上下头脚，头顶脚底
		vector<string> inputExtensions = {".hdr", ".nii", ".nii.gz"};
		std::string inputHdrPath = selectFileFromDirectory("..\\..\\..\\img", "input data", inputExtensions);
		if (inputHdrPath.empty()) {
			cout << "Error: Failed to select input data file, program exit!" << endl;
			system("pause");
			return -1;
		}
		
		//===== 解析JSON配置文件 =====
		cout << "\n======= Parsing Configuration =======" << endl;
		ModelConfig config;
		if (!parseJsonConfig(configPath, config)) {
			cout << "Error: Failed to parse config file, program exit!" << endl;
			system("pause");
			return -1;
		}
		
		// 显示解析的配置信息
		cout << "Configuration loaded successfully:" << endl;
		cout << "  - num_classes: " << config.num_classes << endl;
		cout << "  - num_input_channels: " << config.num_input_channels << endl;
		cout << "  - patch_size: [" << config.patch_size[0] << ", " << config.patch_size[1] << ", " << config.patch_size[2] << "]" << endl;
		cout << "  - target_spacing: [" << config.target_spacing[0] << ", " << config.target_spacing[1] << ", " << config.target_spacing[2] << "]" << endl;
		cout << "  - normalization_scheme: " << config.normalization_scheme << endl;
		cout << "  - intensity mean: " << config.mean << ", std: " << config.std << endl;
		std::cout << "正在使用ITK加载HDR图像文件..." << std::endl;
		std::cout << "文件路径: " << inputHdrPath << std::endl;
		
		// ITK图像类型定义
		using PixelType = short;
		using ImageType = itk::Image<PixelType, 3>;
		using ReaderType = itk::ImageFileReader<ImageType>;
		
		// 使用ITK读取图像
		ReaderType::Pointer reader = ReaderType::New();
		reader->SetFileName(inputHdrPath);
		
		ImageType::Pointer itkImage;
		try {
			reader->Update();
			itkImage = reader->GetOutput();
			std::cout << "ITK图像加载成功" << std::endl;
		} catch (itk::ExceptionObject& e) {
			std::cerr << "ITK读取图像失败: " << e << std::endl;
			return -1;
		}
		
		// 获取图像元数据
		ImageType::SpacingType spacing = itkImage->GetSpacing();
		ImageType::PointType origin = itkImage->GetOrigin();
		ImageType::DirectionType direction = itkImage->GetDirection();
		ImageType::RegionType region = itkImage->GetLargestPossibleRegion();
		ImageType::SizeType size = region.GetSize();
		
		// 从HDR文件读取真实的voxel spacing
		float real_voxel_size[3] = {
			static_cast<float>(spacing[0]),
			static_cast<float>(spacing[1]),
			static_cast<float>(spacing[2])
		};
		
		std::cout << "原始图像真实spacing: X=" << real_voxel_size[0] 
		         << ", Y=" << real_voxel_size[1] 
		         << ", Z=" << real_voxel_size[2] << " mm" << std::endl;
		std::cout << "原始图像origin: X=" << origin[0]
		         << ", Y=" << origin[1]
		         << ", Z=" << origin[2] << " mm" << std::endl;
		
		// 将ITK图像数据复制到CImg
		CImg<short> inputCbctVolume(size[0], size[1], size[2], 1, 0);
		itk::ImageRegionIterator<ImageType> it(itkImage, region);
		short* cimg_data = inputCbctVolume.data();
		for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
			*cimg_data++ = it.Get();
		}

	float VoxelSpacing  = 1.0f; //unit: mm  0.3
	float VoxelSpacingX = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingY = 0.5810545086860657f; //unit: mm  0.3
	float VoxelSpacingZ = 1.0f; //unit: mm  0.3
	
	int Width0 = inputCbctVolume.width();
	int Height0 = inputCbctVolume.height();
	int Depth0 = inputCbctVolume.depth();

	//查看slice的位置显示图像是否正确放置
	//根据数据方位属性，可通过rotate XY 90、180、-90度进行调整
	//inputCbctVolume.rotate(180); //90, -90, 180
	//inputCbctVolume -= 1024;

	//CImg<short> slice_z = inputCbctVolume.get_slice( Depth0 / 2);
	//slice_z.display("slice 120");
	//关闭显示窗口后，程序继续运行

	//创建输入数据结构信息
	short* ptrCbctData = inputCbctVolume.data();
	AI_DataInfo *srcData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	srcData->Width = Width0;
	srcData->Height = Height0;
	srcData->Depth = Depth0;
	srcData->VoxelSpacing = VoxelSpacing;
	srcData->VoxelSpacingX = VoxelSpacingX;
	srcData->VoxelSpacingY = VoxelSpacingY;
	srcData->VoxelSpacingZ = VoxelSpacingZ;
	// 设置原始spacing（从HDR文件读取的真实值）
	srcData->OriginalVoxelSpacingX = real_voxel_size[0];
	srcData->OriginalVoxelSpacingY = real_voxel_size[1];
	srcData->OriginalVoxelSpacingZ = real_voxel_size[2];
	// 设置origin信息
	srcData->OriginX = origin[0];
	srcData->OriginY = origin[1];
	srcData->OriginZ = origin[2];
	srcData->ptr_Data = ptrCbctData; //CBCT数据块指针

	//初始化牙齿分割输出数据信息
	CImg<short> toothLabelMask = CImg<short>(Width0, Height0, Depth0, 1, 0);

	AI_DataInfo *toothSegData = (AI_DataInfo*)malloc(sizeof(AI_DataInfo));
	toothSegData->Width = Width0;
	toothSegData->Height = Height0;
	toothSegData->Depth = Depth0;
	toothSegData->VoxelSpacing = VoxelSpacing;
	toothSegData->VoxelSpacingX = VoxelSpacingX;
	toothSegData->VoxelSpacingY = VoxelSpacingY;
	toothSegData->VoxelSpacingZ = VoxelSpacingZ;
	toothSegData->ptr_Data = toothLabelMask.data();//分割label数据块指针


	auto start = std::chrono::steady_clock::now();

	//调用牙齿分割模型
	//初始化分割模型对象
	std::cout << "\nInitializing segmentation model..." << std::endl;
	AI_HANDLE  AI_Hdl = DentalCbctSegAI_CreateObj();
	if (AI_Hdl == NULL) {
		std::cerr << "Error: Model initialization failed!" << std::endl;
		return DentalCbctSegAI_STATUS_HANDLE_NULL; //模型初始化失败
	}
	std::cout << "Model initialized successfully" << std::endl;

	//===== 配置模型参数 =====
	cout << "\n======= Configuring Model Parameters =======" << endl;
	
	// 设置patch size
	AI_INT status_patch = DentalCbctSegAI_SetPatchSize(AI_Hdl, config.patch_size[0], config.patch_size[1], config.patch_size[2]);
	cout << "SetPatchSize(" << config.patch_size[0] << ", " << config.patch_size[1] << ", " << config.patch_size[2] << ") status: " << status_patch << endl;
	
	// 设置类别数
	AI_INT status_classes = DentalCbctSegAI_SetNumClasses(AI_Hdl, config.num_classes);
	cout << "SetNumClasses(" << config.num_classes << ") status: " << status_classes << endl;
	
	// 设置输入通道数
	AI_INT status_channels = DentalCbctSegAI_SetInputChannels(AI_Hdl, config.num_input_channels);
	cout << "SetInputChannels(" << config.num_input_channels << ") status: " << status_channels << endl;
	
	// 设置目标spacing
	AI_INT status_spacing = DentalCbctSegAI_SetTargetSpacing(AI_Hdl, config.target_spacing[0], config.target_spacing[1], config.target_spacing[2]);
	cout << "SetTargetSpacing(" << config.target_spacing[0] << ", " << config.target_spacing[1] << ", " << config.target_spacing[2] << ") status: " << status_spacing << endl;
	
	// 设置transpose设置
	AI_INT status_transpose = DentalCbctSegAI_SetTransposeSettings(AI_Hdl, 
		config.transpose_forward[0], config.transpose_forward[1], config.transpose_forward[2],
		config.transpose_backward[0], config.transpose_backward[1], config.transpose_backward[2]);
	cout << "SetTransposeSettings status: " << status_transpose << endl;
	
	// 设置归一化类型
	AI_INT status_norm = DentalCbctSegAI_SetNormalizationType(AI_Hdl, config.normalization_scheme.c_str());
	cout << "SetNormalizationType(" << config.normalization_scheme << ") status: " << status_norm << endl;
	
	// 设置强度属性
	AI_INT status_intensity = DentalCbctSegAI_SetIntensityProperties(AI_Hdl, 
		config.mean, config.std, config.min_val, config.max_val, config.percentile_00_5, config.percentile_99_5);
	cout << "SetIntensityProperties status: " << status_intensity << endl;
	
	// 设置mirroring
	AI_INT status_mirror = DentalCbctSegAI_SetUseMirroring(AI_Hdl, false); // 默认false
	cout << "SetUseMirroring(false) status: " << status_mirror << endl;

	std::cout << "\nSetting model path..." << std::endl;
	std::cout << "模型文件: " << modelPath << std::endl;
	// 转换为宽字符串
	std::wstring wModelPath(modelPath.begin(), modelPath.end());
	AI_INT status1 = DentalCbctSegAI_SetModelPath(AI_Hdl, const_cast<wchar_t*>(wModelPath.c_str()));
	std::cout << "SetModelPath返回状态: " << status1 << std::endl;

	// 设置TileStepRatio
	float tileRatio = 0.5f;
	AI_INT status2 = DentalCbctSegAI_SetTileStepRatio(AI_Hdl, tileRatio);
	std::cout << "SetTileStepRatio(" << tileRatio << ")返回状态: " << status2 << std::endl;

	// 设置输出路径以保存三个阶段的中间结果
	std::wstring preprocessPath = L"..\\..\\..\\result\\preprocess";
	std::wstring modelOutputPath = L"..\\..\\..\\result\\model_output";
	std::wstring postprocessPath = L"..\\..\\..\\result\\postprocess";
	
	std::cout << "\n设置中间结果输出路径..." << std::endl;
	AI_INT status3 = DentalCbctSegAI_SetOutputPaths(AI_Hdl, 
	                                                const_cast<wchar_t*>(preprocessPath.c_str()),
	                                                const_cast<wchar_t*>(modelOutputPath.c_str()),
	                                                const_cast<wchar_t*>(postprocessPath.c_str()));
	std::cout << "SetOutputPaths返回状态: " << status3 << std::endl;

	// 打印输入数据信息
	std::cout << "\n输入数据信息:" << std::endl;
	std::cout << "  尺寸: " << srcData->Width << " x " << srcData->Height << " x " << srcData->Depth << std::endl;
	std::cout << "  体素间距: X=" << srcData->VoxelSpacingX << ", Y=" << srcData->VoxelSpacingY << ", Z=" << srcData->VoxelSpacingZ << std::endl;
	std::cout << "  数据指针: " << (void*)srcData->ptr_Data << std::endl;

	// 调用模型推理，可能抛出 ONNX Runtime 异常
	std::cout << "\n开始模型推理..." << std::endl;
	
	AI_INT	AIWorkStatus = DentalCbctSegAI_STATUS_FAIED;
	int result = SafeInfer(AI_Hdl, srcData, AIWorkStatus);
	
	if (result != 0) {
		// SEH异常处理
		// 释放资源
		if (AI_Hdl) DentalCbctSegAI_ReleseObj(AI_Hdl);
		if (srcData) free(srcData);
		if (toothSegData) free(toothSegData);
		
		system("pause");
		return -7001;
	}
	
	std::cout << "模型推理完成，状态码: " << AIWorkStatus << std::endl;
	
	// 检查推理状态
	if (AIWorkStatus != DentalCbctSegAI_STATUS_SUCCESS) {
		std::cerr << "ERROR: Model inference failed with status: " << AIWorkStatus << std::endl;
		// 释放资源
		if (AI_Hdl) DentalCbctSegAI_ReleseObj(AI_Hdl);
		if (srcData) free(srcData);
		if (toothSegData) free(toothSegData);
		system("pause");
		return AIWorkStatus;
	}

	//获取牙齿分割结果
	std::cout << "Getting segmentation results..." << std::endl;
	DentalCbctSegAI_GetResult(AI_Hdl, toothSegData);

	//释放对象
	DentalCbctSegAI_ReleseObj(AI_Hdl);
	free(srcData);
	free(toothSegData);
	// 牙齿分割流程结束

	//关于输出toothSegData说明：
	//针对小视野CBCT：
    //totalToothNumber:分割出的牙齿数量
    //牙齿编号k=1,2,3,...,totalToothNumber，牙髓labelΪ3k，牙本质labelΪ3k+1，牙冠或金属部件labelΪ3k+2
    //针对大视野CBCT：
    //upperToothNumber:上颌牙数
    //lower_tooth_number:下颌牙数
    //上颌牙编号k=1,2,3,...,upperToothNumber，牙髓labelΪ3k，牙本质labelΪ3k+1，牙冠或金属部件labelΪ3k+2
    //下颌牙编号k=-1,-2,-3,...,-lowerToothNumber，牙髓labelΪ-3k，牙本质labelΪ-3k-1，牙冠或金属部件labelΪ-3k-2
	

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	//CImg<short> mask_z = toothLabelMask.get_slice(Depth0 / 2);
	//mask_z.display("mask 205");


	//保存指向结果目录
	// 确保结果目录存在
	std::string resultDir = "..\\..\\..\\result";
	_mkdir(resultDir.c_str()); // 如果目录已存在会失败，但不影响
	
	std::string resultPath = resultDir + "\\finalLabelMask.hdr";
	
	// 使用ITK保存结果，保留origin信息
	std::cout << "\n使用ITK保存分割结果..." << std::endl;
	
	// 创建输出ITK图像
	using WriterType = itk::ImageFileWriter<ImageType>;
	ImageType::Pointer outputImage = ImageType::New();
	
	// 设置图像属性
	outputImage->SetRegions(region);
	outputImage->SetSpacing(spacing);
	outputImage->SetOrigin(origin);  // 使用输入图像的origin
	outputImage->SetDirection(direction);  // 使用输入图像的direction
	outputImage->Allocate();
	
	// 如果toothSegData中有更新的origin信息，使用它
	if (toothSegData->OriginX != 0 || toothSegData->OriginY != 0 || toothSegData->OriginZ != 0) {
		ImageType::PointType outputOrigin;
		outputOrigin[0] = toothSegData->OriginX;
		outputOrigin[1] = toothSegData->OriginY;
		outputOrigin[2] = toothSegData->OriginZ;
		outputImage->SetOrigin(outputOrigin);
		std::cout << "使用分割结果的origin: X=" << outputOrigin[0]
		         << ", Y=" << outputOrigin[1]
		         << ", Z=" << outputOrigin[2] << " mm" << std::endl;
	} else {
		std::cout << "使用原始图像的origin: X=" << origin[0]
		         << ", Y=" << origin[1]
		         << ", Z=" << origin[2] << " mm" << std::endl;
	}
	
	// 复制分割结果到ITK图像
	itk::ImageRegionIterator<ImageType> outIt(outputImage, region);
	short* resultPtr = toothLabelMask.data();
	for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt) {
		outIt.Set(*resultPtr++);
	}
	
	// 使用ITK写入文件
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(resultPath);
	writer->SetInput(outputImage);
	
	try {
		writer->Update();
		std::cout << "分割结果保存成功: " << resultPath << std::endl;
	} catch (itk::ExceptionObject& e) {
		std::cerr << "ITK保存图像失败: " << e << std::endl;
		return -1;
	}

		std::cout << "\n程序执行成功完成!" << std::endl;
		
		// 打印输入输出信息汇总
		std::cout << "\n========== 执行信息汇总 ==========" << std::endl;
		std::cout << "输入数据:" << std::endl;
		std::cout << "  - HDR文件: " << inputHdrPath << std::endl;
		std::cout << "  - 绝对路径: D:\\Project\\nnuNet_cpp\\img\\Series_5_Acq_2_0000.hdr" << std::endl;
		std::cout << "  - 数据尺寸: " << Width0 << " x " << Height0 << " x " << Depth0 << std::endl;
		std::cout << "  - 体素间距: X=" << VoxelSpacingX << ", Y=" << VoxelSpacingY << ", Z=" << VoxelSpacingZ << " mm" << std::endl;
		
		std::cout << "\n使用模型:" << std::endl;
		std::cout << "  - 模型文件: " << modelPath << std::endl;
		std::cout << "  - 绝对路径: D:\\Project\\nnuNet_cpp\\model\\kneeseg_test.onnx" << std::endl;
		std::cout << "  - TileStepRatio: " << tileRatio << std::endl;
		
		std::cout << "\n输出结果:" << std::endl;
		std::cout << "  - 分割结果: " << resultPath << std::endl;
		std::cout << "  - 绝对路径: D:\\Project\\nnuNet_cpp\\result\\finalLabelMask.hdr" << std::endl;
		std::cout << "  - 注意: HDR格式会同时生成.hdr和.img两个文件" << std::endl;
		
		std::cout << "\n中间结果保存路径:" << std::endl;
		std::cout << "  - 预处理结果: ..\\..\\..\\result\\preprocess" << std::endl;
		std::cout << "  - 模型输出: ..\\..\\..\\result\\model_output" << std::endl;
		std::cout << "  - 后处理结果: ..\\..\\..\\result\\postprocess" << std::endl;
		
		// 获取当前时间
		auto now = std::chrono::system_clock::now();
		auto time_t = std::chrono::system_clock::to_time_t(now);
		struct tm timeinfo;
		localtime_s(&timeinfo, &time_t);
		char timeStr[100];
		strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
		std::cout << "  - 生成时间: " << timeStr << std::endl;
		std::cout << "=================================" << std::endl;
		system("pause");
		return AIWorkStatus;
		
	} catch (const CImgIOException& e) {
		std::cerr << "\n===== CImg IO异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -2001;
	} catch (const CImgException& e) {
		std::cerr << "\n===== CImg异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -2002;
	} catch (const Ort::Exception& e) {
		std::cerr << "\n===== ONNX Runtime异常 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -3001;
	} catch (const std::bad_alloc& e) {
		std::cerr << "\n===== 内存分配失败 =====" << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -4001;
	} catch (const std::exception& e) {
		std::cerr << "\n===== 标准异常 =====" << std::endl;
		std::cerr << "异常类型: " << typeid(e).name() << std::endl;
		std::cerr << "错误信息: " << e.what() << std::endl;
		system("pause");
		return -5001;
	} catch (...) {
		std::cerr << "\n===== Unknown Exception =====" << std::endl;
		std::cerr << "Caught unknown exception type" << std::endl;
		system("pause");
		return -6001;
	}
}

