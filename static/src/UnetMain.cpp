#include "UnetMain.h"
#include "UnetInference.h"
#include "UnetTorchInference.h"
#include "UnetPostprocessor.h"
#include "UnetPreprocessor.h"
#include "UnetIO.h"
#include "ConfigParser.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <tuple>
#include <chrono>
#include <cmath>
#include <windows.h>

UnetMain::UnetMain()
{
	NETDEBUG_FLAG = true;
	session_initialized = false;

	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnUNetInference");
	std::vector<std::string> providers = Ort::GetAvailableProviders();
	use_gpu = true;

	for (const auto& provider : providers) {
		if (provider == "CUDAExecutionProvider") {
			use_gpu = true;
		}
	}
	//use_gpu = false;


	// 模型路径应由外部设置通过 setModelFns 函数
	unetConfig.model_file_name = nullptr;  // 初始化为空，等待外部设置
	
	// 初始化默认值（后续可通过setXXX函数调整）
	unetConfig.input_channels = 1;
	unetConfig.num_classes = 3;
	unetConfig.mandible_label = 1;
	unetConfig.maxilla_label = 2;
	unetConfig.sinus_label = 3;
	unetConfig.cimg_transpose_forward  = "xyz";
	unetConfig.cimg_transpose_backward = "xyz";
	unetConfig.transpose_forward  = { 0, 1, 2 };
	unetConfig.transpose_backward = { 0, 1, 2 };
	unetConfig.use_mirroring = false;
	
	// 初始化intensity properties的默认值
	unetConfig.mean = 0.0f;
	unetConfig.std = 1.0f;
	unetConfig.use_mask_for_norm = false;  // 默认不使用mask
	
	// 下面是需要从外部配置的参数
	// voxel_spacing, patch_size, step_size_ratio, normalization_type, intensity properties
	// 这些参数需通过 JSON 配置后再设定
	
	// 初始化输出路径
	saveIntermediateResults = false;
	
	// 初始化spacing向量，避免未初始化的访问
	input_voxel_spacing = { 1.0f, 1.0f, 1.0f };
	original_voxel_spacing = { 1.0f, 1.0f, 1.0f };
	transposed_input_voxel_spacing = { 1.0f, 1.0f, 1.0f };
	transposed_original_voxel_spacing = { 1.0f, 1.0f, 1.0f };
}


UnetMain::~UnetMain()
{
}


UnetMain *UnetMain::CreateUnetMain()
{
	UnetMain *segUnetModel = new UnetMain();


	return segUnetModel;
}

void  UnetMain::setModelFns(const wchar_t* model_fn)
{
	
	if (model_fn == nullptr) {
		return;
	}
	
	// 打印模型路径用于调试
	std::wcout << L"Model path: " << model_fn << std::endl;
	
	unetConfig.model_file_name = model_fn;
	
	// 检测模型类型
	model_backend = detectModelBackend(model_fn);
	
	// 根据模型类型初始化
	if (model_backend == ModelBackend::ONNX) {
		std::cout << "Detected ONNX model, initializing ONNX Runtime..." << std::endl;
		initializeSession();
	} else if (model_backend == ModelBackend::TORCH) {
		std::cout << "Detected TorchScript model, initializing LibTorch..." << std::endl;
		initializeTorchModel();
	} else {
		std::cerr << "Error: Unknown model format. Supported formats: .onnx, .pt, .pth" << std::endl;
	}
}

void  UnetMain::setStepSizeRatio(float ratio)
{
	if (ratio <= 1.f && ratio >= 0.f)
	{
		unetConfig.step_size_ratio = ratio;
	}
	else
	{
		unetConfig.step_size_ratio = 0.5f;
	}
}

void UnetMain::setPatchSize(int64_t x, int64_t y, int64_t z)
{
	unetConfig.patch_size = { x, y, z };
}

void UnetMain::setNumClasses(int classes)
{
	unetConfig.num_classes = classes;
}

void UnetMain::setInputChannels(int channels)
{
	unetConfig.input_channels = channels;
}

void UnetMain::setTargetSpacing(float x, float y, float z)
{
	unetConfig.voxel_spacing = { x, y, z };
}

void UnetMain::setTransposeSettings(int forward_x, int forward_y, int forward_z, 
                                    int backward_x, int backward_y, int backward_z)
{
	unetConfig.transpose_forward = { forward_x, forward_y, forward_z };
	unetConfig.transpose_backward = { backward_x, backward_y, backward_z };
}

void UnetMain::setNormalizationType(const char* type)
{
	unetConfig.normalization_type = type;
}

void UnetMain::setIntensityProperties(float mean, float std, float min_val, float max_val,
                                      float percentile_00_5, float percentile_99_5)
{
	unetConfig.mean_std_HU = { mean, std };
	unetConfig.min_max_HU = { min_val, max_val };
	unetConfig.percentile_00_5 = static_cast<double>(percentile_00_5);
	unetConfig.percentile_99_5 = static_cast<double>(percentile_99_5);
	unetConfig.mean = static_cast<double>(mean);
	unetConfig.std = static_cast<double>(std);
}

void UnetMain::setUseMirroring(bool use_mirroring)
{
	unetConfig.use_mirroring = use_mirroring;
}

bool UnetMain::setConfigFromJsonString(const char* jsonContent)
{
	if (jsonContent == nullptr) {
		return false;
	}
	// 使用ConfigParser解析json配置
	ModelConfig config;
	if (configParser.parseJsonString(std::string(jsonContent), config)) {
		// 使用ConfigParser的静态方法应用配置
		ConfigParser::applyConfigToUnet(config, unetConfig);
		
		// 处理cimg_transpose字符串（这些需要持久存储）
		std::string forward_str, backward_str;
		for (size_t i = 0; i < config.transpose_forward.size(); i++) {
			if (config.transpose_forward[i] == 0) forward_str += 'x';
			else if (config.transpose_forward[i] == 1) forward_str += 'y';
			else if (config.transpose_forward[i] == 2) forward_str += 'z';
		}
		for (size_t i = 0; i < config.transpose_backward.size(); i++) {
			if (config.transpose_backward[i] == 0) backward_str += 'x';
			else if (config.transpose_backward[i] == 1) backward_str += 'y';
			else if (config.transpose_backward[i] == 2) backward_str += 'z';
		}
		
		// 保存转置字符串到成员变量中（需要添加成员变量）
		transposeForwardStr = forward_str;
		transposeBackwardStr = backward_str;
		unetConfig.cimg_transpose_forward = transposeForwardStr.c_str();
		unetConfig.cimg_transpose_backward = transposeBackwardStr.c_str();
		
		// 设置额外的向量字段（这些在applyConfigToUnet中已经处理了，但需要确认）
		unetConfig.mean_std_HU.clear();
		unetConfig.mean_std_HU.push_back(config.mean);
		unetConfig.mean_std_HU.push_back(config.std);
		
		unetConfig.min_max_HU.clear();
		unetConfig.min_max_HU.push_back(config.min_val);
		unetConfig.min_max_HU.push_back(config.max_val);
		
		return true;
	}
	
	return false;
}

void UnetMain::setOutputPaths(const wchar_t* preprocessPath, const wchar_t* modelOutputPath, const wchar_t* postprocessPath)
{
	if (preprocessPath != nullptr) {
		preprocessOutputPath = preprocessPath;
		// 如果目录不存在则创建
		std::filesystem::create_directories(preprocessPath);
	}
	
	if (modelOutputPath != nullptr) {
		this->modelOutputPath = modelOutputPath;
		// 如果目录不存在则创建
		std::filesystem::create_directories(modelOutputPath);
	}
	
	if (postprocessPath != nullptr) {
		postprocessOutputPath = postprocessPath;
		// 如果目录不存在则创建
		std::filesystem::create_directories(postprocessPath);
	}
	
	// 如果设置了任何路径则启用保存
	saveIntermediateResults = (preprocessPath != nullptr || modelOutputPath != nullptr || postprocessPath != nullptr);
}

AI_INT  UnetMain::setOnnxruntimeInstances()
{
	
	if (use_gpu) {
		try {
			//OrtCUDAProviderOptions cuda_options;
			//cuda_options.device_id = 0;  // 设置 GPU 设备 ID
			//session_options.AppendExecutionProvider_CUDA(cuda_options);

			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
		} catch (const Ort::Exception& e) {
			use_gpu = false;
		}
	} else {
	}
	
	// 设置线程数
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);

	// 创建会话
	//semantic_seg_session_ptr = std::make_unique<Ort::Session>(env, unetConfig.model_file_name.c_str(), session_options);

	return UnetSegAI_STATUS_SUCCESS;
}

AI_INT UnetMain::initializeSession()
{
	// 如果已经初始化，先释放旧的Session
	if (session_initialized) {
		semantic_seg_session_ptr.reset();
		session_initialized = false;
	}
	
	// 检查模型文件路径
	if (unetConfig.model_file_name == nullptr) {
		std::cerr << "Error: Model file path not set" << std::endl;
		return UnetSegAI_LOADING_FAIED;
	}
	
	// 配置Session Options
	if (use_gpu) {
		try {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
			std::cout << "Using CUDA execution provider" << std::endl;
		} catch (const Ort::Exception& e) {
			std::cout << "CUDA not available, falling back to CPU" << std::endl;
			use_gpu = false;
		}
	}
	
	// 设置线程数
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	
	try {
		// 创建Session
		semantic_seg_session_ptr = std::make_unique<Ort::Session>(env, unetConfig.model_file_name, session_options);
		
		// 获取并缓存输入输出名称
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::AllocatedStringPtr input_name_ptr = semantic_seg_session_ptr->GetInputNameAllocated(0, allocator);
		Ort::AllocatedStringPtr output_name_ptr = semantic_seg_session_ptr->GetOutputNameAllocated(0, allocator);
		
		cached_input_name = std::string(input_name_ptr.get());
		cached_output_name = std::string(output_name_ptr.get());
		
		// 验证模型输入形状
		auto input_shape = semantic_seg_session_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		if (input_shape.size() != 5) {
			std::cerr << "Error: Expected 5D input tensor, got " << input_shape.size() << "D" << std::endl;
			semantic_seg_session_ptr.reset();
			return UnetSegAI_LOADING_FAIED;
		}
		
		std::cout << "Session initialized successfully" << std::endl;
		std::cout << "Input name: " << cached_input_name << std::endl;
		std::cout << "Output name: " << cached_output_name << std::endl;
		
		session_initialized = true;
		return UnetSegAI_STATUS_SUCCESS;
		
	} catch (const Ort::Exception& e) {
		std::cerr << "Failed to initialize ONNX Runtime session: " << e.what() << std::endl;
		return UnetSegAI_LOADING_FAIED;
	}
}

AI_INT  UnetMain::setInput(AI_DataInfo *srcData)
{
	
	// 验证输入
	if (srcData == nullptr) {
		return UnetSegAI_STATUS_FAIED;
	}
	
	if (srcData->ptr_Data == nullptr) {
		return UnetSegAI_STATUS_FAIED;
	}
	
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //单位: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //单位: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //单位: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //单位: mm
	
	
	// 保存origin信息到元数据
	imageMetadata.origin[0] = srcData->OriginX;
	imageMetadata.origin[1] = srcData->OriginY;
	imageMetadata.origin[2] = srcData->OriginZ;
	
	// 保存spacing信息到元数据
	imageMetadata.spacing[0] = voxelSpacingX;
	imageMetadata.spacing[1] = voxelSpacingY;
	imageMetadata.spacing[2] = voxelSpacingZ;
	

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return UnetSegAI_STATUS_VOLUME_SMALL; //输入体积太小�

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return UnetSegAI_STATUS_VOLUME_LARGE; //输入体积太大

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume太小�
		return UnetSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing太小�
		return UnetSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.1f || voxelSpacingX > 1.1f || voxelSpacingY > 1.1f || voxelSpacingZ > 1.1f)
		return UnetSegAI_STATUS_VOLUME_SMALL; //voxelSpacing太大

	// 创建CImg对象并复制数据
	//RAI: 右-前-上坐标系，与医学图像标准一致
	input_cbct_volume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(input_cbct_volume.data(), srcData->ptr_Data, volSize);

	// 移除统计计算 - 将在预处理管道中的正确位置计算
	// intensity_mean和intensity_std将在crop后计算，或使用JSON配置中的值

	// 清空并重新初始化spacing向量，确保大小正确
	input_voxel_spacing.clear();
	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth
	
	// 读取原始spacing - 增强的防御性检查
	original_voxel_spacing.clear();
	
	// 检查OriginalVoxelSpacing是否为合理值
	// 考虑未初始化内存可能包含的值：负数、零、极大值、NaN等
	bool originalSpacingValid = true;
	
	// 检查是否为合理的医学图像spacing范围 (0.01mm - 10mm)
	const float MIN_VALID_SPACING = 0.01f;
	const float MAX_VALID_SPACING = 10.0f;
	
	if (srcData->OriginalVoxelSpacingX <= MIN_VALID_SPACING || 
	    srcData->OriginalVoxelSpacingX >= MAX_VALID_SPACING ||
	    srcData->OriginalVoxelSpacingY <= MIN_VALID_SPACING || 
	    srcData->OriginalVoxelSpacingY >= MAX_VALID_SPACING ||
	    srcData->OriginalVoxelSpacingZ <= MIN_VALID_SPACING || 
	    srcData->OriginalVoxelSpacingZ >= MAX_VALID_SPACING) {
		originalSpacingValid = false;
	}
	
	// 检查是否为NaN或无穷大
	if (!std::isfinite(srcData->OriginalVoxelSpacingX) ||
	    !std::isfinite(srcData->OriginalVoxelSpacingY) ||
	    !std::isfinite(srcData->OriginalVoxelSpacingZ)) {
		originalSpacingValid = false;
	}
	
	// 检查是否与input_voxel_spacing相差太大（可能是垃圾值）
	if (originalSpacingValid) {
		float ratioX = srcData->OriginalVoxelSpacingX / voxelSpacingX;
		float ratioY = srcData->OriginalVoxelSpacingY / voxelSpacingY;
		float ratioZ = srcData->OriginalVoxelSpacingZ / voxelSpacingZ;
		
		// 如果比例相差超过100倍，很可能是垃圾值
		if (ratioX < 0.01f || ratioX > 100.0f ||
		    ratioY < 0.01f || ratioY > 100.0f ||
		    ratioZ < 0.01f || ratioZ > 100.0f) {
			originalSpacingValid = false;
			std::cout << "Warning: OriginalVoxelSpacing seems invalid (ratio check failed). Using current spacing as original." << std::endl;
		}
	}
	
	if (originalSpacingValid) {
		// 使用提供的原始spacing
		original_voxel_spacing = { srcData->OriginalVoxelSpacingX, srcData->OriginalVoxelSpacingY, srcData->OriginalVoxelSpacingZ };
		std::cout << "Using provided OriginalVoxelSpacing: " << srcData->OriginalVoxelSpacingX 
		          << " x " << srcData->OriginalVoxelSpacingY 
		          << " x " << srcData->OriginalVoxelSpacingZ << " mm" << std::endl;
	} else {
		// 如果原始spacing无效或未提供，使用当前spacing作为原始spacing
		original_voxel_spacing = input_voxel_spacing;
		std::cout << "Note: OriginalVoxelSpacing not provided or invalid. Using current spacing as original." << std::endl;
	}
	
	// 确保spacing向量大小为3
	if (input_voxel_spacing.size() != 3 || original_voxel_spacing.size() != 3) {
		std::cerr << "Error: Failed to initialize spacing vectors properly" << std::endl;
		return UnetSegAI_STATUS_FAIED;
	}

	// 统计信息将在预处理流水线中计算
	std::cout << "Input volume loaded successfully" << endl;
	std::cout << "  Dimensions: " << Width0 << " x " << Height0 << " x " << Depth0 << endl;
	std::cout << "  Spacing: " << input_voxel_spacing[0] << " x " << input_voxel_spacing[1] << " x " << input_voxel_spacing[2] << " mm" << endl;

	return UnetSegAI_STATUS_SUCCESS;
}

AI_INT  UnetMain::performInference(AI_DataInfo *srcData)
{
	// 检查模型是否已加载
	if (model_backend == ModelBackend::ONNX) {
		if (!session_initialized || !semantic_seg_session_ptr) {
			std::cerr << "Error: ONNX Session not initialized. Please set model path first." << std::endl;
			return UnetSegAI_LOADING_FAIED;
		}
	} else if (model_backend == ModelBackend::TORCH) {
		if (!torch_model_loaded) {
			std::cerr << "Error: TorchScript model not loaded. Please set model path first." << std::endl;
			return UnetSegAI_LOADING_FAIED;
		}
	} else {
		std::cerr << "Error: No model loaded. Please set model path first." << std::endl;
		return UnetSegAI_LOADING_FAIED;
	}
	
	int input_status = setInput(srcData);
	if (input_status != UnetSegAI_STATUS_SUCCESS)
		return input_status;

	// 使用新的预处理类进行预处理
	CImg<float> preprocessed_volume;
	AI_INT preprocess_status = UnetPreprocessor::preprocessVolume(this, unetConfig, 
	                                                             input_cbct_volume, 
	                                                             preprocessed_volume);
	if (preprocess_status != UnetSegAI_STATUS_SUCCESS) {
		return preprocess_status;
	}

	// 调用滑窗推理
	std::cout << "\n======= Sliding Window Inference =======" << endl;
	auto inference_start = std::chrono::steady_clock::now();
	
	try {
		AI_INT is_ok = UnetSegAI_STATUS_FAIED;
		
		// 根据模型后端选择推理方法
		if (model_backend == ModelBackend::ONNX) {
			// ONNX Runtime 推理
			is_ok = UnetInference::runSlidingWindow(this, unetConfig, preprocessed_volume, 
			                                        predicted_output_prob, semantic_seg_session_ptr.get(),
			                                        cached_input_name, cached_output_name);
		} else if (model_backend == ModelBackend::TORCH) {
			// TorchScript 推理
			if (!torch_model_loaded) {
				std::cerr << "Error: TorchScript model not loaded" << std::endl;
				return UnetSegAI_LOADING_FAIED;
			}
			is_ok = UnetTorchInference::runSlidingWindowTorch(this, unetConfig, preprocessed_volume,
			                                                  predicted_output_prob, torch_model, use_gpu);
		} else {
			std::cerr << "Error: No valid model backend selected" << std::endl;
			return UnetSegAI_LOADING_FAIED;
		}
		
		if (is_ok != UnetSegAI_STATUS_SUCCESS) {
			return is_ok;
		}
	} catch (const c10::Error& e) {
		std::cerr << "LibTorch error: " << e.what() << std::endl;
		return UnetSegAI_STATUS_FAIED;
	} catch (const std::exception& e) {
		std::cerr << "Inference error: " << e.what() << std::endl;
		return UnetSegAI_STATUS_FAIED;
	} catch (...) {
		std::cerr << "Unknown error during inference" << std::endl;
		return UnetSegAI_STATUS_FAIED;
	}

	auto inference_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> inference_elapsed = inference_end - inference_start;
	std::cout << "Inference completed in " << inference_elapsed.count() << " seconds" << endl;
	std::cout << "======= Inference Complete =======" << endl;

	// 如果进行了重采样，调整大小回原始尺寸
	// 注意：这里需要获取裁剪后的尺寸
	std::vector<int64_t> cropped_size = { 
		crop_bbox.x_max - crop_bbox.x_min + 1,
		crop_bbox.y_max - crop_bbox.y_min + 1,
		crop_bbox.z_max - crop_bbox.z_min + 1
	};
	
	// 检查是否需要resize
	if (predicted_output_prob.width() != cropped_size[0] ||
	    predicted_output_prob.height() != cropped_size[1] ||
	    predicted_output_prob.depth() != cropped_size[2]) {
		predicted_output_prob.resize(cropped_size[0], cropped_size[1], cropped_size[2], unetConfig.num_classes, 3);
	}

	// 保存模型输出（概率体）
	if (saveIntermediateResults && !modelOutputPath.empty()) {
		UnetIO::saveModelOutput(predicted_output_prob, modelOutputPath, L"model_output_probability");
		std::cout << "  Model output saved to: result/model_output/" << endl;
	}

	// 不在这里执行argmax，保持概率图供后续处理
	// argmax将在getSegMask中的后处理流程中执行

	return UnetSegAI_STATUS_SUCCESS;
}

AI_INT  UnetMain::getSegMask(AI_DataInfo *dstData)
{
	// 使用新的UnetPostprocessor类进行后处理
	return UnetPostprocessor::processSegmentationMask(this, predicted_output_prob, dstData);
}

// 检测模型后端类型
UnetMain::ModelBackend UnetMain::detectModelBackend(const wchar_t* model_path)
{
	if (model_path == nullptr) {
		return ModelBackend::UNKNOWN;
	}
	
	std::wstring path(model_path);
	
	// 转换为小写进行比较
	std::transform(path.begin(), path.end(), path.begin(), ::tolower);
	
	if (path.find(L".onnx") != std::wstring::npos) {
		return ModelBackend::ONNX;
	} else if (path.find(L".pt") != std::wstring::npos || path.find(L".pth") != std::wstring::npos) {
		return ModelBackend::TORCH;
	}
	
	return ModelBackend::UNKNOWN;
}

// 宽字符串转窄字符串
std::string UnetMain::wstringToString(const std::wstring& wstr)
{
	if (wstr.empty()) return std::string();
	
	int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
	std::string strTo(size_needed, 0);
	WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
	return strTo;
}

// 初始化 TorchScript 模型
AI_INT UnetMain::initializeTorchModel()
{
	// 如果已经初始化，先释放旧的模型
	if (torch_model_loaded) {
		torch_model = torch::jit::script::Module();
		torch_model_loaded = false;
	}
	
	// 检查模型文件路径
	if (unetConfig.model_file_name == nullptr) {
		std::cerr << "Error: Model file path not set" << std::endl;
		return UnetSegAI_LOADING_FAIED;
	}
	
	try {
		// 检测是否有 CUDA
		if (use_gpu && torch::cuda::is_available()) {
			std::cout << "CUDA is available for LibTorch" << std::endl;
		} else {
			use_gpu = false;
			std::cout << "CUDA not available for LibTorch, using CPU" << std::endl;
		}
		
		torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);
		
		// 转换宽字符路径为窄字符
		std::wstring wpath(unetConfig.model_file_name);
		std::string model_path = wstringToString(wpath);
		
		std::cout << "Loading TorchScript model: " << model_path << std::endl;
		
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
	} catch (const std::exception& e) {
		std::cerr << "Error loading model: " << e.what() << std::endl;
		return UnetSegAI_LOADING_FAIED;
	}
}



