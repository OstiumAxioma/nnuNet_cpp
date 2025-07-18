#include "../header/DentalUnet.h"

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = true;

	// 设置DLL搜索路径以包含系统CUDA目录
	char* cudaPathEnv = nullptr;
	size_t len = 0;
	if (_dupenv_s(&cudaPathEnv, &len, "CUDA_PATH") == 0 && cudaPathEnv != nullptr) {
		std::string cudaBinPath = std::string(cudaPathEnv) + "\\bin";
		std::wstring wcudaBinPath(cudaBinPath.begin(), cudaBinPath.end());
		
		// 添加CUDA bin目录到DLL搜索路径
		SetDllDirectoryW(wcudaBinPath.c_str());
		AddDllDirectory(wcudaBinPath.c_str());
		
		std::cout << "Added CUDA bin to DLL search path: " << cudaBinPath << endl;
		free(cudaPathEnv);
	}
	
	// 诊断CUDA库加载情况
	diagnoseCUDALibraries();

	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnUNetInference");
	
	// 默认启用GPU
	use_gpu = true;
	
	// 检查可用的执行提供者
	std::vector<std::string> providers = Ort::GetAvailableProviders();
	std::cout << "=== Available Execution Providers ===" << endl;
	
	bool cuda_available = false;
	for (const auto& provider : providers) {
		std::cout << "可用Provider: " << provider << std::endl;
		if (provider == "CUDAExecutionProvider") {
			cuda_available = true;
			std::cout << "=== CUDA Provider detected ===" << endl;
		}
	}
	
	if (!cuda_available) {
		std::cerr << "=== ERROR: CUDA Provider not available! ===" << endl;
		throw std::runtime_error("CUDA Provider not available - GPU is required for this application");
	}
	
	// 先初始化配置参数
	unetConfig.model_file_name = nullptr;
	unetConfig.input_channels = 1;
	unetConfig.num_classes = 3;
	unetConfig.mandible_label = 1;
	unetConfig.maxilla_label = 2;
	unetConfig.sinus_label = 3;
	unetConfig.cimg_transpose_forward  = "xyz";
	unetConfig.cimg_transpose_backward = "xyz";
	unetConfig.transpose_forward  = { 0, 1, 2 };
	unetConfig.transpose_backward = { 0, 1, 2 };
	unetConfig.voxel_spacing = { 0.5810545086860657f, 0.5810545086860657f, 1.0f };
	unetConfig.patch_size = { 160, 160, 96 };
	unetConfig.step_size_ratio = 0.75f;
	unetConfig.normalization_type = "CTNormalization";
	unetConfig.min_max_HU = { -172.01852416992188f,  1824.9935302734375f };
	unetConfig.mean_std_HU = { 274.2257080078125f, 366.05450439453125f };
	unetConfig.use_mirroring = false;
	
	// 初始化GPU配置，但不创建会话（因为模型路径还未设置）
	if (!initializeGPU()) {
		std::cerr << "=== ERROR: GPU initialization failed! ===" << endl;
		throw std::runtime_error("GPU initialization failed - GPU is required for this application");
	}
	
	std::cout << "=== GPU initialized successfully ===" << endl;
	
	// 添加调试信息：验证构造函数中vector初始化
	std::cout << "=== DEBUG: DentalUnet constructor end ===" << endl;
	std::cout << "GPU Status: " << (use_gpu ? "ENABLED" : "DISABLED") << endl;
	std::cout << "unetConfig.transpose_forward.size(): " << unetConfig.transpose_forward.size() << endl;
	std::cout << "unetConfig.transpose_backward.size(): " << unetConfig.transpose_backward.size() << endl;
	std::cout << "unetConfig.voxel_spacing.size(): " << unetConfig.voxel_spacing.size() << endl;
	std::cout << "unetConfig.patch_size.size(): " << unetConfig.patch_size.size() << endl;
}


DentalUnet::~DentalUnet()
{
}

void DentalUnet::diagnoseCUDALibraries() 
{
	std::cout << "=== CUDA Library Diagnostic ===" << endl;
	
	// 检查关键CUDA 12.x库
	std::vector<std::string> cuda_dlls = {
		"cublas64_12.dll",
		"cublasLt64_12.dll", 
		"cudnn64_9.dll",
		"cudnn_ops_infer64_9.dll",
		"cudnn_cnn_infer64_9.dll",
		"cudnn_adv_infer64_9.dll",
		"cudart64_12.dll"
	};
	
	// 检查ONNX Runtime Provider库
	std::vector<std::string> ort_dlls = {
		"onnxruntime_providers_cuda.dll",
		"onnxruntime_providers_shared.dll",
		"onnxruntime.dll"
	};
	
	bool allCudaLoaded = true;
	bool allOrtLoaded = true;
	
	std::cout << "\n=== Checking CUDA 12.x/cuDNN 9.x DLLs ===" << endl;
	for (const auto& dll : cuda_dlls) {
		HMODULE handle = LoadLibraryA(dll.c_str());
		if (handle) {
			std::cout << "[OK] " << dll << " - Loaded successfully" << endl;
			FreeLibrary(handle);
		} else {
			DWORD error = GetLastError();
			std::cout << "[FAIL] " << dll << " - Failed to load (Error: " << error << ")" << endl;
			allCudaLoaded = false;
		}
	}
	
	std::cout << "\n=== Checking ONNX Runtime Provider DLLs ===" << endl;
	for (const auto& dll : ort_dlls) {
		HMODULE handle = LoadLibraryA(dll.c_str());
		if (handle) {
			std::cout << "[OK] " << dll << " - Loaded successfully" << endl;
			FreeLibrary(handle);
		} else {
			DWORD error = GetLastError();
			std::cout << "[FAIL] " << dll << " - Failed to load (Error: " << error << ")" << endl;
			allOrtLoaded = false;
		}
	}
	
	std::cout << "\n=== Diagnostic Summary ===" << endl;
	if (!allCudaLoaded) {
		std::cerr << "ERROR: Missing CUDA 12.x or cuDNN 9.x libraries!" << endl;
		std::cerr << "Please ensure CUDA 12.x and cuDNN 9.x are installed." << endl;
		std::cerr << "CUDA/cuDNN path should be: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin" << endl;
	}
	if (!allOrtLoaded) {
		std::cerr << "ERROR: Missing ONNX Runtime GPU libraries!" << endl;
		std::cerr << "Please ensure ONNX Runtime GPU package for CUDA 12.x is properly installed." << endl;
	}
	
	if (!allCudaLoaded || !allOrtLoaded) {
		throw std::runtime_error("Required CUDA 12.x/cuDNN 9.x libraries not found");
	}
	
	std::cout << "All required libraries found successfully!" << endl;
	std::cout << "===================================\n" << endl;
}


DentalUnet *DentalUnet::CreateDentalUnet()
{
	std::cout << "=== DEBUG: CreateDentalUnet function start ===" << endl;
	
	DentalUnet *segUnetModel = new DentalUnet();
	
	std::cout << "=== DEBUG: DentalUnet object created successfully ===" << endl;
	std::cout << "segUnetModel pointer: " << segUnetModel << endl;

	std::cout << "CreateDentalUnet is done. "<<endl;

	return segUnetModel;
}

void  DentalUnet::setModelFns(const char* model_fn)
{
	if (model_fn != nullptr) {
		size_t len = strlen(model_fn) + 1;
		wchar_t* wide_fn = new wchar_t[len];
		mbstowcs(wide_fn, model_fn, len);
		unetConfig.model_file_name = wide_fn;
		// Note: This creates a memory leak, but for simplicity we're not handling it here
	}
}

void  DentalUnet::setModelFns(const wchar_t* model_fn)
{
	unetConfig.model_file_name = model_fn;
}


void  DentalUnet::setStepSizeRatio(float ratio)
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


void  DentalUnet::setDnnOptions()
{
	//��������չӲ������������
}


void  DentalUnet::setAlgParameter()
{
	//�����������㷨����
}


AI_INT  DentalUnet::initializeOnnxruntimeInstances()
{
	// GPU已经在构造函数中初始化，这里只需要确认状态
	if (!use_gpu) {
		std::cerr << "=== ERROR: GPU not available in initializeOnnxruntimeInstances ===" << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	}
	
	std::cout << "=== GPU session options already configured ===" << endl;
	std::cout << "=== ONNX Runtime instances ready ===" << endl;
	
	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::setInput(AI_DataInfo *srcData)
{
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //��λ: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //��λ: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //��λ: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //��λ: mm

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //���������ݹ�С

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return DentalCbctSegAI_STATUS_VOLUME_LARGE; //���������ݹ���

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume��С
		return DentalCbctSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing��С
		return DentalCbctSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.1f || voxelSpacingX > 1.1f || voxelSpacingY > 1.1f || voxelSpacingZ > 1.1f)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //voxelSpacing���

	// �����������ݵ�CImg����
	//RAI: ������ǰ���������ں󣻶��������ң��°����ϣ�ͷ������
	input_cbct_volume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(input_cbct_volume.data(), srcData->ptr_Data, volSize);

	intensity_mean = (float)input_cbct_volume.mean();
	intensity_std = (float)input_cbct_volume.variance();
	intensity_std = std::sqrt(intensity_std);
	if (intensity_std < 0.0001f)
		intensity_std = 0.0001f;

	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth

	// 添加调试信息：检查input_voxel_spacing的创建
	std::cout << "=== DEBUG: setInput function end ===" << endl;
	std::cout << "voxelSpacingX: " << voxelSpacingX << endl;
	std::cout << "voxelSpacingY: " << voxelSpacingY << endl;
	std::cout << "voxelSpacingZ: " << voxelSpacingZ << endl;
	std::cout << "input_voxel_spacing.size() after creation: " << input_voxel_spacing.size() << endl;
	std::cout << "input_voxel_spacing values: ";
	for (size_t i = 0; i < input_voxel_spacing.size(); ++i) {
		std::cout << input_voxel_spacing[i] << " ";
	}
	std::cout << endl;

	std::cout << "input_volume intensity_mean: " << intensity_mean << endl;
	std::cout << "input_volume intensity_std: " << intensity_std << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::performInference(AI_DataInfo *srcData)
{
	std::cout << "=== DEBUG: performInference function start ===" << endl;
	std::cout << "srcData pointer: " << srcData << endl;
	if (srcData != nullptr) {
		std::cout << "srcData->Width: " << srcData->Width << endl;
		std::cout << "srcData->Height: " << srcData->Height << endl;
		std::cout << "srcData->Depth: " << srcData->Depth << endl;
		std::cout << "srcData->VoxelSpacingX: " << srcData->VoxelSpacingX << endl;
		std::cout << "srcData->VoxelSpacingY: " << srcData->VoxelSpacingY << endl;
		std::cout << "srcData->VoxelSpacingZ: " << srcData->VoxelSpacingZ << endl;
	}
	
	std::cout << "About to call setInput..." << endl;
	int input_status = setInput(srcData);
	std::cout << "setInput returned, input_status: " << input_status << endl;
	if (input_status != DentalCbctSegAI_STATUS_SUCCESS)
		return input_status;

	std::cout << "=== CHECKPOINT 1: setInput completed successfully ===" << endl;

	// 添加调试信息：检查关键vector的大小和内容
	std::cout << "=== DEBUG: Vector sizes before permute_axes ===" << endl;
	std::cout << "input_voxel_spacing.size(): " << input_voxel_spacing.size() << endl;
	std::cout << "unetConfig.transpose_forward.size(): " << unetConfig.transpose_forward.size() << endl;
	std::cout << "unetConfig.transpose_backward.size(): " << unetConfig.transpose_backward.size() << endl;
	
	// 打印vector内容
	std::cout << "input_voxel_spacing values: ";
	for (size_t i = 0; i < input_voxel_spacing.size(); ++i) {
		std::cout << input_voxel_spacing[i] << " ";
	}
	std::cout << endl;
	
	std::cout << "unetConfig.transpose_forward values: ";
	for (size_t i = 0; i < unetConfig.transpose_forward.size(); ++i) {
		std::cout << unetConfig.transpose_forward[i] << " ";
	}
	std::cout << endl;
	
	std::cout << "unetConfig.transpose_backward values: ";
	for (size_t i = 0; i < unetConfig.transpose_backward.size(); ++i) {
		std::cout << unetConfig.transpose_backward[i] << " ";
	}
	std::cout << endl;

	std::cout << "=== CHECKPOINT 2: About to call permute_axes ===" << endl;
	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_forward);
	std::cout << "permute_axes completed successfully" << endl;
	
	std::cout << "=== CHECKPOINT 3: permute_axes completed ===" << endl;
	
	transposed_input_voxel_spacing.clear();
	std::cout << "=== DEBUG: About to enter critical loop ===" << endl;
	
	for (int i = 0; i < 3; ++i) {
		std::cout << "Loop iteration i=" << i << endl;
		std::cout << "  Checking unetConfig.transpose_forward[" << i << "]..." << endl;
		
		if (i >= (int)unetConfig.transpose_forward.size()) {
			std::cerr << "ERROR: i=" << i << " >= unetConfig.transpose_forward.size()=" << unetConfig.transpose_forward.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		int transpose_idx = unetConfig.transpose_forward[i];
		std::cout << "  transpose_idx = " << transpose_idx << endl;
		
		if (transpose_idx < 0 || transpose_idx >= (int)input_voxel_spacing.size()) {
			std::cerr << "ERROR: transpose_idx=" << transpose_idx << " is out of range for input_voxel_spacing.size()=" << input_voxel_spacing.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		std::cout << "  Accessing input_voxel_spacing[" << transpose_idx << "]..." << endl;
		float spacing_value = input_voxel_spacing[transpose_idx];
		std::cout << "  spacing_value = " << spacing_value << endl;
		
		transposed_input_voxel_spacing.push_back(spacing_value);
		std::cout << "  Successfully added to transposed_input_voxel_spacing" << endl;
	}
	
	std::cout << "=== DEBUG: Critical loop completed successfully ===" << endl;
	std::cout << "transposed_input_voxel_spacing.size(): " << transposed_input_voxel_spacing.size() << endl;

	//apply CNN
	int infer_status = segModelInfer(unetConfig, input_cbct_volume);
	std::cout << "infer_status: " << infer_status << endl;

	output_seg_mask.permute_axes(unetConfig.cimg_transpose_backward);
	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_backward);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::segModelInfer(nnUNetConfig config, CImg<short> input_volume)
{
	// 添加调试信息：检查关键vector的大小和内容
	std::cout << "=== DEBUG: segModelInfer function start ===" << endl;
	std::cout << "transposed_input_voxel_spacing.size(): " << transposed_input_voxel_spacing.size() << endl;
	std::cout << "config.voxel_spacing.size(): " << config.voxel_spacing.size() << endl;
	std::cout << "config.patch_size.size(): " << config.patch_size.size() << endl;
	
	std::cout << "transposed_input_voxel_spacing values: ";
	for (size_t i = 0; i < transposed_input_voxel_spacing.size(); ++i) {
		std::cout << transposed_input_voxel_spacing[i] << " ";
	}
	std::cout << endl;
	
	std::cout << "config.voxel_spacing values: ";
	for (size_t i = 0; i < config.voxel_spacing.size(); ++i) {
		std::cout << config.voxel_spacing[i] << " ";
	}
	std::cout << endl;
	
	std::cout << "config.patch_size values: ";
	for (size_t i = 0; i < config.patch_size.size(); ++i) {
		std::cout << config.patch_size[i] << " ";
	}
	std::cout << endl;

	if (transposed_input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// 计算目标尺寸
	bool is_volume_scaled = false;
	////input_voxel_spacing = {voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth 
	std::vector<int64_t> input_size = { input_volume.width(), input_volume.height(), input_volume.depth()};
	std::vector<int64_t> output_size;
	float scaled_factor = 1.f;
	
	std::cout << "=== DEBUG: About to enter scaling loop ===" << endl;
	for (int i = 0; i < 3; ++i) {  // 只处理空间维度
		std::cout << "Scaling loop iteration i=" << i << endl;
		
		if (i >= (int)transposed_input_voxel_spacing.size()) {
			std::cerr << "ERROR: i=" << i << " >= transposed_input_voxel_spacing.size()=" << transposed_input_voxel_spacing.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		if (i >= (int)config.voxel_spacing.size()) {
			std::cerr << "ERROR: i=" << i << " >= config.voxel_spacing.size()=" << config.voxel_spacing.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		if (i >= (int)config.patch_size.size()) {
			std::cerr << "ERROR: i=" << i << " >= config.patch_size.size()=" << config.patch_size.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		if (i >= (int)input_size.size()) {
			std::cerr << "ERROR: i=" << i << " >= input_size.size()=" << input_size.size() << endl;
			return DentalCbctSegAI_STATUS_FAIED;
		}
		
		std::cout << "  Accessing transposed_input_voxel_spacing[" << i << "] = " << transposed_input_voxel_spacing[i] << endl;
		std::cout << "  Accessing config.voxel_spacing[" << i << "] = " << config.voxel_spacing[i] << endl;
		
		scaled_factor = transposed_input_voxel_spacing[i] / config.voxel_spacing[i];
		int scaled_sz = std::round(input_size[i] * scaled_factor);
		
		std::cout << "  scaled_factor = " << scaled_factor << ", scaled_sz = " << scaled_sz << endl;
		std::cout << "  config.patch_size[" << i << "] = " << config.patch_size[i] << endl;
		
		if (scaled_factor < 0.9f || scaled_factor > 1.1f || scaled_sz < config.patch_size[i])
			is_volume_scaled = true;

		if (scaled_sz < config.patch_size[i])
			scaled_sz = config.patch_size[i];

		output_size.push_back(static_cast<int64_t>(scaled_sz));
		std::cout << "  output_size[" << i << "] = " << scaled_sz << endl;
	}
	std::cout << "=== DEBUG: Scaling loop completed successfully ===" << endl;

	CImg<float> scaled_input_volume;
	if (is_volume_scaled)
		scaled_input_volume = input_volume.get_resize(output_size[0], output_size[1], output_size[2], -100, 3);
	else
		scaled_input_volume.assign(input_volume);

	std::cout << "scaled_input_volume depth: " << scaled_input_volume.depth() << endl;
	std::cout << "scaled_input_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "scaled_input_volume variance: " << scaled_input_volume.variance() << endl;

	//��һ������
	std::map<std::string, int> normalizationOptionsMap = {
		{"CTNormalization",     10},
		{"CT",                  10},
		{"ct",                  10},
		{"CTNorm",              10},
		{"ctnorm",              10},
		{"ZScoreNormalization", 20},
		{"zscore",              20},
		{"z-score",             20},
	};
	auto it = normalizationOptionsMap.find(config.normalization_type);
	int normlization_type = 20;
	if (it != normalizationOptionsMap.end())
		normlization_type = it->second;
	else
		normlization_type = 20;

	switch (normlization_type) {
	case 10:
		CTNormalization(scaled_input_volume, config);
		break;
	case 20:
		scaled_input_volume -= intensity_mean;
		scaled_input_volume /= intensity_std;
		break;
	default:
		scaled_input_volume -= intensity_mean;
		scaled_input_volume /= intensity_std;
		break;
	}
	std::cout << "normalized_input_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "normalized_input_volume variance: " << scaled_input_volume.variance() << endl;

	//����������Ԥ��
	AI_INT is_ok = slidingWindowInfer(config, scaled_input_volume);
	std::cout << "slidingWindowInfer: " << is_ok << endl;

	//ʹ��3D��ֵ��������
	if (is_volume_scaled)
		predicted_output_prob.resize(input_size[0], input_size[1], input_size[2], config.num_classes, 3);

	output_seg_mask = argmax_spectrum(predicted_output_prob);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume)
{
	std::cout << "=== DEBUG: slidingWindowInfer function start ===" << endl;
	
	if (!use_gpu) {
		std::cerr << "=== ERROR: GPU not available in slidingWindowInfer ===" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}

	std::cout << "=== DEBUG: Using pre-configured GPU session options ===" << endl;

	// 创建会话
	std::cout << "=== DEBUG: Creating ONNX session ===" << endl;
	std::wcout << L"Model file: " << config.model_file_name << endl;
	
	// 验证模型文件存在并获取文件信息
	WIN32_FILE_ATTRIBUTE_DATA fileInfo;
	if (GetFileAttributesExW(config.model_file_name, GetFileExInfoStandard, &fileInfo)) {
		LARGE_INTEGER fileSize;
		fileSize.HighPart = fileInfo.nFileSizeHigh;
		fileSize.LowPart = fileInfo.nFileSizeLow;
		std::cout << "=== Model File Information ===" << endl;
		std::cout << "Model file size: " << (fileSize.QuadPart / (1024.0 * 1024.0)) << " MB" << endl;
		
		// 估算模型内存需求（通常是文件大小的2-4倍）
		std::cout << "Estimated memory requirement: " << (fileSize.QuadPart * 3 / (1024.0 * 1024.0)) << " MB (approx)" << endl;
	} else {
		std::cerr << "=== ERROR: Model file not found! ===" << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	}
	
	// 输出当前内存状态
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	if (GlobalMemoryStatusEx(&memInfo)) {
		std::cout << "=== System Memory Status ===" << endl;
		std::cout << "Total Physical Memory: " << (memInfo.ullTotalPhys / (1024 * 1024)) << " MB" << endl;
		std::cout << "Available Physical Memory: " << (memInfo.ullAvailPhys / (1024 * 1024)) << " MB" << endl;
		std::cout << "Memory Load: " << memInfo.dwMemoryLoad << "%" << endl;
	}
	
	// 声明变量
	Ort::Session* session_ptr = nullptr;
	const char* input_name = nullptr;
	const char* output_name = nullptr;
	std::vector<int64_t> input_tensor_shape;
	std::vector<int64_t> input_shape;
	
	try {
		std::cout << "=== About to create Ort::Session with pre-configured options ===" << endl;
		// 直接使用预配置的session_options创建会话
		session_ptr = new Ort::Session(env, config.model_file_name, session_options);
		std::cout << "=== DEBUG: Session created successfully ===" << endl;
		
		// 获取模型输入输出信息
		size_t num_input_nodes = session_ptr->GetInputCount();
		size_t num_output_nodes = session_ptr->GetOutputCount();
		std::cout << "=== Model Structure Information ===" << endl;
		std::cout << "Number of inputs: " << num_input_nodes << endl;
		std::cout << "Number of outputs: " << num_output_nodes << endl;
		
		Ort::AllocatorWithDefaultOptions allocator;
		input_name = session_ptr->GetInputNameAllocated(0, allocator).get();
		output_name = session_ptr->GetOutputNameAllocated(0, allocator).get();

		std::cout << "Session loading is done: " << endl;
		std::cout << "input_name: " << input_name << endl;
		std::cout << "output_name: " << output_name << endl;
		
		std::cout << "=== DEBUG: Getting input shape ===" << endl;
		input_shape = session_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		auto element_type = session_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
		std::cout << "=== DEBUG: Input shape obtained ===" << endl;
		
		// 显示详细的输入信息
		std::cout << "=== Model Input Tensor Details ===" << endl;
		std::cout << "Input shape: ";
		int64_t total_elements = 1;
		for (size_t i = 0; i < input_shape.size(); ++i) {
			std::cout << input_shape[i] << " ";
			if (input_shape[i] > 0) {
				total_elements *= input_shape[i];
			}
		}
		std::cout << endl;
		std::cout << "Input element type: " << element_type << " (1=float, 7=int64, 9=bool)" << endl;
		
		// 计算输入张量的内存需求
		size_t bytes_per_element = (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? 4 : 4;
		size_t input_memory_mb = (total_elements * bytes_per_element) / (1024 * 1024);
		std::cout << "Input tensor memory requirement: " << input_memory_mb << " MB" << endl;

		if (input_shape.size() != 5) {
			std::cerr << "ERROR: Expected 5D input, got " << input_shape.size() << "D" << endl;
			throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
		}

		// 验证图像大小
		if (config.patch_size.size() != 3) {
			std::cerr << "ERROR: Patch size should be 3D, got " << config.patch_size.size() << "D" << endl;
			throw std::runtime_error("Patch size should be 3D (depth, height, width)");
		}

		std::cout << "=== DEBUG: About to create input_tensor_shape ===" << endl;
		// 准备输入张量形状 (1, 1, D, H, W)
		input_tensor_shape = { 1, 1, config.patch_size[2], config.patch_size[1], config.patch_size[0] };
		std::cout << "=== DEBUG: input_tensor_shape created successfully ===" << endl;
		
	}
	catch (const Ort::Exception& e) {
		std::cerr << "=== ONNX Session Creation Exception ===" << endl;
		std::cerr << "Error code: " << e.GetOrtErrorCode() << endl;
		std::cerr << "Error message: " << e.what() << endl;
		
		// 如果是CUDA相关错误，提供更详细的信息
		std::string error_msg = e.what();
		if (error_msg.find("CUBLAS_STATUS_ALLOC_FAILED") != std::string::npos) {
			std::cerr << "=== CUBLAS Memory Allocation Failed ===" << endl;
			std::cerr << "This usually means insufficient GPU memory" << endl;
			std::cerr << "Try closing other GPU applications" << endl;
		}
		
		if (session_ptr) delete session_ptr;
		return DentalCbctSegAI_LOADING_FAIED;
	}
	catch (const std::exception& e) {
		std::cerr << "=== Session Creation Standard Exception ===" << endl;
		std::cerr << "Error message: " << e.what() << endl;
		if (session_ptr) delete session_ptr;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	catch (...) {
		std::cerr << "=== Unknown Session Creation Exception ===" << endl;
		if (session_ptr) delete session_ptr;
		return DentalCbctSegAI_STATUS_FAIED;
	}

	// 如果到这里说明会话创建成功
	Ort::Session& session = *session_ptr;

	int depth = normalized_volume.depth();
	int width = normalized_volume.width();
	int height = normalized_volume.height();

	std::cout << "=== DEBUG: Volume dimensions: " << width << "x" << height << "x" << depth << endl;

	// x Image width, y Image height, z Image depth
	float step_size_ratio = config.step_size_ratio;
	float actualStepSize[3];
	
	// 添加调试信息：检查patch_size访问
	std::cout << "=== DEBUG: slidingWindowInfer patch_size access ===" << endl;
	std::cout << "config.patch_size.size(): " << config.patch_size.size() << endl;
	if (config.patch_size.size() < 3) {
		std::cerr << "ERROR: config.patch_size.size() < 3, actual size: " << config.patch_size.size() << endl;
		if (session_ptr) delete session_ptr;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	
	std::cout << "About to access config.patch_size[0], [1], [2]..." << endl;
	std::cout << "config.patch_size[0] = " << config.patch_size[0] << endl;
	std::cout << "config.patch_size[1] = " << config.patch_size[1] << endl;
	std::cout << "config.patch_size[2] = " << config.patch_size[2] << endl;
	
	int X_num_steps = (int)ceil(float(width - config.patch_size[0]) / (config.patch_size[0] * step_size_ratio)) + 1; //X
	if (X_num_steps > 1)
		actualStepSize[0] = float(width - config.patch_size[0]) / (X_num_steps - 1);
	else
		actualStepSize[0] = 999999.f;

	int Y_num_steps = (int)ceil(float(height - config.patch_size[1]) / (config.patch_size[1] * step_size_ratio)) + 1; //Y
	if (Y_num_steps > 1)
		actualStepSize[1] = float(height - config.patch_size[1]) / (Y_num_steps - 1);
	else
		actualStepSize[1] = 999999.f;

	int Z_num_steps = (int)ceil(float(depth - config.patch_size[2]) / (config.patch_size[2] * step_size_ratio)) + 1; //Y
	if (Z_num_steps > 1)
		actualStepSize[2] = float(depth - config.patch_size[2]) / (Z_num_steps - 1);
	else
		actualStepSize[2] = 999999.f;

	if (NETDEBUG_FLAG)
		std::cout << "Number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;

	// 初始化预测概率图
	predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(width, height, depth, 1, 0.f);

	CImg<float> win_pob = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], config.num_classes, 0.f);
	CImg<float> gaussisan_weight = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.f);
	create_3d_gaussian_kernel(gaussisan_weight, config.patch_size);

	size_t input_patch_voxel_numel = config.patch_size[0] * config.patch_size[1] * config.patch_size[2];
	size_t output_patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	//遍历所有切片
	int patch_count = 0;
	for (int sz = 0; sz < Z_num_steps; sz++)
	{
		int lb_z = (int)std::round(sz * actualStepSize[2]);
		int ub_z = lb_z + config.patch_size[2] - 1;

		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[1]);
			int ub_y = lb_y + config.patch_size[1] - 1;

			for (int sx = 0; sx < X_num_steps; sx++)
			{
				int lb_x = (int)std::round(sx * actualStepSize[0]);
				int ub_x = lb_x + config.patch_size[0] - 1;

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << "current tile#: " << patch_count << endl;

				CImg<float> input_patch = normalized_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
				std::cout << "input_patch width: " << input_patch.width() << endl;
				std::cout << "input_patch height: " << input_patch.height() << endl;
				std::cout << "input_patch depth: " << input_patch.depth() << endl;

				float* input_data_ptr = input_patch.data();

				// 准备输入张量
				Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
					OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
					input_data_ptr,
					input_patch_voxel_numel,
					input_tensor_shape.data(),
					input_tensor_shape.size());

				// 运行ONNX会话
				std::cout << "Run onnx session." << endl;
				std::cout << "=== DEBUG: About to run ONNX session ===" << endl;
				std::cout << "input_name: " << input_name << endl;
				std::cout << "output_name: " << output_name << endl;
				std::cout << "input_tensor_shape: ";
				for (size_t i = 0; i < input_tensor_shape.size(); ++i) {
					std::cout << input_tensor_shape[i] << " ";
				}
				std::cout << endl;
				std::cout << "input_patch_voxel_numel: " << input_patch_voxel_numel << endl;

				try {
					std::cout << "=== DEBUG: Calling session.Run() ===" << endl;
					auto output_tensors = session.Run(
						Ort::RunOptions{ nullptr },
						&input_name,
						&input_tensor,
						1,
						&output_name,
						1
					);
					std::cout << "=== DEBUG: session.Run() completed successfully ===" << endl;
					
					if (config.use_mirroring && use_gpu)
					{
						input_patch = input_patch.mirror('x');
					}

					std::cout << "onnx session running is done." << endl;

					// 获取输出
					std::cout << "=== DEBUG: Getting output data ===" << endl;
					float* output_data = output_tensors[0].GetTensorMutableData<float>();
					std::cout << "=== DEBUG: Output data obtained ===" << endl;

					// 转换为CImg
					std::cout << "=== DEBUG: Copying output data to CImg ===" << endl;
					std::memcpy(win_pob.data(), output_data, output_patch_vol_sz);
					std::cout << "=== DEBUG: Data copied successfully ===" << endl;
					
					output_tensors.clear();

					std::cout << "output_patch min: " << (win_pob.min)() << endl;
					std::cout << "output_patch max: " << (win_pob.max)() << endl;
					std::cout << "output_patch mean: " << win_pob.mean() << endl;
				}
				catch (const Ort::Exception& e) {
					std::cerr << "=== ONNX Runtime Exception ===" << endl;
					std::cerr << "Error code: " << e.GetOrtErrorCode() << endl;
					std::cerr << "Error message: " << e.what() << endl;
					if (session_ptr) delete session_ptr;
					return DentalCbctSegAI_STATUS_FAIED;
				}
				catch (const std::exception& e) {
					std::cerr << "=== Standard Exception ===" << endl;
					std::cerr << "Error message: " << e.what() << endl;
					if (session_ptr) delete session_ptr;
					return DentalCbctSegAI_STATUS_FAIED;
				}
				catch (...) {
					std::cerr << "=== Unknown Exception ===" << endl;
					if (session_ptr) delete session_ptr;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				cimg_forXYZC(win_pob, x, y, z, c) {
					predicted_output_prob(lb_x + x, lb_y + y, lb_z + z, c) += (win_pob(x, y, z, c) * gaussisan_weight(x, y, z));
				}
				cimg_forXYZ(gaussisan_weight, x, y, z) {
					count_vol(lb_x + x, lb_y + y, lb_z + z) += gaussisan_weight(x, y, z);
				}
			}
		}
	}

	// 平均化
	cimg_forXYZC(predicted_output_prob, x, y, z, c) {
		predicted_output_prob(x, y, z, c) /= count_vol(x, y, z);
	}
	std::cout << "Sliding window inference is done." << endl;

	// 清理内存
	if (session_ptr) {
		delete session_ptr;
		session_ptr = nullptr;
	}

	return DentalCbctSegAI_STATUS_SUCCESS;
}


void  DentalUnet::CTNormalization(CImg<float>& input_volume, nnUNetConfig config)
{
	//HU值归一化
	float min_hu4dentalCTNormalization = config.min_max_HU[0];
	float max_hu4dentalCTNormalization = config.min_max_HU[1];
	input_volume.cut(min_hu4dentalCTNormalization, max_hu4dentalCTNormalization);

	//进行z-score
	float mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	float std_hu4dentalCTNormalization = config.mean_std_HU[1];
	input_volume -= mean_hu4dentalCTNormalization;
	input_volume /= std_hu4dentalCTNormalization;
}


void  DentalUnet::create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes)
{
	// 添加调试信息：检查patch_sizes访问
	std::cout << "=== DEBUG: create_3d_gaussian_kernel function start ===" << endl;
	std::cout << "patch_sizes.size(): " << patch_sizes.size() << endl;
	if (patch_sizes.size() < 3) {
		std::cerr << "ERROR: patch_sizes.size() < 3, actual size: " << patch_sizes.size() << endl;
		return;
	}
	
	std::cout << "patch_sizes values: ";
	for (size_t i = 0; i < patch_sizes.size(); ++i) {
		std::cout << patch_sizes[i] << " ";
	}
	std::cout << endl;
	
	std::vector<float> sigmas(3);
	for (int i = 0; i < 3; ++i) {
		std::cout << "Accessing patch_sizes[" << i << "] = " << patch_sizes[i] << endl;
		sigmas[i] = (patch_sizes[i] - 1) / 5.0f; // 按W=5σ+1推导
	}

	int64_t depth  = patch_sizes[0];
	int64_t height = patch_sizes[1]; 
	int64_t width  = patch_sizes[2];
	
	std::cout << "depth=" << depth << ", height=" << height << ", width=" << width << endl;

	// 每层
	float z_center = (depth - 1)  / 2.0f;
	float y_center = (height - 1) / 2.0f;
	float x_center = (width - 1)  / 2.0f;

	// 中心化（中心点为原点）
	float z_sigma = depth  / 4.0f;
	float y_sigma = height / 4.0f;
	float x_sigma = width  / 4.0f;

	float z_part = 0.f;
	float y_part = 0.f;
	float x_part = 0.f;
	cimg_forXYZ(gaussisan_weight, x, y, z) {
		z_part = static_cast<float>(std::exp(-0.5f * std::pow((z - z_center) / z_sigma, 2)));
		y_part = static_cast<float>(std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2)));
		x_part = static_cast<float>(std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2)));
		gaussisan_weight(x, y, z) = z_part * y_part * x_part;
	}

	gaussisan_weight /= gaussisan_weight.mean();
}


CImg<short> DentalUnet::argmax_spectrum(const CImg<float>& input) {
	if (input.is_empty() || input.spectrum() == 0) {
		throw std::invalid_argument("Input must be a non-empty 4D CImg with spectrum dimension.");
	}

	// 初始化结果图像，维度为(W, H, D, 1)
	CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

	// 遍历每个体素位置 (x,y,z)
	cimg_forXYZ(input, x, y, z) {
		short max_idx = 0;
		float max_val = input(x, y, z, 0);

		// 遍历spectrum维度
		for (short s = 1; s < input.spectrum(); ++s) {
			const float current_val = input(x, y, z, s);
			if (current_val > max_val) {
				max_val = current_val;
				max_idx = s;
			}
		}
		result(x, y, z) = max_idx; // 存储最大值的类别
	}
	return result;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(dstData->ptr_Data, output_seg_mask.data(), volSize);

	return DentalCbctSegAI_STATUS_SUCCESS;
}

// 新增GPU初始化函数
bool DentalUnet::initializeGPU()
{
	std::cout << "=== Starting GPU initialization ===" << endl;
	
	// 输出ONNX Runtime版本信息
	std::cout << "=== ONNX Runtime Version Information ===" << endl;
	std::cout << "ONNX Runtime version: " << ORT_API_VERSION << endl;
	
	// 查询GPU信息
	try {
		// 尝试从系统CUDA路径加载
		std::string cudaPath;
		char* cudaPathEnv = nullptr;
		size_t len = 0;
		if (_dupenv_s(&cudaPathEnv, &len, "CUDA_PATH") == 0 && cudaPathEnv != nullptr) {
			cudaPath = std::string(cudaPathEnv) + "\\bin\\";
			free(cudaPathEnv);
			std::cout << "Found CUDA_PATH: " << cudaPath << endl;
		}
		
		// 动态加载CUDA运行时库 - 优先从系统路径加载
		HMODULE cudaRt = nullptr;
		if (!cudaPath.empty()) {
			// 尝试从CUDA_PATH加载
			std::string cudart12Path = cudaPath + "cudart64_12.dll";
			cudaRt = LoadLibraryA(cudart12Path.c_str());
			if (cudaRt) {
				std::cout << "Loaded CUDA Runtime from system: " << cudart12Path << endl;
			}
		}
		
		// 如果系统路径加载失败，尝试当前目录
		if (!cudaRt) {
			cudaRt = LoadLibraryA("cudart64_12.dll");
			if (cudaRt) {
				std::cout << "Loaded CUDA Runtime: cudart64_12.dll from current directory" << endl;
			} else {
				std::cerr << "ERROR: Failed to load cudart64_12.dll" << endl;
				std::cerr << "CUDA 12.x is required. Please install CUDA 12.x." << endl;
			}
		}
		
		if (cudaRt) {
			// 定义函数指针类型
			typedef int (*cudaGetDeviceCount_t)(int*);
			typedef int (*cudaGetDeviceProperties_t)(void*, int);
			typedef int (*cudaMemGetInfo_t)(size_t*, size_t*);
			typedef const char* (*cudaGetErrorString_t)(int);
			typedef int (*cudaRuntimeGetVersion_t)(int*);
			typedef int (*cudaDriverGetVersion_t)(int*);
			
			// 获取函数地址
			auto cudaGetDeviceCount = (cudaGetDeviceCount_t)GetProcAddress(cudaRt, "cudaGetDeviceCount");
			auto cudaGetDeviceProperties = (cudaGetDeviceProperties_t)GetProcAddress(cudaRt, "cudaGetDeviceProperties");
			auto cudaMemGetInfo = (cudaMemGetInfo_t)GetProcAddress(cudaRt, "cudaMemGetInfo");
			auto cudaGetErrorString = (cudaGetErrorString_t)GetProcAddress(cudaRt, "cudaGetErrorString");
			auto cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)GetProcAddress(cudaRt, "cudaRuntimeGetVersion");
			auto cudaDriverGetVersion = (cudaDriverGetVersion_t)GetProcAddress(cudaRt, "cudaDriverGetVersion");
			
			// 获取CUDA版本信息
			if (cudaRuntimeGetVersion && cudaDriverGetVersion) {
				int runtimeVersion = 0;
				int driverVersion = 0;
				cudaRuntimeGetVersion(&runtimeVersion);
				cudaDriverGetVersion(&driverVersion);
				
				std::cout << "=== CUDA Version Information ===" << endl;
				std::cout << "CUDA Runtime version: " << (runtimeVersion/1000) << "." << ((runtimeVersion%1000)/10) << endl;
				std::cout << "CUDA Driver version: " << (driverVersion/1000) << "." << ((driverVersion%1000)/10) << endl;
			}
			
			if (cudaGetDeviceCount && cudaGetDeviceProperties && cudaMemGetInfo) {
				int deviceCount = 0;
				int result = cudaGetDeviceCount(&deviceCount);
				
				if (result == 0 && deviceCount > 0) {
					std::cout << "=== CUDA Device Information ===" << endl;
					std::cout << "Number of CUDA devices: " << deviceCount << endl;
					
					// 只查询内存信息，避免结构体兼容性问题
					size_t free_mem = 0, total_mem = 0;
					result = cudaMemGetInfo(&free_mem, &total_mem);
					if (result == 0) {
						std::cout << "  Total GPU memory: " << (total_mem / 1024 / 1024) << " MB" << endl;
						std::cout << "  Free GPU memory: " << (free_mem / 1024 / 1024) << " MB" << endl;
						std::cout << "  Used GPU memory: " << ((total_mem - free_mem) / 1024 / 1024) << " MB" << endl;
					} else {
						std::cout << "  Could not query GPU memory info" << endl;
					}
					std::cout << "===========================" << endl << endl;
				}
			}
			FreeLibrary(cudaRt);
		}
	} catch (...) {
		std::cout << "Note: Could not query GPU information, but continuing..." << endl;
	}
	
	try {
		// 创建测试会话选项
		Ort::SessionOptions test_options;
		
		// RTX 3060 12GB 专用优化配置
		OrtCUDAProviderOptions cuda_options{};
		
		// 内存配置 - 调整策略以支持CUBLAS初始化
		cuda_options.device_id = 0;
		cuda_options.gpu_mem_limit = 0;  // 0 表示不限制，让ONNX Runtime自行管理
		cuda_options.arena_extend_strategy = 1;  // 1 = 按需分配（kSameAsRequested）
		
		// 性能优化配置 - 只使用支持的选项
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;  // 启发式搜索，更快
		cuda_options.do_copy_in_default_stream = 1;  // 使用默认流进行拷贝
		
		std::cout << "=== GPU Memory Configuration ===" << endl;
		std::cout << "GPU Memory Limit: Unlimited (0 = let ONNX Runtime manage)" << endl;
		std::cout << "Arena Extend Strategy: On-demand (kSameAsRequested)" << endl;
		std::cout << "CUDNN Conv Algo: Heuristic" << endl;
		
		// 添加CUDA提供者
		std::cout << "=== Appending CUDA Execution Provider ===" << endl;
		test_options.AppendExecutionProvider_CUDA(cuda_options);
		std::cout << "=== CUDA Provider appended successfully ===" << endl;
		
		// 基本优化设置
		test_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);  // 基本优化，避免过度优化
		test_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);  // 顺序执行
		test_options.SetIntraOpNumThreads(1);  // GPU模式下使用单线程
		test_options.SetInterOpNumThreads(1);
		
		// 内存优化
		test_options.EnableMemPattern();
		test_options.EnableCpuMemArena();
		
		// 创建会话选项，但延迟会话创建直到模型路径设置
		std::cout << "=== Configuring GPU session options ===" << endl;
		std::cout << "=== About to check model path ===" << endl;
		
		// 检查模型路径是否已设置
		if (unetConfig.model_file_name != nullptr) {
			std::cout << "=== Model path is not null ===" << endl;
			std::wstring model_path = unetConfig.model_file_name;
			
			// 如果文件存在，尝试创建测试会话
			if (GetFileAttributesW(model_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
				std::cout << "=== DEBUG: About to create Ort::Session ===" << endl;
				std::wcout << L"Model file: " << model_path << endl;
				
				std::unique_ptr<Ort::Session> test_session;
				test_session = std::make_unique<Ort::Session>(env, model_path.c_str(), test_options);
				
				std::cout << "=== GPU test session created successfully ===" << endl;
				
				// 验证会话信息
				Ort::AllocatorWithDefaultOptions allocator;
				auto input_name = test_session->GetInputNameAllocated(0, allocator);
				auto output_name = test_session->GetOutputNameAllocated(0, allocator);
				
				std::cout << "Input name: " << input_name.get() << endl;
				std::cout << "Output name: " << output_name.get() << endl;
				
				// 获取输入形状信息
				auto input_shape = test_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
				std::cout << "Input shape: ";
				for (size_t i = 0; i < input_shape.size(); ++i) {
					std::cout << input_shape[i] << " ";
				}
				std::cout << endl;
			} else {
				std::cout << "=== Model file not found, skipping test session ===" << endl;
			}
		} else {
			std::cout << "=== Skipping test session creation (model path not set yet) ===" << endl;
		}
		
		// 保存成功的配置
		session_options = std::move(test_options);
		
		std::cout << "=== GPU initialization completed successfully ===" << endl;
		return true;
		
	} catch (const Ort::Exception& e) {
		std::cerr << "=== ONNX GPU Initialization Exception ===" << endl;
		std::cerr << "Error code: " << e.GetOrtErrorCode() << endl;
		std::cerr << "Error message: " << e.what() << endl;
		
		// 分析错误类型
		std::string error_msg = e.what();
		if (error_msg.find("CUDA") != std::string::npos || 
			error_msg.find("CUBLAS") != std::string::npos ||
			error_msg.find("CUDNN") != std::string::npos) {
			std::cerr << "=== CUDA/CUBLAS/CUDNN related error detected ===" << endl;
			
			// 尝试降低内存限制重试
			return retryGPUWithLowerMemory();
		}
		
		return false;
		
	} catch (const std::exception& e) {
		std::cerr << "=== Standard GPU Initialization Exception ===" << endl;
		std::cerr << "Error message: " << e.what() << endl;
		return false;
		
	} catch (...) {
		std::cerr << "=== Unknown GPU Initialization Exception ===" << endl;
		return false;
	}
}

// 新增GPU内存降级重试函数
bool DentalUnet::retryGPUWithLowerMemory()
{
	std::cout << "=== Retrying GPU initialization with lower memory settings ===" << endl;
	
	std::vector<size_t> memory_limits = {
		4ULL * 1024 * 1024 * 1024,  // 4GB
		3ULL * 1024 * 1024 * 1024,  // 3GB
		2ULL * 1024 * 1024 * 1024,  // 2GB
		1ULL * 1024 * 1024 * 1024   // 1GB
	};
	
	for (size_t mem_limit : memory_limits) {
		std::cout << "=== Trying with " << (mem_limit / (1024*1024*1024)) << "GB memory limit ===" << endl;
		
		try {
			Ort::SessionOptions retry_options;
			
			OrtCUDAProviderOptions cuda_options{};
			cuda_options.device_id = 0;
			cuda_options.gpu_mem_limit = mem_limit;
			cuda_options.arena_extend_strategy = 0;  // 固定内存分配
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;  // 默认算法
			cuda_options.do_copy_in_default_stream = 1;
			
			retry_options.AppendExecutionProvider_CUDA(cuda_options);
			retry_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);  // 禁用优化
			retry_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
			retry_options.SetIntraOpNumThreads(1);
			retry_options.SetInterOpNumThreads(1);
			
			// 测试会话创建 - 使用配置中的模型路径
			if (unetConfig.model_file_name != nullptr) {
				std::wstring model_path = unetConfig.model_file_name;
				if (GetFileAttributesW(model_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
					std::unique_ptr<Ort::Session> test_session;
					test_session = std::make_unique<Ort::Session>(env, model_path.c_str(), retry_options);
					std::cout << "=== GPU retry test session created ===" << endl;
				}
			}
			
			std::cout << "=== GPU retry successful with " << (mem_limit / (1024*1024*1024)) << "GB ===" << endl;
			
			// 保存成功的配置
			session_options = std::move(retry_options);
			return true;
			
		} catch (const Ort::Exception& e) {
			std::cerr << "=== Retry failed with " << (mem_limit / (1024*1024*1024)) << "GB: " << e.what() << endl;
			continue;
		} catch (...) {
			std::cerr << "=== Unknown error with " << (mem_limit / (1024*1024*1024)) << "GB ===" << endl;
			continue;
		}
	}
	
	std::cerr << "=== All GPU memory retry attempts failed ===" << endl;
	return false;
}


