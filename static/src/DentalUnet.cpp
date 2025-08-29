#include "DentalUnet.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <tuple>

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = true;

	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnUNetInference");
	std::vector<std::string> providers = Ort::GetAvailableProviders();
	use_gpu = true;

	for (const auto& provider : providers) {
		std::cout << "可用Provider: " << provider << std::endl;
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
}


DentalUnet::~DentalUnet()
{
}


DentalUnet *DentalUnet::CreateDentalUnet()
{
	DentalUnet *segUnetModel = new DentalUnet();

	std::cout << "CreateDentalUnet is done. "<<endl;

	return segUnetModel;
}

void  DentalUnet::setModelFns(const wchar_t* model_fn)
{
	std::cout << "[DEBUG] DentalUnet::setModelFns() called" << endl;
	
	if (model_fn == nullptr) {
		std::cerr << "[ERROR] Model filename is NULL!" << endl;
		return;
	}
	
	// 打印模型路径用于调试
	std::wcout << L"[DEBUG] Model path: " << model_fn << endl;
	
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

// 新增：参数设置接口实现
void DentalUnet::setPatchSize(int64_t x, int64_t y, int64_t z)
{
	unetConfig.patch_size = { x, y, z };
}

void DentalUnet::setNumClasses(int classes)
{
	unetConfig.num_classes = classes;
}

void DentalUnet::setInputChannels(int channels)
{
	unetConfig.input_channels = channels;
}

void DentalUnet::setTargetSpacing(float x, float y, float z)
{
	unetConfig.voxel_spacing = { x, y, z };
}

void DentalUnet::setTransposeSettings(int forward_x, int forward_y, int forward_z, 
                                    int backward_x, int backward_y, int backward_z)
{
	unetConfig.transpose_forward = { forward_x, forward_y, forward_z };
	unetConfig.transpose_backward = { backward_x, backward_y, backward_z };
}

void DentalUnet::setNormalizationType(const char* type)
{
	unetConfig.normalization_type = type;
}

void DentalUnet::setIntensityProperties(float mean, float std, float min_val, float max_val,
                                      float percentile_00_5, float percentile_99_5)
{
	unetConfig.mean_std_HU = { mean, std };
	unetConfig.min_max_HU = { min_val, max_val };
	unetConfig.percentile_00_5 = static_cast<double>(percentile_00_5);
	unetConfig.percentile_99_5 = static_cast<double>(percentile_99_5);
	unetConfig.mean = static_cast<double>(mean);
	unetConfig.std = static_cast<double>(std);
}

void DentalUnet::setUseMirroring(bool use_mirroring)
{
	unetConfig.use_mirroring = use_mirroring;
}

// 新增：JSON配置接口实现
bool DentalUnet::setConfigFromJsonString(const char* jsonContent)
{
	if (jsonContent == nullptr) {
		return false;
	}
	
	ModelConfig config;
	if (configParser.parseJsonString(std::string(jsonContent), config)) {
		// 应用配置到unetConfig
		unetConfig.patch_size.clear();
		if (config.patch_size.size() >= 3) {
			unetConfig.patch_size.push_back(config.patch_size[0]);
			unetConfig.patch_size.push_back(config.patch_size[1]);
			unetConfig.patch_size.push_back(config.patch_size[2]);
		}
		
		unetConfig.voxel_spacing.clear();
		if (config.target_spacing.size() >= 3) {
			unetConfig.voxel_spacing.push_back(config.target_spacing[0]);
			unetConfig.voxel_spacing.push_back(config.target_spacing[1]);
			unetConfig.voxel_spacing.push_back(config.target_spacing[2]);
		}
		
		unetConfig.transpose_forward.clear();
		if (config.transpose_forward.size() >= 3) {
			unetConfig.transpose_forward.push_back(config.transpose_forward[0]);
			unetConfig.transpose_forward.push_back(config.transpose_forward[1]);
			unetConfig.transpose_forward.push_back(config.transpose_forward[2]);
		}
		
		unetConfig.transpose_backward.clear();
		if (config.transpose_backward.size() >= 3) {
			unetConfig.transpose_backward.push_back(config.transpose_backward[0]);
			unetConfig.transpose_backward.push_back(config.transpose_backward[1]);
			unetConfig.transpose_backward.push_back(config.transpose_backward[2]);
		}
		
		unetConfig.num_classes = config.num_classes;
		unetConfig.input_channels = config.num_input_channels;
		unetConfig.normalization_type = config.normalization_scheme;
		unetConfig.use_mirroring = config.use_tta;
		
		// 设置intensity properties
		unetConfig.mean_std_HU.clear();
		unetConfig.mean_std_HU.push_back(config.mean);
		unetConfig.mean_std_HU.push_back(config.std);
		
		unetConfig.min_max_HU.clear();
		unetConfig.min_max_HU.push_back(config.min_val);
		unetConfig.min_max_HU.push_back(config.max_val);
		
		// 添加直接访问的intensity properties到config（转换为double）
		unetConfig.mean = static_cast<double>(config.mean);
		unetConfig.std = static_cast<double>(config.std);
		unetConfig.percentile_00_5 = static_cast<double>(config.percentile_00_5);
		unetConfig.percentile_99_5 = static_cast<double>(config.percentile_99_5);
		
		// 添加归一化相关参数
		unetConfig.use_mask_for_norm = config.use_mask_for_norm;
		
		// 调试输出确认配置
		std::cout << "[DEBUG] Configuration loaded from JSON:" << endl;
		std::cout << "  - num_classes: " << unetConfig.num_classes << endl;
		std::cout << "  - patch_size: [" << unetConfig.patch_size[0] << ", " 
		          << unetConfig.patch_size[1] << ", " << unetConfig.patch_size[2] << "]" << endl;
		std::cout << "  - normalization_type: " << unetConfig.normalization_type << endl;
		std::cout << "  - use_mask_for_norm: " << (unetConfig.use_mask_for_norm ? "true" : "false") << endl;
		std::cout << "  - mean: " << unetConfig.mean << ", std: " << unetConfig.std << endl;
		
		return true;
	}
	
	return false;
}


void  DentalUnet::setDnnOptions()
{
	//??????????????????????
}


void  DentalUnet::setAlgParameter()
{
	//????????????????
}


void DentalUnet::setOutputPaths(const wchar_t* preprocessPath, const wchar_t* modelOutputPath, const wchar_t* postprocessPath)
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


AI_INT  DentalUnet::initializeOnnxruntimeInstances()
{
	std::cout << "[DEBUG] Initializing ONNX Runtime instances..." << endl;
	
	if (use_gpu) {
		std::cout << "[DEBUG] GPU mode enabled, configuring CUDA provider..." << endl;
		try {
			//OrtCUDAProviderOptions cuda_options;
			//cuda_options.device_id = 0;  // 设置 GPU 设备 ID
			//session_options.AppendExecutionProvider_CUDA(cuda_options);

			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
			std::cout << "[DEBUG] CUDA provider added successfully" << endl;
		} catch (const Ort::Exception& e) {
			std::cerr << "[WARNING] Failed to add CUDA provider: " << e.what() << endl;
			std::cerr << "[WARNING] Falling back to CPU" << endl;
			use_gpu = false;
		}
	} else {
		std::cout << "[DEBUG] Using CPU mode" << endl;
	}
	
	// 设置线程数
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	std::cout << "[DEBUG] Thread settings: IntraOp=1, InterOp=1" << endl;

	// 创建会话
	//semantic_seg_session_ptr = std::make_unique<Ort::Session>(env, unetConfig.model_file_name.c_str(), session_options);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::setInput(AI_DataInfo *srcData)
{
	std::cout << "[DEBUG] DentalUnet::setInput() called" << endl;
	
	// 验证输入
	if (srcData == nullptr) {
		std::cerr << "[ERROR] srcData is NULL!" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	
	if (srcData->ptr_Data == nullptr) {
		std::cerr << "[ERROR] srcData->ptr_Data is NULL!" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //单位: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //单位: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //单位: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //单位: mm
	
	std::cout << "[DEBUG] Input volume dimensions: " << Width0 << "x" << Height0 << "x" << Depth0 << endl;
	std::cout << "[DEBUG] Voxel spacing: X=" << voxelSpacingX << ", Y=" << voxelSpacingY << ", Z=" << voxelSpacingZ << endl;
	
	// 保存origin信息到元数据
	imageMetadata.origin[0] = srcData->OriginX;
	imageMetadata.origin[1] = srcData->OriginY;
	imageMetadata.origin[2] = srcData->OriginZ;
	
	// 保存spacing信息到元数据
	imageMetadata.spacing[0] = voxelSpacingX;
	imageMetadata.spacing[1] = voxelSpacingY;
	imageMetadata.spacing[2] = voxelSpacingZ;
	
	std::cout << "[DEBUG] Origin: X=" << imageMetadata.origin[0] 
	         << ", Y=" << imageMetadata.origin[1] 
	         << ", Z=" << imageMetadata.origin[2] << endl;

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //输入体积太小�

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return DentalCbctSegAI_STATUS_VOLUME_LARGE; //输入体积太大

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume太小�
		return DentalCbctSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing太小�
		return DentalCbctSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.1f || voxelSpacingX > 1.1f || voxelSpacingY > 1.1f || voxelSpacingZ > 1.1f)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //voxelSpacing太大

	// 创建CImg对象并复制数据
	//RAI: 右-前-上坐标系，与医学图像标准一致
	input_cbct_volume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(input_cbct_volume.data(), srcData->ptr_Data, volSize);

	// 移除统计计算 - 将在预处理管道中的正确位置计算
	// intensity_mean和intensity_std将在crop后计算，或使用JSON配置中的值

	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth
	
	// 读取原始spacing（如果提供了的话）
	if (srcData->OriginalVoxelSpacingX > 0 && srcData->OriginalVoxelSpacingY > 0 && srcData->OriginalVoxelSpacingZ > 0) {
		original_voxel_spacing = { srcData->OriginalVoxelSpacingX, srcData->OriginalVoxelSpacingY, srcData->OriginalVoxelSpacingZ };
		std::cout << "[DEBUG] Original voxel spacing: X=" << srcData->OriginalVoxelSpacingX 
		          << ", Y=" << srcData->OriginalVoxelSpacingY 
		          << ", Z=" << srcData->OriginalVoxelSpacingZ << endl;
	} else {
		// 如果没有提供原始spacing，则使用当前spacing作为原始spacing
		original_voxel_spacing = input_voxel_spacing;
		std::cout << "[DEBUG] No original spacing provided, using current spacing as original" << endl;
	}

	// 统计信息将在预处理流水线中计算
	std::cout << "Input volume loaded successfully" << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}

// 简单的3D binary_fill_holes实现（匹配scipy.ndimage.binary_fill_holes）
void binary_fill_holes_3d(CImg<bool>& mask) {
	// 使用flood fill从边界开始，标记所有外部背景
	// 未被标记的背景即为内部孔洞
	
	int width = mask.width();
	int height = mask.height();
	int depth = mask.depth();
	
	// 创建visited标记
	CImg<bool> visited(width, height, depth, 1, false);
	std::queue<std::tuple<int, int, int>> queue;
	
	// 从所有边界的背景点开始flood fill
	// X边界 (x=0 和 x=width-1)
	for (int y = 0; y < height; y++) {
		for (int z = 0; z < depth; z++) {
			if (!mask(0, y, z) && !visited(0, y, z)) {
				queue.push(std::make_tuple(0, y, z));
				visited(0, y, z) = true;
			}
			if (!mask(width-1, y, z) && !visited(width-1, y, z)) {
				queue.push(std::make_tuple(width-1, y, z));
				visited(width-1, y, z) = true;
			}
		}
	}
	
	// Y边界 (y=0 和 y=height-1)
	for (int x = 0; x < width; x++) {
		for (int z = 0; z < depth; z++) {
			if (!mask(x, 0, z) && !visited(x, 0, z)) {
				queue.push(std::make_tuple(x, 0, z));
				visited(x, 0, z) = true;
			}
			if (!mask(x, height-1, z) && !visited(x, height-1, z)) {
				queue.push(std::make_tuple(x, height-1, z));
				visited(x, height-1, z) = true;
			}
		}
	}
	
	// Z边界 (z=0 和 z=depth-1)
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (!mask(x, y, 0) && !visited(x, y, 0)) {
				queue.push(std::make_tuple(x, y, 0));
				visited(x, y, 0) = true;
			}
			if (!mask(x, y, depth-1) && !visited(x, y, depth-1)) {
				queue.push(std::make_tuple(x, y, depth-1));
				visited(x, y, depth-1) = true;
			}
		}
	}
	
	// BFS找到所有连接到边界的背景点
	while (!queue.empty()) {
		auto [x, y, z] = queue.front();
		queue.pop();
		
		// 检查6个邻居（3D中的6连通）
		int dx[] = {-1, 1, 0, 0, 0, 0};
		int dy[] = {0, 0, -1, 1, 0, 0};
		int dz[] = {0, 0, 0, 0, -1, 1};
		
		for (int i = 0; i < 6; i++) {
			int nx = x + dx[i];
			int ny = y + dy[i];
			int nz = z + dz[i];
			
			// 检查边界条件
			if (nx >= 0 && nx < width && 
			    ny >= 0 && ny < height && 
			    nz >= 0 && nz < depth) {
				// 如果是背景且未访问过
				if (!mask(nx, ny, nz) && !visited(nx, ny, nz)) {
					queue.push(std::make_tuple(nx, ny, nz));
					visited(nx, ny, nz) = true;
				}
			}
		}
	}
	
	// 填充所有内部孔洞（未被访问的背景点）
	int filled_count = 0;
	cimg_forXYZ(mask, x, y, z) {
		if (!mask(x, y, z) && !visited(x, y, z)) {
			mask(x, y, z) = true;  // 填充孔洞
			filled_count++;
		}
	}
	
	std::cout << "[DEBUG] binary_fill_holes filled " << filled_count << " pixels" << endl;
}

// 实现crop_to_nonzero函数，与Python版本对齐
CImg<short> DentalUnet::crop_to_nonzero(const CImg<short>& input, CropBBox& bbox) {
	// 找到非零区域的边界
	bbox.x_min = input.width();
	bbox.x_max = -1;
	bbox.y_min = input.height();
	bbox.y_max = -1;
	bbox.z_min = input.depth();
	bbox.z_max = -1;
	
	// 创建非零mask（与Python的nonzero_mask对应）
	CImg<bool> nonzero_mask(input.width(), input.height(), input.depth(), 1, false);
	
	// 扫描整个体积找到非零区域
	cimg_forXYZ(input, x, y, z) {
		if (input(x, y, z) != 0) {
			nonzero_mask(x, y, z) = true;
		}
	}
	
	// 应用binary_fill_holes（与Python的scipy.ndimage.binary_fill_holes一致）
	// 暂时禁用以测试是否是fill hole导致的差异
	// binary_fill_holes_3d(nonzero_mask);
	std::cout << "[DEBUG] binary_fill_holes DISABLED for testing" << endl;
	
	// 重新计算bbox（基于填充后的mask）
	cimg_forXYZ(input, x, y, z) {
		if (nonzero_mask(x, y, z)) {
			if (x < bbox.x_min) bbox.x_min = x;
			if (x > bbox.x_max) bbox.x_max = x;
			if (y < bbox.y_min) bbox.y_min = y;
			if (y > bbox.y_max) bbox.y_max = y;
			if (z < bbox.z_min) bbox.z_min = z;
			if (z > bbox.z_max) bbox.z_max = z;
		}
	}
	
	// 如果没有找到非零像素，返回原图像
	if (bbox.x_max == -1) {
		std::cout << "[WARNING] No non-zero pixels found, using full volume" << endl;
		bbox.x_min = 0; bbox.x_max = input.width() - 1;
		bbox.y_min = 0; bbox.y_max = input.height() - 1;
		bbox.z_min = 0; bbox.z_max = input.depth() - 1;
		std::cout << "[WARNING] Full volume bbox set to: X[0:" << bbox.x_max 
		          << "], Y[0:" << bbox.y_max << "], Z[0:" << bbox.z_max << "]" << endl;
		
		// 创建全为0的seg_mask（因为全是背景）
		seg_mask = CImg<short>(input.width(), input.height(), input.depth(), 1, -1);
		return input;
	}
	
	// 验证bbox是否合理
	if (bbox.x_min > bbox.x_max || bbox.y_min > bbox.y_max || bbox.z_min > bbox.z_max) {
		std::cout << "[ERROR] Invalid bbox detected after computation!" << endl;
		std::cout << "[ERROR] Bbox values: X[" << bbox.x_min << ":" << bbox.x_max 
		          << "], Y[" << bbox.y_min << ":" << bbox.y_max 
		          << "], Z[" << bbox.z_min << ":" << bbox.z_max << "]" << endl;
		// 重置为全图像
		bbox.x_min = 0; bbox.x_max = input.width() - 1;
		bbox.y_min = 0; bbox.y_max = input.height() - 1;
		bbox.z_min = 0; bbox.z_max = input.depth() - 1;
		
		// 创建seg_mask
		seg_mask = CImg<short>(input.width(), input.height(), input.depth(), 1);
		cimg_forXYZ(seg_mask, x, y, z) {
			seg_mask(x, y, z) = (input(x, y, z) != 0) ? 0 : -1;
		}
		return input;
	}
	
	std::cout << "[DEBUG] Crop bbox: X[" << bbox.x_min << ":" << bbox.x_max 
	          << "], Y[" << bbox.y_min << ":" << bbox.y_max 
	          << "], Z[" << bbox.z_min << ":" << bbox.z_max << "]" << endl;
	          
	// 执行裁剪
	CImg<short> cropped = input.get_crop(bbox.x_min, bbox.y_min, bbox.z_min, 
	                                     bbox.x_max, bbox.y_max, bbox.z_max);
	
	// 裁剪nonzero_mask
	CImg<bool> cropped_mask = nonzero_mask.get_crop(bbox.x_min, bbox.y_min, bbox.z_min,
	                                                bbox.x_max, bbox.y_max, bbox.z_max);
	
	// 创建seg_mask（与Python的seg对应）
	// Python: seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label=-1))
	seg_mask = CImg<short>(cropped.width(), cropped.height(), cropped.depth(), 1);
	cimg_forXYZ(cropped, x, y, z) {
		// 使用填充后的mask：mask区域设为0，背景设为-1（与Python一致）
		seg_mask(x, y, z) = cropped_mask(x, y, z) ? 0 : -1;
	}
	
	// 统计seg_mask信息用于调试
	int seg_zero_count = 0, seg_neg_count = 0;
	cimg_forXYZ(seg_mask, x, y, z) {
		if (seg_mask(x, y, z) == 0) seg_zero_count++;
		else if (seg_mask(x, y, z) == -1) seg_neg_count++;
	}
	std::cout << "[DEBUG] seg_mask created: " << seg_zero_count << " pixels with value 0 (nonzero region), " 
	          << seg_neg_count << " pixels with value -1 (background)" << endl;
	                                     
	std::cout << "[DEBUG] Shape after cropping: " << cropped.width() << "x" 
	          << cropped.height() << "x" << cropped.depth() << endl;
	          
	return cropped;
}

AI_INT  DentalUnet::performInference(AI_DataInfo *srcData)
{
	int input_status = setInput(srcData);
	std::cout << "input_status: " << input_status << endl;
	if (input_status != DentalCbctSegAI_STATUS_SUCCESS)
		return input_status;

	// 按照Python版本的顺序进行预处理：
	// 1. 转置
	std::cout << "[DEBUG] Step 1: Transpose" << endl;
	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_forward);
	transposed_input_voxel_spacing.clear();
	transposed_original_voxel_spacing.clear();
	for (int i = 0; i < 3; ++i) {
		transposed_input_voxel_spacing.push_back(input_voxel_spacing[unetConfig.transpose_forward[i]]);
		transposed_original_voxel_spacing.push_back(original_voxel_spacing[unetConfig.transpose_forward[i]]);
	}
	
	// 2. 裁剪到非零区域
	std::cout << "[DEBUG] Step 2: Crop to non-zero region" << endl;
	std::cout << "[DEBUG] Shape before cropping: " << input_cbct_volume.width() 
	          << "x" << input_cbct_volume.height() << "x" << input_cbct_volume.depth() << endl;
	CImg<short> cropped_volume = crop_to_nonzero(input_cbct_volume, crop_bbox);
	
	// 3. 在裁剪后的数据上计算或使用配置的归一化参数
	std::cout << "[DEBUG] Step 3: Determine intensity statistics strategy" << endl;
	std::cout << "[DEBUG] Normalization type: " << unetConfig.normalization_type << endl;
	
	// 根据归一化类型决定是否使用JSON配置的值
	if (unetConfig.normalization_type == "ZScoreNormalization") {
		// ZScoreNormalization总是动态计算mean和std（MRI等模态）
		std::cout << "[DEBUG] ZScoreNormalization detected - will dynamically calculate mean/std" << endl;
		
		if (unetConfig.use_mask_for_norm) {
			// 在mask区域动态计算
			std::cout << "[DEBUG] Will calculate on mask region in normalization step" << endl;
			// 这里暂不计算，将在segModelInfer的归一化步骤中动态计算
			intensity_mean = 0.0;  // 占位值
			intensity_std = 1.0;   // 占位值
		} else {
			// 在整个裁剪后的数据上动态计算
			intensity_mean = cropped_volume.mean();  // CImg::mean()返回double
			double var = cropped_volume.variance();  // CImg::variance()返回double
			intensity_std = std::sqrt(var);
			if (intensity_std < 1e-8) intensity_std = 1e-8;  // 匹配Python的max(std, 1e-8)
			std::cout << "[DEBUG] Dynamically calculated global stats: mean=" 
			          << intensity_mean << ", std=" << intensity_std << endl;
		}
	} else if (unetConfig.normalization_type == "CTNormalization" || 
	           unetConfig.normalization_type == "CT" || 
	           unetConfig.normalization_type == "ct") {
		// CTNormalization使用JSON配置的值（CT等标准化模态）
		std::cout << "[DEBUG] CTNormalization detected - using JSON intensity_properties" << endl;
		intensity_mean = unetConfig.mean;
		intensity_std = unetConfig.std;
		std::cout << "[DEBUG] Using configured intensity properties: mean=" 
		          << intensity_mean << ", std=" << intensity_std << endl;
	} else {
		// 默认行为：动态计算
		std::cout << "[DEBUG] Unknown normalization type, using dynamic calculation" << endl;
		intensity_mean = cropped_volume.mean();  // CImg::mean()返回double
		double var = cropped_volume.variance();  // CImg::variance()返回double
		intensity_std = std::sqrt(var);
		if (intensity_std < 1e-8) intensity_std = 1e-8;
		std::cout << "[DEBUG] Dynamically calculated stats: mean=" 
		          << intensity_mean << ", std=" << intensity_std << endl;
	}

	// 4. 调用推理（包含归一化和重采样）
	std::cout << "[DEBUG] Step 4: Model inference (normalize -> resample -> infer)" << endl;
	int infer_status = segModelInfer(unetConfig, cropped_volume);
	std::cout << "infer_status: " << infer_status << endl;

	// 不在这里恢复转置，将在getSegMask的最后步骤中处理
	// 保持数据在转置后的坐标系中，以便正确处理裁剪恢复

	return infer_status;
}


AI_INT  DentalUnet::segModelInfer(nnUNetConfig config, CImg<short> input_volume)
{

	if (transposed_input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// bool is_volume_scaled = false;  // 注释掉条件判断，改为始终缩放
	bool is_volume_scaled = true;  // 使用与Python相同的逻辑：始终进行缩放
	////input_voxel_spacing = {voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth 
	std::vector<int64_t> input_size = { input_volume.width(), input_volume.height(), input_volume.depth()};
	std::vector<int64_t> output_size;
	float scaled_factor = 1.f;
	
	std::cout << "[DEBUG] Scaling calculation:" << endl;
	std::cout << "  Current spacing: [" << transposed_input_voxel_spacing[0] 
	          << ", " << transposed_input_voxel_spacing[1] 
	          << ", " << transposed_input_voxel_spacing[2] << "]" << endl;
	std::cout << "  Original spacing: [" << transposed_original_voxel_spacing[0] 
	          << ", " << transposed_original_voxel_spacing[1] 
	          << ", " << transposed_original_voxel_spacing[2] << "]" << endl;
	std::cout << "  Target spacing: [" << config.voxel_spacing[0] 
	          << ", " << config.voxel_spacing[1] 
	          << ", " << config.voxel_spacing[2] << "]" << endl;
	
	for (int i = 0; i < 3; ++i) {  // 遍历三个维度
		// 使用原始spacing计算缩放因子，与Python保持一致
		scaled_factor = transposed_original_voxel_spacing[i] / config.voxel_spacing[i];
		int scaled_sz = std::round(input_size[i] * scaled_factor);
		
		// 注释掉原有的条件判断逻辑，保留以供参考
		// if (scaled_factor < 0.9f || scaled_factor > 1.1f || scaled_sz < config.patch_size[i])
		//     is_volume_scaled = true;

		if (scaled_sz < config.patch_size[i])
			scaled_sz = config.patch_size[i];

		output_size.push_back(static_cast<int64_t>(scaled_sz));
		
		// 输出每个轴的缩放信息
		std::cout << "  Axis " << i << ":" << endl;
		std::cout << "    original_spacing: " << transposed_original_voxel_spacing[i] << endl;
		std::cout << "    current_spacing: " << transposed_input_voxel_spacing[i] << endl;
		std::cout << "    target_spacing: " << config.voxel_spacing[i] << endl;
		std::cout << "    scaled_factor: " << scaled_factor << " (" << transposed_original_voxel_spacing[i] 
		          << " / " << config.voxel_spacing[i] << ")" << endl;
		std::cout << "    size: " << input_size[i] << " -> " << scaled_sz << endl;
	}

	std::cout << "[DEBUG] Original size: " << input_size[0] << "x" << input_size[1] << "x" << input_size[2] << endl;
	std::cout << "[DEBUG] Scaled size: " << output_size[0] << "x" << output_size[1] << "x" << output_size[2] << endl;
	std::cout << "[DEBUG] Scale factors: " << transposed_original_voxel_spacing[0]/config.voxel_spacing[0] 
	          << ", " << transposed_original_voxel_spacing[1]/config.voxel_spacing[1] 
	          << ", " << transposed_original_voxel_spacing[2]/config.voxel_spacing[2] << endl;

	// 按照Python版本的顺序：先归一化，后重采样
	
	// Step 1: 归一化（在原始分辨率上进行）
	std::cout << "[DEBUG] Step 1: Normalization (before resampling)" << endl;
	CImg<float> normalized_volume;
	normalized_volume.assign(input_volume);  // 转换为float
	
	// 保存归一化前的数据
	if (saveIntermediateResults) {
		savePreprocessedData(normalized_volume, L"before_normalization");
	}
	
	// 输出归一化前的统计信息
	std::cout << "[DEBUG] === BEFORE NORMALIZATION ===" << endl;
	std::cout << "[DEBUG] Data shape: " << normalized_volume.width() << "x" 
	          << normalized_volume.height() << "x" << normalized_volume.depth() << endl;
	std::cout << "[DEBUG] Data min: " << normalized_volume.min() << endl;
	std::cout << "[DEBUG] Data max: " << normalized_volume.max() << endl;
	std::cout << "[DEBUG] Data mean: " << normalized_volume.mean() << endl;
	std::cout << "[DEBUG] Data std: " << std::sqrt(normalized_volume.variance()) << endl;
	
	// 输出seg_mask的维度以确认它与normalized_volume匹配
	std::cout << "[DEBUG] seg_mask shape: " << seg_mask.width() << "x" 
	          << seg_mask.height() << "x" << seg_mask.depth() << endl;
	
	// 统计seg_mask的值分布
	int seg_positive = 0, seg_negative = 0, seg_zero = 0;
	cimg_forXYZ(seg_mask, x, y, z) {
		if (seg_mask(x, y, z) > 0) seg_positive++;
		else if (seg_mask(x, y, z) < 0) seg_negative++;
		else seg_zero++;
	}
	std::cout << "[DEBUG] seg_mask value distribution: positive=" << seg_positive 
	          << ", zero=" << seg_zero << ", negative=" << seg_negative << endl;
	
	// 计算非零区域的统计（用于对比）
	// 使用seg_mask判断前景区域，与Python一致
	int nonzero_count = 0;
	double nonzero_sum = 0.0;  // 使用double提高精度
	cimg_forXYZ(normalized_volume, x, y, z) {
		// seg_mask >= 0 表示前景区域（包括值为0的前景像素）
		if (seg_mask(x, y, z) >= 0) {
			nonzero_sum += normalized_volume(x, y, z);
			nonzero_count++;
		}
	}
	if (nonzero_count > 0) {
		double nonzero_mean = nonzero_sum / nonzero_count;
		double nonzero_var = 0.0;
		cimg_forXYZ(normalized_volume, x, y, z) {
			if (seg_mask(x, y, z) >= 0) {
				double diff = normalized_volume(x, y, z) - nonzero_mean;
				nonzero_var += diff * diff;
			}
		}
		double nonzero_std = std::sqrt(nonzero_var / nonzero_count);
		std::cout << "[DEBUG] Non-zero region stats (using seg_mask>=0): mean=" << nonzero_mean 
		          << ", std=" << nonzero_std << ", pixels=" << nonzero_count << endl;
		
		// 也计算使用简单>0判断的统计作为对比
		int simple_count = 0;
		double simple_sum = 0.0;  // 使用double提高精度
		cimg_forXYZ(normalized_volume, x, y, z) {
			if (normalized_volume(x, y, z) > 0) {
				simple_sum += normalized_volume(x, y, z);
				simple_count++;
			}
		}
		if (simple_count > 0) {
			double simple_mean = simple_sum / simple_count;
			double simple_var = 0.0;
			cimg_forXYZ(normalized_volume, x, y, z) {
				if (normalized_volume(x, y, z) > 0) {
					double diff = normalized_volume(x, y, z) - simple_mean;
					simple_var += diff * diff;
				}
			}
			double simple_std = std::sqrt(simple_var / simple_count);
			std::cout << "[DEBUG] Non-zero region stats (using value>0): mean=" << simple_mean 
			          << ", std=" << simple_std << ", pixels=" << simple_count << endl;
		}
	}
	
	// 输出特定位置的值（用于精确对比）
	if (normalized_volume.depth() > 2 && normalized_volume.height() > 162 && normalized_volume.width() > 44) {
		std::cout << "[DEBUG] Value at (44,162,2) before norm: " << normalized_volume(44, 162, 2) << endl;
		std::cout << "[DEBUG] seg_mask at (44,162,2): " << seg_mask(44, 162, 2) << endl;
	}
	
	// 输出一些采样点的值用于对比
	std::cout << "[DEBUG] Sample points before normalization:" << endl;
	for (int i = 0; i < 5 && i < normalized_volume.depth(); i++) {
		for (int j = 0; j < 5 && j < normalized_volume.height(); j++) {
			for (int k = 0; k < 5 && k < normalized_volume.width(); k++) {
				if (normalized_volume(k, j, i) != 0) {
					std::cout << "  (" << k << "," << j << "," << i << "): value=" 
					          << normalized_volume(k, j, i) << ", seg=" << seg_mask(k, j, i) << endl;
					break;
				}
			}
		}
	}
	
	std::cout << "[DEBUG] ================================" << endl;
	
	// 执行归一化
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
		std::cout << "[DEBUG] Using CT Normalization" << endl;
		CTNormalization(normalized_volume, config);
		break;
	case 20:
		std::cout << "[DEBUG] Using Z-Score Normalization" << endl;
		if (config.use_mask_for_norm) {
			std::cout << "[DEBUG] Using mask-based normalization (dynamic calculation)" << endl;
			// 使用seg_mask创建mask（与Python一致：seg >= 0表示非零区域）
			// Python: mask = seg[0] >= 0
			CImg<bool> mask(normalized_volume.width(), normalized_volume.height(), normalized_volume.depth());
			cimg_forXYZ(normalized_volume, x, y, z) {
				// seg_mask中：0表示非零区域，-1表示背景
				// 所以seg_mask >= 0就是非零区域
				mask(x, y, z) = (seg_mask(x, y, z) >= 0);
			}
			
			// 在mask区域动态计算mean和std（匹配Python行为）
			double mask_mean = 0.0;  // 使用double提高精度
			double mask_std = 0.0;
			int mask_count = 0;
			
			// 计算mask区域的mean
			cimg_forXYZ(normalized_volume, x, y, z) {
				if (mask(x, y, z)) {
					mask_mean += normalized_volume(x, y, z);
					mask_count++;
				}
			}
			
			if (mask_count > 0) {
				mask_mean /= mask_count;
				
				// 计算mask区域的std
				cimg_forXYZ(normalized_volume, x, y, z) {
					if (mask(x, y, z)) {
						double diff = normalized_volume(x, y, z) - mask_mean;
						mask_std += diff * diff;
					}
				}
				mask_std = std::sqrt(mask_std / mask_count);
				if (mask_std < 1e-8) mask_std = 1e-8;  // 匹配Python的max(std, 1e-8)
				
				std::cout << "[DEBUG] Dynamically calculated mask-based stats: mean=" << mask_mean 
				          << ", std=" << mask_std << ", mask_pixels=" << mask_count << endl;
				
				// 验证seg_mask的统计信息
				int seg_positive = 0, seg_negative = 0;
				cimg_forXYZ(seg_mask, x, y, z) {
					if (seg_mask(x, y, z) >= 0) seg_positive++;
					else seg_negative++;
				}
				std::cout << "[DEBUG] seg_mask stats: " << seg_positive << " pixels >= 0, " 
				          << seg_negative << " pixels < 0" << endl;
				
				// 只对mask区域进行归一化，背景设为0
				cimg_forXYZ(normalized_volume, x, y, z) {
					if (mask(x, y, z)) {
						normalized_volume(x, y, z) = (normalized_volume(x, y, z) - mask_mean) / mask_std;
					} else {
						normalized_volume(x, y, z) = 0.0f;
					}
				}
				std::cout << "[DEBUG] Normalized " << mask_count << " non-zero pixels, set " 
				          << (normalized_volume.size() - mask_count) << " background pixels to 0" << endl;
			} else {
				std::cout << "[WARNING] No non-zero pixels found for mask normalization" << endl;
			}
		} else {
			// 传统的全局归一化
			std::cout << "[DEBUG] Using global normalization" << endl;
			std::cout << "[DEBUG] intensity_mean: " << intensity_mean << ", intensity_std: " << intensity_std << endl;
			normalized_volume -= intensity_mean;
			normalized_volume /= intensity_std;
		}
		break;
	default:
		std::cout << "[DEBUG] Using default Z-Score Normalization" << endl;
		normalized_volume -= intensity_mean;
		normalized_volume /= intensity_std;
		break;
	}
	
	// 输出归一化后的详细统计信息
	std::cout << "[DEBUG] === AFTER NORMALIZATION ===" << endl;
	std::cout << "[DEBUG] Data shape: " << normalized_volume.width() << "x" 
	          << normalized_volume.height() << "x" << normalized_volume.depth() << endl;
	std::cout << "[DEBUG] Data min: " << normalized_volume.min() << endl;
	std::cout << "[DEBUG] Data max: " << normalized_volume.max() << endl;
	std::cout << "[DEBUG] Data mean: " << normalized_volume.mean() << endl;
	std::cout << "[DEBUG] Data std: " << std::sqrt(normalized_volume.variance()) << endl;
	
	// 计算非零区域的统计（用于对比）
	// 使用seg_mask判断前景区域，与Python一致
	int nonzero_count_after = 0;
	double nonzero_sum_after = 0.0;  // 使用double提高精度
	cimg_forXYZ(normalized_volume, x, y, z) {
		// seg_mask >= 0 表示前景区域
		if (seg_mask(x, y, z) >= 0) {
			nonzero_sum_after += normalized_volume(x, y, z);
			nonzero_count_after++;
		}
	}
	if (nonzero_count_after > 0) {
		double nonzero_mean_after = nonzero_sum_after / nonzero_count_after;
		double nonzero_var_after = 0.0;
		cimg_forXYZ(normalized_volume, x, y, z) {
			if (seg_mask(x, y, z) >= 0) {
				double diff = normalized_volume(x, y, z) - nonzero_mean_after;
				nonzero_var_after += diff * diff;
			}
		}
		double nonzero_std_after = std::sqrt(nonzero_var_after / nonzero_count_after);
		std::cout << "[DEBUG] Non-zero region stats after norm: mean=" << nonzero_mean_after 
		          << ", std=" << nonzero_std_after << ", pixels=" << nonzero_count_after << endl;
	}
	
	// 输出特定位置的值（用于精确对比）
	if (normalized_volume.depth() > 2 && normalized_volume.height() > 162 && normalized_volume.width() > 44) {
		std::cout << "[DEBUG] Value at (44,162,2) after norm: " << normalized_volume(44, 162, 2) << endl;
	}
	
	// 输出一些采样点的归一化后值用于对比
	std::cout << "[DEBUG] Sample points after normalization:" << endl;
	for (int i = 0; i < 5 && i < normalized_volume.depth(); i++) {
		for (int j = 0; j < 5 && j < normalized_volume.height(); j++) {
			for (int k = 0; k < 5 && k < normalized_volume.width(); k++) {
				if (std::abs(normalized_volume(k, j, i)) > 0.01) {  // 归一化后可能接近0
					std::cout << "  (" << k << "," << j << "," << i << "): value=" 
					          << normalized_volume(k, j, i) << endl;
					break;
				}
			}
		}
	}
	std::cout << "[DEBUG] ================================" << endl;

	// Step 2: 重采样（在归一化后进行）
	std::cout << "[DEBUG] Step 2: Resampling (after normalization)" << endl;
	CImg<float> scaled_input_volume;
	if (is_volume_scaled) {
		// 使用三次插值（5）而不是线性插值（3）以匹配Python的order=3
		// CImg插值模式: 0=最近邻, 1=线性, 2=移动平均, 3=线性, 5=三次(cubic)
		scaled_input_volume = normalized_volume.get_resize(output_size[0], output_size[1], output_size[2], -100, 5);
		std::cout << "Resampled from " << normalized_volume.width() << "x" << normalized_volume.height() << "x" << normalized_volume.depth()
		          << " to " << scaled_input_volume.width() << "x" << scaled_input_volume.height() << "x" << scaled_input_volume.depth() << endl;
		std::cout << "[DEBUG] Using cubic interpolation (mode=5) to match Python's order=3" << endl;
	} else {
		scaled_input_volume.assign(normalized_volume);
	}

	std::cout << "final_preprocessed_volume depth: " << scaled_input_volume.depth() << endl;
	std::cout << "final_preprocessed_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "final_preprocessed_volume variance: " << scaled_input_volume.variance() << endl;
	
	// 添加详细的调试输出以便与Python版本对比
	std::cout << "[DEBUG] === Preprocessing Summary ===" << endl;
	std::cout << "[DEBUG] Normalization type: " << config.normalization_type << endl;
	std::cout << "[DEBUG] Use mask for norm: " << (config.use_mask_for_norm ? "true" : "false") << endl;
	if (!config.use_mask_for_norm) {
		std::cout << "[DEBUG] Intensity mean used: " << intensity_mean << endl;
		std::cout << "[DEBUG] Intensity std used: " << intensity_std << endl;
	} else {
		std::cout << "[DEBUG] Intensity stats: dynamically calculated on mask region" << endl;
	}
	std::cout << "[DEBUG] Final volume shape: " << scaled_input_volume.width() << "x" 
	          << scaled_input_volume.height() << "x" << scaled_input_volume.depth() << endl;
	std::cout << "[DEBUG] Final volume min: " << scaled_input_volume.min() << endl;
	std::cout << "[DEBUG] Final volume max: " << scaled_input_volume.max() << endl;
	std::cout << "[DEBUG] Final volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "[DEBUG] Final volume std: " << std::sqrt(scaled_input_volume.variance()) << endl;
	
	// 输出一些采样点的值以便详细对比
	if (scaled_input_volume.size() > 0) {
		int sample_z = scaled_input_volume.depth() / 2;
		int sample_y = scaled_input_volume.height() / 2;
		int sample_x = scaled_input_volume.width() / 2;
		std::cout << "[DEBUG] Sample value at center (" << sample_x << "," << sample_y << "," << sample_z << "): " 
		          << scaled_input_volume(sample_x, sample_y, sample_z) << endl;
		
		// 输出索引(2,162,44)处的值以对比最大差异位置
		if (scaled_input_volume.depth() > 2 && scaled_input_volume.height() > 162 && scaled_input_volume.width() > 44) {
			std::cout << "[DEBUG] Value at index (44,162,2) [x,y,z]: " 
			          << scaled_input_volume(44, 162, 2) << endl;
		}
	}
	std::cout << "[DEBUG] =========================" << endl;

	// 保存预处理数据
	if (saveIntermediateResults) {
		// 保存归一化后但重采样前的数据（用于精确对比）
		savePreprocessedData(normalized_volume, L"after_normalization_before_resample");
		// 保存最终的预处理数据
		savePreprocessedData(scaled_input_volume, L"preprocessed_normalized_volume");
	}

	//调用滑窗推理函数
	std::cout << "[DEBUG] Calling slidingWindowInfer..." << endl;
	try {
		AI_INT is_ok = slidingWindowInfer(config, scaled_input_volume);
		std::cout << "slidingWindowInfer returned: " << is_ok << endl;
		if (is_ok != DentalCbctSegAI_STATUS_SUCCESS) {
			std::cerr << "[ERROR] slidingWindowInfer failed with code: " << is_ok << endl;
			return is_ok;
		}
	} catch (const std::exception& e) {
		std::cerr << "[ERROR] Exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	} catch (...) {
		std::cerr << "[ERROR] Unknown exception in slidingWindowInfer" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}

	//如果进行了3D重采样，调整大小
	if (is_volume_scaled)
		predicted_output_prob.resize(input_size[0], input_size[1], input_size[2], config.num_classes, 3);

	// 保存模型输出（概率体）
	if (saveIntermediateResults) {
		saveModelOutput(predicted_output_prob, L"model_output_probability");
	}

	// 不在这里执行argmax，保持概率图供后续处理
	// argmax将在getSegMask中的后处理流程中执行

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume)
{
	if (use_gpu) {
		std::cout << "[DEBUG] Configuring CUDA provider..." << endl;
		try {
			OrtCUDAProviderOptions cuda_options;
			//cuda_options.gpu_mem_limit = 6 * 1024 * 1024 * 1024;  // 设置显存6GB限制[6,12](@ref)
			cuda_options.device_id = 0;
			session_options.AppendExecutionProvider_CUDA(cuda_options);
			std::cout << "[DEBUG] CUDA provider configured successfully" << endl;
		} catch (const Ort::Exception& e) {
			std::cerr << "[WARNING] Failed to configure CUDA provider: " << e.what() << endl;
			std::cerr << "[WARNING] Falling back to CPU" << endl;
		}
	}

	std::cout << "env setting is done" << endl;

	// 创建会话
	Ort::AllocatorWithDefaultOptions allocator;
	
	// 检查模型文件名
	if (config.model_file_name == nullptr) {
		std::cerr << "ERROR: Model file name is NULL!" << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	}
	
	//try-catch处理ONNX Runtime异常
	try {
		std::cout << "Creating ONNX session..." << endl;
		
		Ort::Session session(env, config.model_file_name, session_options);
		
		// 使用AllocatedStringPtr来管理内存
		Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
		Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
		
		const char* input_name = input_name_ptr.get();
		const char* output_name = output_name_ptr.get();

		std::cout << "Session loading is done: " << endl;
		std::cout << "input_name: " << input_name << endl;
		std::cout << "output_name: " << output_name << endl;
		
		// 验证输入输出名称
		if (strcmp(input_name, output_name) == 0) {
			std::cerr << "[WARNING] Input and output have the same name: " << input_name << endl;
			std::cerr << "[WARNING] This might indicate a problem with the ONNX model" << endl;
		}
		auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	if (input_shape.size() != 5) {
		throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
	}

	// 验证
	if (config.patch_size.size() != 3) {
		throw std::runtime_error("Patch size should be 3D (depth, height, width)");
	}

	// ONNX张量形状: (batch, channel, depth, height, width)
	std::vector<int64_t> input_tensor_shape = { 1, 1, config.patch_size[0], config.patch_size[1], config.patch_size[2] };

	int depth = normalized_volume.depth();
	int width = normalized_volume.width();
	int height = normalized_volume.height();
	
	// Padding步骤（匹配Python的pad_nd_image）
	// 计算需要的padding以确保尺寸可被patch_size整除
	int padded_depth = depth;
	int padded_width = width;
	int padded_height = height;
	
	// 如果尺寸小于patch_size，需要padding到至少patch_size
	if (padded_depth < config.patch_size[0]) {
		padded_depth = config.patch_size[0];
	}
	if (padded_height < config.patch_size[1]) {
		padded_height = config.patch_size[1];
	}
	if (padded_width < config.patch_size[2]) {
		padded_width = config.patch_size[2];
	}
	
	// 计算padding量（居中padding，如果需要奇数padding，则"上"侧多padding 1）
	int pad_depth_before = (padded_depth - depth) / 2;
	int pad_depth_after = padded_depth - depth - pad_depth_before;
	int pad_width_before = (padded_width - width) / 2;
	int pad_width_after = padded_width - width - pad_width_before;
	int pad_height_before = (padded_height - height) / 2;
	int pad_height_after = padded_height - height - pad_height_before;
	
	// 创建padded volume
	CImg<float> padded_volume(padded_width, padded_height, padded_depth, 1, 0.0f);
	
	// 复制原始数据到padded volume的中心
	if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
		cimg_forXYZ(normalized_volume, x, y, z) {
			padded_volume(x + pad_width_before, y + pad_height_before, z + pad_depth_before) = 
				normalized_volume(x, y, z);
		}
	} else {
		// 如果没有padding，直接使用原始volume
		padded_volume = normalized_volume;
	}
	
	std::cout << "[DEBUG] Padding information:" << endl;
	std::cout << "  Original size: " << width << "x" << height << "x" << depth << endl;
	std::cout << "  Padded size: " << padded_width << "x" << padded_height << "x" << padded_depth << endl;
	std::cout << "  Padding: depth[" << pad_depth_before << "," << pad_depth_after 
	          << "] width[" << pad_width_before << "," << pad_width_after 
	          << "] height[" << pad_height_before << "," << pad_height_after << "]" << endl;
	
	// 使用padded dimensions进行后续计算
	int working_depth = padded_depth;
	int working_width = padded_width;
	int working_height = padded_height;

	// x图像宽度, y图像高度, z图像深度
	float step_size_ratio = config.step_size_ratio;
	
	// 匹配Python的compute_steps_for_sliding_window逻辑
	// 首先计算目标步长
	float target_step_x = config.patch_size[2] * step_size_ratio;
	float target_step_y = config.patch_size[1] * step_size_ratio;
	float target_step_z = config.patch_size[0] * step_size_ratio;
	
	// 计算步数
	int X_num_steps = std::max(1, (int)ceil(float(working_width - config.patch_size[2]) / target_step_x) + 1);
	int Y_num_steps = std::max(1, (int)ceil(float(working_height - config.patch_size[1]) / target_step_y) + 1);
	int Z_num_steps = std::max(1, (int)ceil(float(working_depth - config.patch_size[0]) / target_step_z) + 1);
	
	// 计算实际步长（匹配Python: 如果有多步，均匀分配）
	float actualStepSize[3];
	if (X_num_steps > 1) {
		actualStepSize[0] = float(working_width - config.patch_size[2]) / (X_num_steps - 1);
	} else {
		actualStepSize[0] = 0;  // 只有一步时，始终在位置0
	}
	
	if (Y_num_steps > 1) {
		actualStepSize[1] = float(working_height - config.patch_size[1]) / (Y_num_steps - 1);
	} else {
		actualStepSize[1] = 0;
	}
	
	if (Z_num_steps > 1) {
		actualStepSize[2] = float(working_depth - config.patch_size[0]) / (Z_num_steps - 1);
	} else {
		actualStepSize[2] = 0;
	}

	if (NETDEBUG_FLAG) {
		std::cout << "[DEBUG] Tile calculation:" << endl;
		std::cout << "  Working volume dimensions (padded): " << working_width << "x" << working_height << "x" << working_depth << endl;
		std::cout << "  Patch size: " << config.patch_size[2] << "x" << config.patch_size[1] << "x" << config.patch_size[0] << " (WxHxD)" << endl;
		std::cout << "  Step size ratio: " << step_size_ratio << endl;
		std::cout << "  Actual step sizes: X=" << actualStepSize[0] << ", Y=" << actualStepSize[1] << ", Z=" << actualStepSize[2] << endl;
		std::cout << "  Number of steps: X=" << X_num_steps << ", Y=" << Y_num_steps << ", Z=" << Z_num_steps << endl;
		std::cout << "  Total number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;
		std::cout << "  TTA (use_mirroring): " << (config.use_mirroring ? "enabled (NOT IMPLEMENTED)" : "disabled") << endl;
	}

	//初始化输出概率体（使用padded dimensions）
	CImg<float> padded_output_prob = CImg<float>(working_width, working_height, working_depth, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(working_width, working_height, working_depth, 1, 0.f);
	//std::cout << "predSegProbVolume shape: " << depth << width << height << endl;

	//CImg<float> input_patch = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
	CImg<float> win_pob = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.f);
	CImg<float> gaussisan_weight = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
	create_3d_gaussian_kernel(gaussisan_weight, config.patch_size);

	size_t input_patch_voxel_numel = config.patch_size[0] * config.patch_size[1] * config.patch_size[2];
	size_t output_patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	//
	int patch_count = 0;
	for (int sz = 0; sz < Z_num_steps; sz++)
	{
		int lb_z = (int)std::round(sz * actualStepSize[2]);
		// 确保不超出边界（使用padded dimensions）
		if (lb_z + config.patch_size[0] > working_depth) {
			lb_z = working_depth - config.patch_size[0];
		}
		lb_z = std::max(0, lb_z);
		int ub_z = lb_z + config.patch_size[0] - 1;

		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[1]);
			// 确保不超出边界（使用padded dimensions）
			if (lb_y + config.patch_size[1] > working_height) {
				lb_y = working_height - config.patch_size[1];
			}
			lb_y = std::max(0, lb_y);
			int ub_y = lb_y + config.patch_size[1] - 1;

			for (int sx = 0; sx < X_num_steps; sx++)
			{
				int lb_x = (int)std::round(sx * actualStepSize[0]);
				// 确保不超出边界（使用padded dimensions）
				if (lb_x + config.patch_size[2] > working_width) {
					lb_x = working_width - config.patch_size[2];
				}
				lb_x = std::max(0, lb_x);
				int ub_x = lb_x + config.patch_size[2] - 1;

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << "current tile#: " << patch_count << endl;

				std::cout << "[DEBUG] Patch bounds - X: [" << lb_x << ", " << ub_x << "], Y: [" << lb_y << ", " << ub_y << "], Z: [" << lb_z << ", " << ub_z << "]" << endl;

				CImg<float> input_patch;
				try {
					input_patch = padded_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
					//std::cout << "input_patch mean: " << input_patch.mean() << endl;
					//std::cout << "input_patch variance: " << input_patch.variance() << endl;
					std::cout << "input_patch dimensions: " << input_patch.width() << "x" << input_patch.height() << "x" << input_patch.depth() << endl;
					if (input_patch.width() != config.patch_size[2] || input_patch.height() != config.patch_size[1] || input_patch.depth() != config.patch_size[0]) {
						std::cerr << "[ERROR] Patch size mismatch! Expected: " << config.patch_size[2] << "x" << config.patch_size[1] << "x" << config.patch_size[0] << " (WxHxD)" << endl;
						return DentalCbctSegAI_STATUS_FAIED;
					}
				} catch (const CImgException& e) {
					std::cerr << "[ERROR] CImg exception during cropping: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				//std::vector<float> input_tensor_data;
				//const float* input_patch_ptr = input_patch.data(0, 0, 0, 0);
				//input_tensor_data.insert(input_tensor_data.end(), input_patch_ptr, input_patch_ptr + input_patch_voxel_numel);

				float* input_data_ptr = input_patch.data();

				// 
				Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
					OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
					input_data_ptr,
					input_patch_voxel_numel,
					input_tensor_shape.data(),
					input_tensor_shape.size());

				// 
				std::cout << "[DEBUG] Running ONNX inference for patch #" << patch_count << endl;
				
				// 验证输入 tensor
				if (input_data_ptr == nullptr) {
					std::cerr << "[ERROR] Input data pointer is null!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// 在try块外声明输出张量
				std::vector<Ort::Value> output_tensors;
				
				try {
					//session_ptr = std::make_unique<Ort::Session>(env, config.model_file_name, session_options);

					//auto output_tensors = session_ptr->Run(
					std::cout << "[DEBUG] Calling session.Run()..." << endl;
					output_tensors = session.Run(
						Ort::RunOptions{ nullptr },
						&input_name,
						&input_tensor,
						1,
						&output_name,
						1
					);
					std::cout << "[DEBUG] session.Run() completed successfully" << endl;

					// 暂时禁用TTA以简化调试
					// TODO: 实现完整的TTA（7种镜像组合）
					/*
					if (config.use_mirroring && use_gpu)
					{
						input_patch = input_patch.mirror('x');
					}
					*/

					std::cout << "[DEBUG] ONNX inference completed for patch #" << patch_count << endl;
				} catch (const Ort::Exception& e) {
					std::cerr << "[ERROR] ONNX Runtime exception during inference: " << e.what() << endl;
					std::cerr << "[ERROR] Error code: " << e.GetOrtErrorCode() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				} catch (const std::exception& e) {
					std::cerr << "[ERROR] Standard exception during inference: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				} catch (...) {
					std::cerr << "[ERROR] Unknown exception during inference" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// 处理输出张量
				std::cout << "[DEBUG] Processing output tensor..." << endl;
				
				if (output_tensors.empty()) {
					std::cerr << "[ERROR] No output tensors returned from inference!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}
				
				float* output_data = output_tensors[0].GetTensorMutableData<float>();
				
				if (output_data == nullptr) {
					std::cerr << "[ERROR] Output data pointer is null!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// 复制到CImg
				std::cout << "[DEBUG] Copying output data to win_pob..." << endl;
				std::memcpy(win_pob.data(), output_data, output_patch_vol_sz);
				output_tensors.clear();
				//input_tensor.release();

				std::cout << "[DEBUG] Output patch statistics:" << endl;
				std::cout << "  - Min: " << win_pob.min() << endl;
				std::cout << "  - Max: " << win_pob.max() << endl;
				std::cout << "  - Mean: " << win_pob.mean() << endl;
				std::cout << "  - Dimensions: " << win_pob.width() << "x" << win_pob.height() << "x" << win_pob.depth() << "x" << win_pob.spectrum() << endl;

				// 保存单个tile
				if (saveIntermediateResults) {
					saveTile(win_pob, patch_count, lb_x, lb_y, lb_z);
				}

				std::cout << "[DEBUG] Accumulating patch results..." << endl;
				try {
					cimg_forXYZC(win_pob, x, y, z, c) {
						int gx = lb_x + x;
						int gy = lb_y + y;
						int gz = lb_z + z;
						
						// 写入前验证边界（使用padded dimensions）
						if (gx < 0 || gx >= working_width || gy < 0 || gy >= working_height || gz < 0 || gz >= working_depth) {
							std::cerr << "[ERROR] Out of bounds write attempt: (" << gx << ", " << gy << ", " << gz << ")" << endl;
							return DentalCbctSegAI_STATUS_FAIED;
						}
						
						padded_output_prob(gx, gy, gz, c) += (win_pob(x, y, z, c) * gaussisan_weight(x, y, z));
					}
					cimg_forXYZ(gaussisan_weight, x, y, z) {
						count_vol(lb_x + x, lb_y + y, lb_z + z) += gaussisan_weight(x, y, z);
					}
				} catch (const std::exception& e) {
					std::cerr << "[ERROR] Exception during patch accumulation: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}
			}
		}
	}

	//归一化
	cimg_forXYZC(padded_output_prob, x, y, z, c) {
		padded_output_prob(x, y, z, c) /= count_vol(x, y, z);
	}
	
	// 从padded结果中提取原始尺寸的输出（移除padding）
	predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.f);
	if (pad_depth_before >= 0 && pad_width_before >= 0 && pad_height_before >= 0) {
		cimg_forXYZC(predicted_output_prob, x, y, z, c) {
			predicted_output_prob(x, y, z, c) = padded_output_prob(x + pad_width_before, 
			                                                        y + pad_height_before, 
			                                                        z + pad_depth_before, c);
		}
	} else {
		predicted_output_prob = padded_output_prob;
	}
	
	std::cout << "[DEBUG] Removed padding, final output size: " << predicted_output_prob.width() 
	          << "x" << predicted_output_prob.height() << "x" << predicted_output_prob.depth() << endl;
	std::cout << "Sliding window inference is done." << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
	} catch (const Ort::Exception& e) {
		std::cerr << "ONNX Runtime exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	} catch (const std::exception& e) {
		std::cerr << "Standard exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
}


void DentalUnet::CTNormalization(CImg<float>& input_volume, nnUNetConfig config)
{
	//使用percentile值进行裁剪（与Python版本一致）
	double lower_bound = config.percentile_00_5;
	double upper_bound = config.percentile_99_5;
	
	std::cout << "[DEBUG] CTNormalization parameters:" << endl;
	std::cout << "  - lower_bound (percentile_00_5): " << lower_bound << endl;
	std::cout << "  - upper_bound (percentile_99_5): " << upper_bound << endl;
	std::cout << "  - mean: " << config.mean_std_HU[0] << endl;
	std::cout << "  - std: " << config.mean_std_HU[1] << endl;
	std::cout << "  - Data range before clipping: [" << input_volume.min() << ", " << input_volume.max() << "]" << endl;
	
	input_volume.cut(lower_bound, upper_bound);
	
	std::cout << "  - Data range after clipping: [" << input_volume.min() << ", " << input_volume.max() << "]" << endl;

	//应用z-score标准化（使用double提高精度）
	double mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	double std_hu4dentalCTNormalization = config.mean_std_HU[1];
	input_volume -= mean_hu4dentalCTNormalization;
	input_volume /= std_hu4dentalCTNormalization;
	
	std::cout << "  - Data range after normalization: [" << input_volume.min() << ", " << input_volume.max() << "]" << endl;
}


void DentalUnet::create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes)
{
	// 匹配Python版本：sigma_scale = 1/8
	float sigma_scale = 1.0f / 8.0f;
	float value_scaling_factor = 10.0f;

	int64_t depth  = patch_sizes[0];
	int64_t height = patch_sizes[1]; 
	int64_t width  = patch_sizes[2];

	// 计算中心点坐标
	float z_center = (depth - 1)  / 2.0f;
	float y_center = (height - 1) / 2.0f;
	float x_center = (width - 1)  / 2.0f;

	// 使用与Python相同的sigma计算方法
	float z_sigma = depth  * sigma_scale;
	float y_sigma = height * sigma_scale;
	float x_sigma = width  * sigma_scale;
	
	std::cout << "[DEBUG] Gaussian kernel parameters:" << endl;
	std::cout << "  - Patch size: [" << depth << ", " << height << ", " << width << "]" << endl;
	std::cout << "  - Sigmas: [" << z_sigma << ", " << y_sigma << ", " << x_sigma << "]" << endl;

	float z_part = 0.f;
	float y_part = 0.f;
	float x_part = 0.f;
	cimg_forXYZ(gaussisan_weight, x, y, z) {
		z_part = std::exp(-0.5f * std::pow((z - z_center) / z_sigma, 2));
		y_part = std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2));
		x_part = std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2));
		gaussisan_weight(x, y, z) = z_part * y_part * x_part;
	}

	// 匹配Python的归一化方法：除以max再乘以value_scaling_factor
	float max_val = gaussisan_weight.max();
	if (max_val > 0) {
		gaussisan_weight *= (value_scaling_factor / max_val);
	}
	
	// 处理0值（匹配Python：将0值设置为最小非零值）
	float min_non_zero = std::numeric_limits<float>::max();
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
	
	std::cout << "[DEBUG] Gaussian weight statistics:" << endl;
	std::cout << "  - Min: " << gaussisan_weight.min() << endl;
	std::cout << "  - Max: " << gaussisan_weight.max() << endl;
	std::cout << "  - Mean: " << gaussisan_weight.mean() << endl;
}


CImg<short> DentalUnet::argmax_spectrum(const CImg<float>& input) {
	if (input.is_empty() || input.spectrum() == 0) {
		throw std::invalid_argument("Input must be a non-empty 4D CImg with spectrum dimension.");
	}

	// 创建结果图像，大小与输入相同，但spectrum维度为1
	CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

	// 遍历每个体素 (x,y,z)
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
		result(x, y, z) = max_idx; // 存储最大值的索引
	}
	return result;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	std::cout << "[DEBUG] getSegMask called - performing post-processing" << endl;
	std::cout << "[DEBUG] Original dimensions: " << Width0 << "x" << Height0 << "x" << Depth0 << endl;
	std::cout << "[DEBUG] Probability volume dimensions: " << predicted_output_prob.width() 
	          << "x" << predicted_output_prob.height() << "x" << predicted_output_prob.depth() 
	          << "x" << predicted_output_prob.spectrum() << " (spectrum=" << predicted_output_prob.spectrum() << ")" << endl;
	
	// 步骤1：对概率图执行argmax（在转置后的坐标系中）
	std::cout << "[DEBUG] Step 1: Performing argmax on probability volume" << endl;
	output_seg_mask = argmax_spectrum(predicted_output_prob);
	std::cout << "[DEBUG] Segmentation mask dimensions after argmax: " 
	          << output_seg_mask.width() << "x" << output_seg_mask.height() << "x" << output_seg_mask.depth() << endl;
	
	// 保存后处理数据（在转置撤销前）
	if (saveIntermediateResults) {
		savePostprocessedData(output_seg_mask, L"postprocessed_segmentation_mask_before_transpose");
	}
	
	// 步骤2：撤销转置（恢复到原始坐标系）
	std::cout << "[DEBUG] Step 2: Reverting transpose" << endl;
	output_seg_mask.permute_axes(unetConfig.cimg_transpose_backward);
	std::cout << "[DEBUG] Segmentation mask dimensions after transpose revert: " 
	          << output_seg_mask.width() << "x" << output_seg_mask.height() << "x" << output_seg_mask.depth() << endl;
	
	// 步骤3：检查是否需要恢复裁剪
	if (output_seg_mask.width() != Width0 || output_seg_mask.height() != Height0 || output_seg_mask.depth() != Depth0) {
		std::cout << "[DEBUG] Step 3: Restoring cropped result to original size" << endl;
		
		// 创建原始尺寸的结果mask，初始化为0
		CImg<short> full_result(Width0, Height0, Depth0, 1, 0);
		
		// 先检查bbox是否已经初始化
		if (crop_bbox.x_max == -1 || crop_bbox.y_max == -1 || crop_bbox.z_max == -1) {
			std::cout << "[ERROR] Crop bbox appears uninitialized!" << endl;
			std::cout << "[ERROR] Bbox values: X[" << crop_bbox.x_min << ":" << crop_bbox.x_max 
			          << "], Y[" << crop_bbox.y_min << ":" << crop_bbox.y_max 
			          << "], Z[" << crop_bbox.z_min << ":" << crop_bbox.z_max << "]" << endl;
			std::cout << "[ERROR] This suggests crop_to_nonzero was not called properly" << endl;
			
			// 使用fallback逻辑
			int copy_width = std::min(output_seg_mask.width(), Width0);
			int copy_height = std::min(output_seg_mask.height(), Height0);
			int copy_depth = std::min(output_seg_mask.depth(), Depth0);
			for (int z = 0; z < copy_depth; z++) {
				for (int y = 0; y < copy_height; y++) {
					for (int x = 0; x < copy_width; x++) {
						full_result(x, y, z) = output_seg_mask(x, y, z);
					}
				}
			}
		}
		// 将裁剪后的结果放回到原始位置
		else if (crop_bbox.x_min >= 0 && crop_bbox.x_max < Width0 && 
		    crop_bbox.y_min >= 0 && crop_bbox.y_max < Height0 &&
		    crop_bbox.z_min >= 0 && crop_bbox.z_max < Depth0) {
		    
		    std::cout << "[DEBUG] Restoring cropped result using bbox: X[" << crop_bbox.x_min << ":" << crop_bbox.x_max 
		              << "], Y[" << crop_bbox.y_min << ":" << crop_bbox.y_max 
		              << "], Z[" << crop_bbox.z_min << ":" << crop_bbox.z_max << "]" << endl;
		    
		    // 将output_seg_mask的内容复制到full_result的对应位置
		    cimg_forXYZ(output_seg_mask, x, y, z) {
		        int orig_x = x + crop_bbox.x_min;
		        int orig_y = y + crop_bbox.y_min;
		        int orig_z = z + crop_bbox.z_min;
		        if (orig_x < Width0 && orig_y < Height0 && orig_z < Depth0) {
		            full_result(orig_x, orig_y, orig_z) = output_seg_mask(x, y, z);
		        }
		    }
		} else {
		    std::cout << "[WARNING] Invalid crop bbox detected!" << endl;
		    std::cout << "[WARNING] Crop bbox values: X[" << crop_bbox.x_min << ":" << crop_bbox.x_max 
		              << "], Y[" << crop_bbox.y_min << ":" << crop_bbox.y_max 
		              << "], Z[" << crop_bbox.z_min << ":" << crop_bbox.z_max << "]" << endl;
		    std::cout << "[WARNING] Original image bounds: X[0:" << (Width0-1) 
		              << "], Y[0:" << (Height0-1) << "], Z[0:" << (Depth0-1) << "]" << endl;
		    std::cout << "[WARNING] Attempting to copy as much as possible..." << endl;
		    
		    // 如果bbox无效，尝试直接复制能复制的部分
		    int copy_width = std::min(output_seg_mask.width(), Width0);
		    int copy_height = std::min(output_seg_mask.height(), Height0);
		    int copy_depth = std::min(output_seg_mask.depth(), Depth0);
		    
		    std::cout << "[WARNING] Copying dimensions: " << copy_width << "x" << copy_height << "x" << copy_depth << endl;
		    
		    for (int z = 0; z < copy_depth; z++) {
		        for (int y = 0; y < copy_height; y++) {
		            for (int x = 0; x < copy_width; x++) {
		                full_result(x, y, z) = output_seg_mask(x, y, z);
		            }
		        }
		    }
		}
		
		// 复制恢复后的结果
		long volSize = Width0 * Height0 * Depth0 * sizeof(short);
		std::memcpy(dstData->ptr_Data, full_result.data(), volSize);
		std::cout << "[DEBUG] Result restored to original dimensions" << endl;
		
	} else {
		std::cout << "[DEBUG] Using output_seg_mask directly (dimensions match)" << endl;
		// 如果尺寸匹配，直接复制
		long volSize = Width0 * Height0 * Depth0 * sizeof(short);
		std::memcpy(dstData->ptr_Data, output_seg_mask.data(), volSize);
	}
	
	// 将保存的origin信息传回给调用者
	dstData->OriginX = imageMetadata.origin[0];
	dstData->OriginY = imageMetadata.origin[1];
	dstData->OriginZ = imageMetadata.origin[2];
	
	// 同时确保spacing信息也正确传回
	dstData->VoxelSpacingX = imageMetadata.spacing[0];
	dstData->VoxelSpacingY = imageMetadata.spacing[1];
	dstData->VoxelSpacingZ = imageMetadata.spacing[2];
	
	// 保存最终的后处理结果
	if (saveIntermediateResults) {
		// 需要从dstData创建一个CImg对象来保存
		CImg<short> final_result(dstData->Width, dstData->Height, dstData->Depth, 1);
		std::memcpy(final_result.data(), dstData->ptr_Data, dstData->Width * dstData->Height * dstData->Depth * sizeof(short));
		savePostprocessedData(final_result, L"postprocessed_segmentation_mask_final");
	}

	std::cout << "[DEBUG] getSegMask completed" << endl;
	return DentalCbctSegAI_STATUS_SUCCESS;
}


void DentalUnet::savePreprocessedData(const CImg<float>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || preprocessOutputPath.empty()) return;
	
	// 使用ITK保存为NIfTI格式以保留origin信息
	std::wstring niftiPath = preprocessOutputPath + L"\\" + filename + L".nii.gz";
	std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
	
	// 定义ITK类型
	using FloatImageType = itk::Image<float, 3>;
	using WriterType = itk::ImageFileWriter<FloatImageType>;
	
	// 创建ITK图像
	FloatImageType::Pointer image = FloatImageType::New();
	
	// 设置图像大小
	FloatImageType::SizeType size;
	size[0] = data.width();
	size[1] = data.height();
	size[2] = data.depth();
	
	FloatImageType::IndexType start;
	start.Fill(0);
	
	FloatImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(start);
	
	image->SetRegions(region);
	image->Allocate();
	
	// 设置元数据
	FloatImageType::PointType origin;
	origin[0] = imageMetadata.origin[0];
	origin[1] = imageMetadata.origin[1];
	origin[2] = imageMetadata.origin[2];
	image->SetOrigin(origin);
	
	FloatImageType::SpacingType spacing;
	spacing[0] = imageMetadata.spacing[0];
	spacing[1] = imageMetadata.spacing[1];
	spacing[2] = imageMetadata.spacing[2];
	image->SetSpacing(spacing);
	
	// 复制数据
	itk::ImageRegionIterator<FloatImageType> it(image, region);
	const float* cimg_data = data.data();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		it.Set(*cimg_data++);
	}
	
	// 写入图像
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(narrowNiftiPath);
	writer->SetInput(image);
	
	try {
		writer->Update();
	} catch (itk::ExceptionObject& e) {
		std::cerr << "Error saving preprocessed data: " << e << std::endl;
	}
	
	// 保存为二进制格式供numpy使用
	std::wstring rawPath = preprocessOutputPath + L"\\" + filename + L".raw";
	std::wstring metaPath = preprocessOutputPath + L"\\" + filename + L"_meta.txt";
	
	std::string narrowRawPath(rawPath.begin(), rawPath.end());
	std::string narrowMetaPath(metaPath.begin(), metaPath.end());
	
	// 保存原始数据
	std::ofstream rawFile(narrowRawPath, std::ios::binary);
	rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
	rawFile.close();
	
	// 保存Python使用的元数据
	std::ofstream metaFile(narrowMetaPath);
	metaFile << "dtype: float32" << std::endl;
	metaFile << "shape: (" << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
	metaFile << "order: C" << std::endl;
	metaFile << "description: Preprocessed normalized volume" << std::endl;
	metaFile.close();
	
	std::cout << "[DEBUG] Saved preprocessed data:" << std::endl;
	std::cout << "  - NIfTI format: " << narrowNiftiPath << std::endl;
	std::cout << "  - Raw binary: " << narrowRawPath << std::endl;
}


void DentalUnet::saveModelOutput(const CImg<float>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || modelOutputPath.empty()) return;
	
	// 使用ITK保存为NIfTI格式以保留origin信息
	std::wstring niftiPath = modelOutputPath + L"\\" + filename + L".nii.gz";
	std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
	
	// 定义ITK类型
	using FloatImageType = itk::Image<float, 3>;
	using WriterType = itk::ImageFileWriter<FloatImageType>;
	
	// 对于多通道数据，我们可能需要分别保存每个通道
	// 目前，如果是概率图则保存第一个通道
	CImg<float> dataToSave;
	if (data.spectrum() > 1) {
		dataToSave = data.get_channel(0);
	} else {
		dataToSave = data;
	}
	
	// 创建ITK图像
	FloatImageType::Pointer image = FloatImageType::New();
	
	// 设置图像大小
	FloatImageType::SizeType size;
	size[0] = dataToSave.width();
	size[1] = dataToSave.height();
	size[2] = dataToSave.depth();
	
	FloatImageType::IndexType start;
	start.Fill(0);
	
	FloatImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(start);
	
	image->SetRegions(region);
	image->Allocate();
	
	// 设置元数据
	FloatImageType::PointType origin;
	origin[0] = imageMetadata.origin[0];
	origin[1] = imageMetadata.origin[1];
	origin[2] = imageMetadata.origin[2];
	image->SetOrigin(origin);
	
	FloatImageType::SpacingType spacing;
	spacing[0] = imageMetadata.spacing[0];
	spacing[1] = imageMetadata.spacing[1];
	spacing[2] = imageMetadata.spacing[2];
	image->SetSpacing(spacing);
	
	// 复制数据
	itk::ImageRegionIterator<FloatImageType> it(image, region);
	const float* cimg_data = dataToSave.data();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		it.Set(*cimg_data++);
	}
	
	// 写入图像
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(narrowNiftiPath);
	writer->SetInput(image);
	
	try {
		writer->Update();
	} catch (itk::ExceptionObject& e) {
		std::cerr << "Error saving model output: " << e << std::endl;
	}
	
	// 保存为二进制格式供numpy使用
	std::wstring rawPath = modelOutputPath + L"\\" + filename + L".raw";
	std::wstring metaPath = modelOutputPath + L"\\" + filename + L"_meta.txt";
	
	std::string narrowRawPath(rawPath.begin(), rawPath.end());
	std::string narrowMetaPath(metaPath.begin(), metaPath.end());
	
	// 保存原始数据
	std::ofstream rawFile(narrowRawPath, std::ios::binary);
	rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
	rawFile.close();
	
	// 保存Python使用的元数据
	std::ofstream metaFile(narrowMetaPath);
	metaFile << "dtype: float32" << std::endl;
	metaFile << "shape: (" << data.spectrum() << ", " << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
	metaFile << "order: C" << std::endl;
	metaFile << "description: Model output probability volume (channels, depth, height, width)" << std::endl;
	metaFile.close();
	
	std::cout << "[DEBUG] Saved model output:" << std::endl;
	std::cout << "  - NIfTI format: " << narrowNiftiPath << std::endl;
	std::cout << "  - Raw binary: " << narrowRawPath << std::endl;
}


void DentalUnet::savePostprocessedData(const CImg<short>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || postprocessOutputPath.empty()) return;
	
	// 使用ITK保存为NIfTI格式以保留origin信息
	std::wstring niftiPath = postprocessOutputPath + L"\\" + filename + L".nii.gz";
	std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
	
	// 定义ITK类型
	using ShortImageType = itk::Image<short, 3>;
	using WriterType = itk::ImageFileWriter<ShortImageType>;
	
	// 创建ITK图像
	ShortImageType::Pointer image = ShortImageType::New();
	
	// 设置图像大小
	ShortImageType::SizeType size;
	size[0] = data.width();
	size[1] = data.height();
	size[2] = data.depth();
	
	ShortImageType::IndexType start;
	start.Fill(0);
	
	ShortImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(start);
	
	image->SetRegions(region);
	image->Allocate();
	
	// 设置元数据
	ShortImageType::PointType origin;
	origin[0] = imageMetadata.origin[0];
	origin[1] = imageMetadata.origin[1];
	origin[2] = imageMetadata.origin[2];
	image->SetOrigin(origin);
	
	ShortImageType::SpacingType spacing;
	spacing[0] = imageMetadata.spacing[0];
	spacing[1] = imageMetadata.spacing[1];
	spacing[2] = imageMetadata.spacing[2];
	image->SetSpacing(spacing);
	
	// 复制数据
	itk::ImageRegionIterator<ShortImageType> it(image, region);
	const short* cimg_data = data.data();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		it.Set(*cimg_data++);
	}
	
	// 写入图像
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(narrowNiftiPath);
	writer->SetInput(image);
	
	try {
		writer->Update();
	} catch (itk::ExceptionObject& e) {
		std::cerr << "Error saving postprocessed data: " << e << std::endl;
	}
	
	// 保存为二进制格式供numpy使用
	std::wstring rawPath = postprocessOutputPath + L"\\" + filename + L".raw";
	std::wstring metaPath = postprocessOutputPath + L"\\" + filename + L"_meta.txt";
	
	std::string narrowRawPath(rawPath.begin(), rawPath.end());
	std::string narrowMetaPath(metaPath.begin(), metaPath.end());
	
	// 保存原始数据
	std::ofstream rawFile(narrowRawPath, std::ios::binary);
	rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
	rawFile.close();
	
	// 保存Python使用的元数据
	std::ofstream metaFile(narrowMetaPath);
	metaFile << "dtype: int16" << std::endl;
	metaFile << "shape: (" << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
	metaFile << "order: C" << std::endl;
	metaFile << "description: Postprocessed segmentation mask" << std::endl;
	metaFile.close();
	
	std::cout << "[DEBUG] Saved postprocessed data:" << std::endl;
	std::cout << "  - NIfTI format: " << narrowNiftiPath << std::endl;
	std::cout << "  - Raw binary: " << narrowRawPath << std::endl;
}


void DentalUnet::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z)
{
	if (!saveIntermediateResults || modelOutputPath.empty()) return;
	
	// 如果不存在则创建tiles子目录
	std::filesystem::create_directories(modelOutputPath + L"\\tiles");
	
	std::wstringstream ss;
	ss << L"tile_" << std::setfill(L'0') << std::setw(4) << tileIndex 
	   << L"_x" << x << L"_y" << y << L"_z" << z;
	
	// 保存为NIfTI格式（未压缩）
	std::wstring niftiPath = modelOutputPath + L"\\tiles\\" + ss.str() + L".nii";
	std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
	tile.save(narrowNiftiPath.c_str());
	
	// 保存为二进制格式
	std::wstring rawPath = modelOutputPath + L"\\tiles\\" + ss.str() + L".raw";
	std::wstring metaPath = modelOutputPath + L"\\tiles\\" + ss.str() + L"_meta.txt";
	
	std::string narrowRawPath(rawPath.begin(), rawPath.end());
	std::string narrowMetaPath(metaPath.begin(), metaPath.end());
	
	// 保存原始数据
	std::ofstream rawFile(narrowRawPath, std::ios::binary);
	rawFile.write(reinterpret_cast<const char*>(tile.data()), tile.size() * sizeof(float));
	rawFile.close();
	
	// 保存元数据
	std::ofstream metaFile(narrowMetaPath);
	metaFile << "dtype: float32" << std::endl;
	metaFile << "shape: (" << tile.spectrum() << ", " << tile.depth() << ", " << tile.height() << ", " << tile.width() << ")" << std::endl;
	metaFile << "order: C" << std::endl;
	metaFile << "position: (" << x << ", " << y << ", " << z << ")" << std::endl;
	metaFile << "tile_index: " << tileIndex << std::endl;
	metaFile.close();
	
	if (tileIndex % 10 == 0) {  // Log every 10th tile to reduce output
		std::cout << "[DEBUG] Saved tile " << tileIndex << endl;
	}
}


