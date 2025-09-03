#include "UnetMain.h"
#include "UnetInference.h"
#include "UnetPostprocessor.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <tuple>
#include <chrono>

UnetMain::UnetMain()
{
	NETDEBUG_FLAG = true;

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
	
	unetConfig.model_file_name = model_fn;
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

// 新增：参数设置接口实现
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

// 新增：JSON配置接口实现
bool UnetMain::setConfigFromJsonString(const char* jsonContent)
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
		
		return true;
	}
	
	return false;
}


void  UnetMain::setDnnOptions()
{
	//??????????????????????
}


void  UnetMain::setAlgParameter()
{
	//????????????????
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


AI_INT  UnetMain::initializeOnnxruntimeInstances()
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

	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth
	
	// 读取原始spacing（如果提供了的话）
	if (srcData->OriginalVoxelSpacingX > 0 && srcData->OriginalVoxelSpacingY > 0 && srcData->OriginalVoxelSpacingZ > 0) {
		original_voxel_spacing = { srcData->OriginalVoxelSpacingX, srcData->OriginalVoxelSpacingY, srcData->OriginalVoxelSpacingZ };
	} else {
		// 如果没有提供原始spacing，则使用当前spacing作为原始spacing
		original_voxel_spacing = input_voxel_spacing;
	}

	// 统计信息将在预处理流水线中计算
	std::cout << "Input volume loaded successfully" << endl;
	std::cout << "  Dimensions: " << Width0 << " x " << Height0 << " x " << Depth0 << endl;
	std::cout << "  Spacing: " << input_voxel_spacing[0] << " x " << input_voxel_spacing[1] << " x " << input_voxel_spacing[2] << " mm" << endl;

	return UnetSegAI_STATUS_SUCCESS;
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
	
}

// 实现crop_to_nonzero函数，与Python版本对齐
CImg<short> UnetMain::crop_to_nonzero(const CImg<short>& input, CropBBox& bbox) {
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
		bbox.x_min = 0; bbox.x_max = input.width() - 1;
		bbox.y_min = 0; bbox.y_max = input.height() - 1;
		bbox.z_min = 0; bbox.z_max = input.depth() - 1;
		
		// 创建全为0的seg_mask（因为全是背景）
		seg_mask = CImg<short>(input.width(), input.height(), input.depth(), 1, -1);
		return input;
	}
	
	// 验证bbox是否合理
	if (bbox.x_min > bbox.x_max || bbox.y_min > bbox.y_max || bbox.z_min > bbox.z_max) {
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
	                                     
	          
	return cropped;
}

AI_INT  UnetMain::performInference(AI_DataInfo *srcData)
{
	int input_status = setInput(srcData);
	if (input_status != UnetSegAI_STATUS_SUCCESS)
		return input_status;

	// 按照Python版本的顺序进行预处理：
	// 1. 转置
	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_forward);
	transposed_input_voxel_spacing.clear();
	transposed_original_voxel_spacing.clear();
	for (int i = 0; i < 3; ++i) {
		transposed_input_voxel_spacing.push_back(input_voxel_spacing[unetConfig.transpose_forward[i]]);
		transposed_original_voxel_spacing.push_back(original_voxel_spacing[unetConfig.transpose_forward[i]]);
	}
	
	// 2. 裁剪到非零区域
	CImg<short> cropped_volume = crop_to_nonzero(input_cbct_volume, crop_bbox);
	
	// 3. 在裁剪后的数据上计算或使用配置的归一化参数
	
	// 根据归一化类型决定是否使用JSON配置的值
	if (unetConfig.normalization_type == "ZScoreNormalization") {
		// ZScoreNormalization总是动态计算mean和std（MRI等模态）
		
		if (unetConfig.use_mask_for_norm) {
			// 在mask区域动态计算
			// 这里暂不计算，将在segModelInfer的归一化步骤中动态计算
			intensity_mean = 0.0;  // 占位值
			intensity_std = 1.0;   // 占位值
		} else {
			// 在整个裁剪后的数据上动态计算
			intensity_mean = cropped_volume.mean();  // CImg::mean()返回double
			double var = cropped_volume.variance();  // CImg::variance()返回double
			intensity_std = std::sqrt(var);
			if (intensity_std < 1e-8) intensity_std = 1e-8;  // 匹配Python的max(std, 1e-8)
		}
	} else if (unetConfig.normalization_type == "CTNormalization" || 
	           unetConfig.normalization_type == "CT" || 
	           unetConfig.normalization_type == "ct") {
		// CTNormalization使用JSON配置的值（CT等标准化模态）
		intensity_mean = unetConfig.mean;
		intensity_std = unetConfig.std;
	} else {
		// 默认行为：动态计算
		intensity_mean = cropped_volume.mean();  // CImg::mean()返回double
		double var = cropped_volume.variance();  // CImg::variance()返回double
		intensity_std = std::sqrt(var);
		if (intensity_std < 1e-8) intensity_std = 1e-8;
	}

	// 4. 调用推理（包含归一化和重采样）
	int infer_status = segModelInfer(unetConfig, cropped_volume);

	// 不在这里恢复转置，将在getSegMask的最后步骤中处理
	// 保持数据在转置后的坐标系中，以便正确处理裁剪恢复

	return infer_status;
}


AI_INT  UnetMain::segModelInfer(nnUNetConfig config, CImg<short> input_volume)
{
	std::cout << "\n======= Preprocessing Stage =======" << endl;
	auto preprocess_start = std::chrono::steady_clock::now();

	if (transposed_input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// bool is_volume_scaled = false;  // 注释掉条件判断，改为始终缩放
	bool is_volume_scaled = true;  // 使用与Python相同的逻辑：始终进行缩放
	////input_voxel_spacing = {voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth 
	std::vector<int64_t> input_size = { input_volume.width(), input_volume.height(), input_volume.depth()};
	std::vector<int64_t> output_size;
	float scaled_factor = 1.f;
	
	
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
	}


	// 按照Python版本的顺序：先归一化，后重采样
	
	// Step 1: 归一化（在原始分辨率上进行）
	CImg<float> normalized_volume;
	normalized_volume.assign(input_volume);  // 转换为float
	
	// 保存归一化前的数据
	if (saveIntermediateResults) {
		savePreprocessedData(normalized_volume, L"before_normalization");
	}
	
	// 输出归一化前的统计信息
	
	// 输出seg_mask的维度以确认它与normalized_volume匹配
	
	// 统计seg_mask的值分布
	int seg_positive = 0, seg_negative = 0, seg_zero = 0;
	cimg_forXYZ(seg_mask, x, y, z) {
		if (seg_mask(x, y, z) > 0) seg_positive++;
		else if (seg_mask(x, y, z) < 0) seg_negative++;
		else seg_zero++;
	}
	
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
		}
	}
	
	// 输出特定位置的值（用于精确对比）
	if (normalized_volume.depth() > 2 && normalized_volume.height() > 162 && normalized_volume.width() > 44) {
	}
	
	// 输出一些采样点的值用于对比
	for (int i = 0; i < 5 && i < normalized_volume.depth(); i++) {
		for (int j = 0; j < 5 && j < normalized_volume.height(); j++) {
			for (int k = 0; k < 5 && k < normalized_volume.width(); k++) {
				if (normalized_volume(k, j, i) != 0) {
					break;
				}
			}
		}
	}
	
	
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
		CTNormalization(normalized_volume, config);
		break;
	case 20:
		if (config.use_mask_for_norm) {
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
				
				
				// 验证seg_mask的统计信息
				int seg_positive = 0, seg_negative = 0;
				cimg_forXYZ(seg_mask, x, y, z) {
					if (seg_mask(x, y, z) >= 0) seg_positive++;
					else seg_negative++;
				}
				
				// 只对mask区域进行归一化，背景设为0
				cimg_forXYZ(normalized_volume, x, y, z) {
					if (mask(x, y, z)) {
						normalized_volume(x, y, z) = (normalized_volume(x, y, z) - mask_mean) / mask_std;
					} else {
						normalized_volume(x, y, z) = 0.0f;
					}
				}
			} else {
			}
		} else {
			// 传统的全局归一化
			normalized_volume -= intensity_mean;
			normalized_volume /= intensity_std;
		}
		break;
	default:
		normalized_volume -= intensity_mean;
		normalized_volume /= intensity_std;
		break;
	}
	
	// 输出归一化后的详细统计信息
	
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
	}
	
	// 输出特定位置的值（用于精确对比）
	if (normalized_volume.depth() > 2 && normalized_volume.height() > 162 && normalized_volume.width() > 44) {
	}
	

	// Step 2: 重采样（在归一化后进行）
	CImg<float> scaled_input_volume;
	if (is_volume_scaled) {
		// 使用三次插值（5）而不是线性插值（3）以匹配Python的order=3
		// CImg插值模式: 0=最近邻, 1=线性, 2=移动平均, 3=线性, 5=三次(cubic)
		scaled_input_volume = normalized_volume.get_resize(output_size[0], output_size[1], output_size[2], -100, 5);
	} else {
		scaled_input_volume.assign(normalized_volume);
	}

	

	// 保存预处理数据
	if (saveIntermediateResults) {
		// 保存归一化后但重采样前的数据（用于精确对比）
		savePreprocessedData(normalized_volume, L"after_normalization_before_resample");
		// 保存最终的预处理数据
		savePreprocessedData(scaled_input_volume, L"preprocessed_normalized_volume");
	}

	auto preprocess_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> preprocess_elapsed = preprocess_end - preprocess_start;
	std::cout << "Preprocessing completed in " << preprocess_elapsed.count() << " seconds" << endl;
	std::cout << "  Preprocessed volume shape: " << scaled_input_volume.width() << " x " << scaled_input_volume.height() << " x " << scaled_input_volume.depth() << endl;
	std::cout << "  Mean: " << scaled_input_volume.mean() << ", Std: " << std::sqrt(scaled_input_volume.variance()) << endl;
	std::cout << "======= Preprocessing Complete =======" << endl;

	//调用滑窗推理函数
	std::cout << "\n======= Sliding Window Inference =======" << endl;
	auto inference_start = std::chrono::steady_clock::now();
	try {
		// 使用新的UnetInference类进行推理
		AI_INT is_ok = UnetInference::runSlidingWindow(this, config, scaled_input_volume, 
		                                              predicted_output_prob, env, session_options, use_gpu);
		if (is_ok != UnetSegAI_STATUS_SUCCESS) {
			return is_ok;
		}
	} catch (const std::exception& e) {
		return UnetSegAI_STATUS_FAIED;
	} catch (...) {
		return UnetSegAI_STATUS_FAIED;
	}

	auto inference_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> inference_elapsed = inference_end - inference_start;
	std::cout << "Inference completed in " << inference_elapsed.count() << " seconds" << endl;
	std::cout << "======= Inference Complete =======" << endl;

	//如果进行了3D重采样，调整大小
	if (is_volume_scaled)
		predicted_output_prob.resize(input_size[0], input_size[1], input_size[2], config.num_classes, 3);

	// 保存模型输出（概率体）
	if (saveIntermediateResults) {
		saveModelOutput(predicted_output_prob, L"model_output_probability");
		std::cout << "  Model output saved to: result/model_output/" << endl;
	}

	// 不在这里执行argmax，保持概率图供后续处理
	// argmax将在getSegMask中的后处理流程中执行

	return UnetSegAI_STATUS_SUCCESS;
}

void UnetMain::CTNormalization(CImg<float>& input_volume, nnUNetConfig config)
{
	//使用percentile值进行裁剪（与Python版本一致）
	double lower_bound = config.percentile_00_5;
	double upper_bound = config.percentile_99_5;
	
	input_volume.cut(lower_bound, upper_bound);

	//应用z-score标准化（使用double提高精度）
	double mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	double std_hu4dentalCTNormalization = config.mean_std_HU[1];
	input_volume -= mean_hu4dentalCTNormalization;
	input_volume /= std_hu4dentalCTNormalization;
}

AI_INT  UnetMain::getSegMask(AI_DataInfo *dstData)
{
	// 使用新的UnetPostprocessor类进行后处理
	return UnetPostprocessor::processSegmentationMask(this, predicted_output_prob, dstData);
}


void UnetMain::savePreprocessedData(const CImg<float>& data, const std::wstring& filename)
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
	
}


void UnetMain::saveModelOutput(const CImg<float>& data, const std::wstring& filename)
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
	
}


void UnetMain::savePostprocessedData(const CImg<short>& data, const std::wstring& filename)
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
	
}


void UnetMain::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z)
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
	
}


