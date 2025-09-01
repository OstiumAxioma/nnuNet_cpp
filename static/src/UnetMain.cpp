#include "UnetMain.h"
#include "UnetSegAI_API.h"
#include "UnetPreprocessor.h"
#include "UnetInference.h"
#include "UnetPostprocessor.h"
#include "UnetIO.h"
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

// JSON配置接口实现
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
	CImg<short> cropped_volume = UnetPreprocessor::cropToNonzero(input_cbct_volume, crop_bbox);
	
	// 创建seg_mask（为归一化准备）
	seg_mask = UnetPreprocessor::createSegMask(cropped_volume);
	
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
	// 调用UnetInference模块执行完整的推理流程
	return UnetInference::segModelInfer(this, config, input_volume);
}

AI_INT  UnetMain::getSegMask(AI_DataInfo *dstData)
{
	// 调用UnetPostprocessor模块执行完整的后处理流程
	return UnetPostprocessor::processSegmentationMask(this, dstData);
}


void UnetMain::savePreprocessedData(const CImg<float>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || preprocessOutputPath.empty()) return;
	
	// 委托给UnetIO模块处理
	UnetIO::savePreprocessedData(this, data, preprocessOutputPath, filename);
}


void UnetMain::saveModelOutput(const CImg<float>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || modelOutputPath.empty()) return;
	
	// 委托给UnetIO模块处理
	UnetIO::saveModelOutput(data, modelOutputPath, filename);
}


void UnetMain::savePostprocessedData(const CImg<short>& data, const std::wstring& filename)
{
	if (!saveIntermediateResults || postprocessOutputPath.empty()) return;
	
	// 委托给UnetIO模块处理
	UnetIO::savePostprocessedData(this, data, postprocessOutputPath, filename);
}


void UnetMain::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z)
{
	if (!saveIntermediateResults || modelOutputPath.empty()) return;
	
	// 委托给UnetIO模块处理
	UnetIO::saveTile(tile, tileIndex, x, y, z, modelOutputPath);
}


