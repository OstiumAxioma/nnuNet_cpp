#ifndef _UnetMain__H
#define _UnetMain__H
#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "onnxruntime_cxx_api.h"
#include <torch/script.h>

#define cimg_display_type 2
#include "CImg.h"

#include "UnetSegAI_API.h"
#include "ConfigParser.h"

// ITK headers for image I/O
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

using namespace std;
using namespace cimg_library;


struct nnUNetConfig {
	const wchar_t* model_file_name;

	const char* cimg_transpose_forward;
	const char* cimg_transpose_backward;

	std::vector<int> transpose_forward;
	std::vector<int> transpose_backward;

	int input_channels;
	int num_classes;

	short mandible_label; //1
	short maxilla_label;
	short sinus_label;
	short ian_label; //4
	short uppertooth_label;
	short lowertooth_label;

	std::vector<float> voxel_spacing;
	std::vector<int64_t> patch_size;
	float step_size_ratio;

	std::string normalization_type;
	std::string task_type; //"classification, segmentation, regression, detection"
	std::vector<float> min_max_HU;
	std::vector<float> mean_std_HU;
	
	// 直接访问的intensity properties
	double mean;  // 改为double提高精度
	double std;   // 改为double提高精度
	
	// percentile值用于CT归一化裁剪
	double percentile_00_5;  // 改为double提高精度
	double percentile_99_5;  // 改为double提高精度
	
	// 归一化相关参数
	bool use_mask_for_norm;

	bool use_mirroring;
};

// 预处理相关结构体定义
struct CropBBox {
	int x_min, x_max, y_min, y_max, z_min, z_max;
};

class UnetMain
{
	// 声明友元类以访问私有成员
	friend class UnetPreprocessor;
	friend class UnetInference;
	friend class UnetTorchInference;
	friend class UnetPostprocessor;
	friend class UnetIO;

public:
	UnetMain();
	~UnetMain();

	static UnetMain *CreateUnetMain();

	void    setModelFns(const wchar_t* model_fn);

	void    setStepSizeRatio(float ratio);
	
	// 参数设置接口
	void    setPatchSize(int64_t x, int64_t y, int64_t z);
	void    setNumClasses(int classes);
	void    setInputChannels(int channels);
	void    setTargetSpacing(float x, float y, float z);
	void    setTransposeSettings(int forward_x, int forward_y, int forward_z, 
	                           int backward_x, int backward_y, int backward_z);
	void    setNormalizationType(const char* type);
	void    setIntensityProperties(float mean, float std, float min_val, float max_val,
	                             float percentile_00_5, float percentile_99_5);
	void    setUseMirroring(bool use_mirroring);
	
	// 新增：JSON配置接口
	bool    setConfigFromJsonString(const char* jsonContent);

	AI_INT  performInference(AI_DataInfo *srcData); //执行分割推理

	AI_INT  getSegMask(AI_DataInfo *dstData); //获取分割掩码
	
	// 设置输出路径
	void    setOutputPaths(const wchar_t* preprocessPath, const wchar_t* modelOutputPath, const wchar_t* postprocessPath);

private:
	// 模型后端类型
	enum class ModelBackend {
		ONNX,
		TORCH,
		UNKNOWN
	};
	
	bool   NETDEBUG_FLAG;
	bool   use_gpu;
	Ort::Env env;
	
	// 模型后端相关
	ModelBackend model_backend = ModelBackend::UNKNOWN;
	
	// TorchScript 模型相关
	torch::jit::script::Module torch_model;
	bool torch_model_loaded = false;

	//输入：原始CBCT体数据
	CImg<short> input_cbct_volume;

	double intensity_mean;  // 改为double提高精度
	double intensity_std;   // 改为double提高精度
	
	// 预处理相关成员变量
	CropBBox crop_bbox;  // 保存裁剪边界信息
	CImg<short> seg_mask;  // 用于归一化的mask（与Python的seg对应）

	int Width0;
	int Height0;
	int Depth0;

	std::vector<float> input_voxel_spacing;
	std::vector<float> transposed_input_voxel_spacing;
	// 保存原始spacing（从文件读取的真实物理spacing）
	std::vector<float> original_voxel_spacing;
	std::vector<float> transposed_original_voxel_spacing;
	
	// 保存图像元数据（origin, spacing, direction）
	struct ImageMetadata {
		double origin[3];
		double spacing[3];
		double direction[9];  // 3x3 direction matrix stored as 1D array
		
		ImageMetadata() {
			// 默认值
			origin[0] = origin[1] = origin[2] = 0.0;
			spacing[0] = spacing[1] = spacing[2] = 1.0;
			// 默认方向为单位矩阵
			direction[0] = direction[4] = direction[8] = 1.0;
			direction[1] = direction[2] = direction[3] = 0.0;
			direction[5] = direction[6] = direction[7] = 0.0;
		}
	} imageMetadata;

	//输出：分割结果三维Mask
	CImg<float> predicted_output_prob;
	CImg<short> output_seg_mask;

	Ort::SessionOptions session_options;

	//segmentation sessions ptr
	std::unique_ptr<Ort::Session> semantic_seg_session_ptr;
	
	// Session相关的缓存信息
	std::string cached_input_name;
	std::string cached_output_name;
	bool session_initialized;

	//模型配置参数
	nnUNetConfig unetConfig;
	
	// JSON配置解析器
	ConfigParser configParser;
	
	// 存储转置字符串（需要持久化存储）
	std::string transposeForwardStr;
	std::string transposeBackwardStr;
	
	// 输出路径设置
	std::wstring preprocessOutputPath;
	std::wstring modelOutputPath;
	std::wstring postprocessOutputPath;
	bool saveIntermediateResults;

	//设置CBCT输入数据
	AI_INT  setInput(AI_DataInfo *srcData); 

	AI_INT  setOnnxruntimeInstances();
	
	// 初始化ONNX Session
	AI_INT  initializeSession();
	
	// 初始化 TorchScript 模型
	AI_INT  initializeTorchModel();
	
	// 辅助函数
	ModelBackend detectModelBackend(const wchar_t* model_path);
	std::string wstringToString(const std::wstring& wstr);

	// 以下函数已移至相应的模块类：
	// segModelInfer -> UnetPreprocessor::preprocessVolume + UnetInference::runSlidingWindow
	// CTNormalization -> UnetPreprocessor::CTNormalization
	// crop_to_nonzero -> UnetPreprocessor::cropToNonzero
	// binaryFillHoles3d -> UnetPreprocessor内部函数
	// slidingWindowInfer -> UnetInference::runSlidingWindow
	// argmax_spectrum -> UnetPostprocessor::argmaxSpectrum
	
	AI_INT   postProcessing();// 对分割结果进行后处理（已废弃）
	
	// 注意：save函数已移至UnetIO类作为静态方法
	// 这些声明保留是为了兼容性，但实际实现已被移除
	// 新代码应该直接使用 UnetIO::saveXXX 函数
};
#endif //_UnetMain__H