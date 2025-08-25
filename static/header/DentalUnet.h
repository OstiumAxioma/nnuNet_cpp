#ifndef _DentalUnet__H
#define _DentalUnet__H
#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "onnxruntime_cxx_api.h"

#define cimg_display_type 2
#include "CImg.h"

#include "DentalCbctSegAI_API.h"
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

	bool use_mirroring;
};


class DentalUnet
{
public:
	DentalUnet();
	~DentalUnet();

	static DentalUnet *CreateDentalUnet();

	void    setModelFns(const wchar_t* model_fn);

	void    setStepSizeRatio(float ratio);
	
	// 新增：参数设置接�?
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

	AI_INT  performInference(AI_DataInfo *srcData); //ִ�зָ�����

	AI_INT  getSegMask(AI_DataInfo *dstData); //��ȡ�ָ���

	void    setDnnOptions(); //�������ã��Ƿ�cuda��opengl����չ��
	void    setAlgParameter();
	
	// �����������·��?
	void    setOutputPaths(const wchar_t* preprocessPath, const wchar_t* modelOutputPath, const wchar_t* postprocessPath);

private:
	bool   NETDEBUG_FLAG;
	bool   use_gpu;
	Ort::Env env;

	//���룺����CBCT������
	CImg<short> input_cbct_volume;

	float intensity_mean;
	float intensity_std;

	int Width0;
	int Height0;
	int Depth0;

	std::vector<float> input_voxel_spacing;
	std::vector<float> transposed_input_voxel_spacing;
	// 新增：保存原始spacing（从文件读取的真实物理spacing�?
	std::vector<float> original_voxel_spacing;
	std::vector<float> transposed_original_voxel_spacing;
	
	// 新增：保存图像元数据（origin, spacing, direction�?
	struct ImageMetadata {
		double origin[3];
		double spacing[3];
		double direction[9];  // 3x3 direction matrix stored as 1D array
		
		ImageMetadata() {
			// 默认�?
			origin[0] = origin[1] = origin[2] = 0.0;
			spacing[0] = spacing[1] = spacing[2] = 1.0;
			// 默认方向为单位矩�?
			direction[0] = direction[4] = direction[8] = 1.0;
			direction[1] = direction[2] = direction[3] = 0.0;
			direction[5] = direction[6] = direction[7] = 0.0;
		}
	} imageMetadata;

	//������ָ�������άMask
	CImg<float> predicted_output_prob;
	CImg<short> output_seg_mask;

	Ort::SessionOptions session_options;

	//segmentation sessions ptr
	//std::unique_ptr<Ort::Session> semantic_seg_session_ptr;
	//std::unique_ptr<Ort::Session> ian_seg_session_ptr;

	//ģ�����ò���
	nnUNetConfig unetConfig;
	
	// JSON����������
	ConfigParser configParser;
	
	// �������·��?
	std::wstring preprocessOutputPath;
	std::wstring modelOutputPath;
	std::wstring postprocessOutputPath;
	bool saveIntermediateResults;

	//����CBCT������
	AI_INT  setInput(AI_DataInfo *srcData); 

	AI_INT  initializeOnnxruntimeInstances();

	//uunet�ָ�ģ�ͻ����������ָ�
	AI_INT  segModelInfer(nnUNetConfig config, CImg<short> input_volume);
	AI_INT  slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume);

	AI_INT   postProcessing();// �Էָ������к���

	void    CTNormalization(CImg<float>& input_volume, nnUNetConfig config);
	void    create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes);
	CImg<short> argmax_spectrum(const CImg<float>& input);
	
	// �����������ļ��ķ���
	void    savePreprocessedData(const CImg<float>& data, const std::wstring& filename);
	void    saveModelOutput(const CImg<float>& data, const std::wstring& filename);
	void    savePostprocessedData(const CImg<short>& data, const std::wstring& filename);
	void    saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z);
};
#endif