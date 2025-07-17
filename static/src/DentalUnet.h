#ifndef _DentalUnet__H
#define _DentalUnet__H
#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <onnxruntime_cxx_api.h>

#define cimg_display_type 2
#include "./utility/CImg/CImg.h"

#include "DentalCbctSegAI_API.h"

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

	AI_INT  performInference(AI_DataInfo *srcData); //执行分割流程

	AI_INT  getSegMask(AI_DataInfo *dstData); //获取分割结果

	void    setDnnOptions(); //参数设置，是否cuda、opengl，扩展用
	void    setAlgParameter();

private:
	bool   NETDEBUG_FLAG;
	bool   use_gpu;
	Ort::Env env;

	//输入：输入CBCT体数据
	CImg<short> input_cbct_volume;

	float intensity_mean;
	float intensity_std;

	int Width0;
	int Height0;
	int Depth0;

	std::vector<float> input_voxel_spacing;
	std::vector<float> transposed_input_voxel_spacing;

	//输出：分割结果，三维Mask
	CImg<float> predicted_output_prob;
	CImg<short> output_seg_mask;

	Ort::SessionOptions session_options;

	//segmentation sessions ptr
	//std::unique_ptr<Ort::Session> semantic_seg_session_ptr;
	//std::unique_ptr<Ort::Session> ian_seg_session_ptr;

	//模型内置参数
	nnUNetConfig unetConfig;

	//输入CBCT体数据
	AI_INT  setInput(AI_DataInfo *srcData); 

	AI_INT  initializeOnnxruntimeInstances();

	//uunet分割模型滑动窗推理分割
	AI_INT  segModelInfer(nnUNetConfig config, CImg<short> input_volume);
	AI_INT  slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume);

	AI_INT   postProcessing();// 对分割结果进行后处理

	void    CTNormalization(CImg<float>& input_volume, nnUNetConfig config);
	void    create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes);
	CImg<short> argmax_spectrum(const CImg<float>& input);
};
#endif