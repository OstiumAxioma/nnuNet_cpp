#ifndef _DentalUnet__H
#define _DentalUnet__H
#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

#define cimg_display_type 2
#include "../utility/CImg/CImg.h"

#include "DentalCbctSegAI_API.h"

using namespace std;
using namespace cimg_library;


struct nnUNetConfig {
	std::string model_file_name;

	std::vector<int> transpose_forward;
	std::vector<int> transpose_backward;

	int input_channels;
	int num_classes;

	std::vector<float> voxel_spacing;
	std::vector<int64_t> patch_size;
	float step_size_ratio;

	std::string normalization_type;
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

	void    setModelFns(std::string, std::string);

	void    setStepSizeRatio(float ratio);

	AI_INT  performInference(AI_DataInfo *srcData); //执行分割流程

	AI_INT  getSegMask(AI_DataInfo *dstData); //获取分割结果

	void    setDnnOptions(); //参数设置，是否cuda、opengl，扩展用
	void    setAlgParameter();

private:
	bool   NETDEBUG_FLAG;
	bool   use_cuda;

	//输入：输入CBCT体数据
	CImg<short> inputCbctVolume;
	CImg<float> workVolume;
	torch::Tensor input_volume_tensor;

	int Width0;
	int Height0;
	int Depth0;

	std::vector<float> input_voxel_spacing;

	//输出：分割结果，三维Mask
	CImg<short> segMaskVolume;
	torch::Tensor seg_label_tensor;
	CImg<float> predSegProbVolume;

	//segmentation model
	torch::jit::script::Module  dentalStructureSegModule;
	torch::jit::script::Module  dentalIANSegModule;

	// 载入分割模型
	AI_INT  loadSegModels();

	//模型内置参数
	nnUNetConfig structureSegModelConfig;
	nnUNetConfig ianSegModelConfig;

	//输入CBCT体数据
	AI_INT  setInput(AI_DataInfo *srcData); 

	AI_INT  segModelInfer(torch::jit::script::Module& model, nnUNetConfig config);
	AI_INT  segModelInfer0(torch::jit::script::Module& model, nnUNetConfig config);

	torch::Tensor CTNormalization(torch::Tensor input_volume, nnUNetConfig config);
	torch::Tensor create_3d_gaussian_kernel(const std::vector<int64_t>& window_sizes);
	//AI_INT        sliding_window_inference(torch::jit::script::Module& model, torch::Tensor input_volume, nnUNetConfig config);
	AI_INT        sliding_window_inference(torch::jit::script::Module& model, nnUNetConfig config);
	torch::Tensor resize_volume(torch::Tensor& input_volume, const std::vector<int64_t>& output_size);
};

#endif