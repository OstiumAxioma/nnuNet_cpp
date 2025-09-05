#ifndef _DentalUnet__H
#define _DentalUnet__H
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include "cc3d.hpp"

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <unordered_map>

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
	std::vector<float> min_max_HU;
	std::vector<float> mean_std_HU;
	bool use_mirroring;

	bool remove_metal_markers;
	float marker_diameter;
};


class DentalUnet
{
public:
	DentalUnet();
	~DentalUnet();

	static DentalUnet *CreateDentalUnet();

	void    setModelFns(std::string);// ����ģ���ļ�·��

	void    setStepSizeRatio(float ratio);// ���û�������������

	void    setMarkerBallDiameter(float diameter, bool remove_metal_markers); //���ñ����ֱ������λ��mm

	AI_INT  performInference(AI_DataInfo *srcData); //ִ�зָ�����

	AI_INT  getSegMask(AI_DataInfo *dstData); //��ȡ�ָ���

	AI_INT  getMarkerCount();		//��ȡ���������

	AI_INT  getMarkerInfo(float* markerInfo);		//��ȡ�����������Ϣ

	void    setDnnOptions(); //�������ã��Ƿ�cuda��opengl����չ��
	void    setAlgParameter();

private:
	bool   NETDEBUG_FLAG;
	bool   use_cuda;
	bool   model_is_loaded;

	torch::Tensor input_volume_tensor;//���룺����CBCT������
	float intensity_mean;
	float intensity_std;

	int Width0;
	int Height0;
	int Depth0;

	std::vector<float> input_voxel_spacing;
	std::vector<float> transposed_input_voxel_spacing;


	CImg<float> pred_prob_volume;
	torch::Tensor seg_label_tensor;//������ָ�������άMask

	torch::jit::script::Module  unetModule;//segmentation model
	nnUNetConfig unetConfig; //ģ�����ò���

	//����CBCT������
	AI_INT  setInput(AI_DataInfo *srcData); 

	AI_INT  loadSegModels();// ����ָ�ģ��

	//uunet�ָ�ģ�ͻ���������ָ�
	torch::Tensor  segModelInfer(torch::jit::script::Module& model, torch::Tensor input_volume, nnUNetConfig config);
	torch::Tensor  sliding_window_inference(torch::jit::script::Module& model, torch::Tensor input_volume, nnUNetConfig config);

	AI_INT   removeMetalBalls(float diameter, short metal_thresh);//ȥ���ָ����еĽ������մɱ����

	std::vector<float> markerPosition;		//���ҵ��ı���������

	AI_INT   postProcessing();// �Էָ������к���

	torch::Tensor  CTNormalization(torch::Tensor input_volume, nnUNetConfig config);
	torch::Tensor  create_3d_gaussian_kernel(const std::vector<int64_t>& window_sizes);

	torch::Tensor  resize_volume(torch::Tensor& input_volume, const std::vector<int64_t>& output_size);
	torch::Tensor  compute_boundingbox(torch::Tensor mask);
};


#endif