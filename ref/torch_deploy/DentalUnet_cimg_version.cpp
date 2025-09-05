#include "DentalUnet.h"

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = true;

	std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
	//use_cuda = torch::cuda::is_available();
	use_cuda = false;

	std::cout << "use_cuda: " << use_cuda << endl;

	/*
	structureSegModelConfig.model_file_name = "..\\models\\dentalCBCT_Segmentator.pt";
	structureSegModelConfig.input_channels = 1;
	structureSegModelConfig.num_classes = 6;
	structureSegModelConfig.transpose_forward  = { 1, 0, 2};//2 0 1
	structureSegModelConfig.transpose_backward = { 1, 0, 2};
	structureSegModelConfig.voxel_spacing = { 0.4f, 0.4f, 0.4f };
	structureSegModelConfig.patch_size = { 128, 128, 128 };
	structureSegModelConfig.step_size_ratio = 0.75f;
	structureSegModelConfig.normalization_type = "ZScoreNormalization";
	structureSegModelConfig.min_max_HU = { -208.0f,  3070.0f };
	structureSegModelConfig.mean_std_HU = { 1178.261474609375f, 611.7098999023438f };
	structureSegModelConfig.use_mirroring = false;
	*/

	structureSegModelConfig.model_file_name = "..\\models\\kneeCartilageMR_unet_seger_cpu.pt";
	structureSegModelConfig.input_channels = 1;
	structureSegModelConfig.num_classes = 5;
	structureSegModelConfig.transpose_forward = { 0, 1, 2 };//2 0 1
	structureSegModelConfig.transpose_backward = { 0, 1, 2 };
	structureSegModelConfig.voxel_spacing = { 0.3906f, 0.3906f, 1.0f }; //x, y, z: width, height depth
	structureSegModelConfig.patch_size = { 192, 160, 64 };//y, x, z: height, width,  depth
	structureSegModelConfig.step_size_ratio = 0.5f;
	structureSegModelConfig.normalization_type = "ZScoreNormalization";
	structureSegModelConfig.min_max_HU = { -208.0f,  3070.0f };
	structureSegModelConfig.mean_std_HU = { 1178.261474609375f, 611.7098999023438f };
	structureSegModelConfig.use_mirroring = false;


	ianSegModelConfig.model_file_name = "..\\models\\dental_segmentator4IAN.pt";
	ianSegModelConfig.input_channels = 1;
	ianSegModelConfig.num_classes = 6;
	ianSegModelConfig.transpose_forward  = { 2, 0, 1 };//{ 1, 0, 2 };
	ianSegModelConfig.transpose_backward = { 1, 2, 0 };//{ 1, 0, 2 };
	ianSegModelConfig.voxel_spacing = { 0.43164101243019104f, 0.43164101243019104f, 0.31200000643730164f };
	ianSegModelConfig.patch_size = { 128, 128, 128 };
	ianSegModelConfig.step_size_ratio = 0.75f;
	ianSegModelConfig.normalization_type = "CTNormalization";
	ianSegModelConfig.min_max_HU = { -208.0f,  3070.0f };
	ianSegModelConfig.mean_std_HU = { 1178.261474609375f, 611.7098999023438f };
	ianSegModelConfig.use_mirroring = false;
}


DentalUnet::~DentalUnet()
{
}


DentalUnet *DentalUnet::CreateDentalUnet()
{
	DentalUnet *segUnetModel = new DentalUnet();

	//if (segUnetModel->loadSegModels() != DentalCbctSegAI_STATUS_SUCCESS)
	//{
		//delete segUnetModel;
		//return NULL;
	//}

	return segUnetModel;
}

void  DentalUnet::setModelFns(std::string structure_model_fn, std::string ian_model_fn)
{
	structureSegModelConfig.model_file_name = structure_model_fn;
	ianSegModelConfig.model_file_name = ian_model_fn;
}


void  DentalUnet::setStepSizeRatio(float ratio)
{
	if (ratio <= 1.f && ratio >= 0.f)
	{
		structureSegModelConfig.step_size_ratio = ratio;
		ianSegModelConfig.step_size_ratio = ratio;
	}
	else
	{
		structureSegModelConfig.step_size_ratio = 0.5f;
		ianSegModelConfig.step_size_ratio = 0.5f;
	}
}

void  DentalUnet::setDnnOptions()
{
	//保留，扩展硬件加速设置用
}


void  DentalUnet::setAlgParameter()
{
	//保留，设置算法参数
}


AI_INT  DentalUnet::performInference(AI_DataInfo *srcData)
{
	int input_status = setInput(srcData);
	if (input_status != DentalCbctSegAI_STATUS_SUCCESS)
		return input_status;

	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);

	dentalStructureSegModule = torch::jit::load(structureSegModelConfig.model_file_name, device);
	dentalStructureSegModule.eval();

	//dentalIANSegModule = torch::jit::load("F://TorchProjects//DentalCbctSegmentation_v3//models//dental_segmentator4IAN.pt", device);
	//dentalIANSegModule.eval();

	//input_volume_tensor = input_volume_tensor.permute({ianSegModelConfig.transpose_forward[0], ianSegModelConfig.transpose_forward[1], ianSegModelConfig.transpose_forward[2] });
	//input_volume_tensor = input_volume_tensor.contiguous();
	//std::cout << "transposed input_volume_tensor size: " << input_volume_tensor.sizes() << endl;

	//apply CNN
	segModelInfer(dentalStructureSegModule, structureSegModelConfig);
	//segModelInfer(dentalIANSegModule, ianSegModelConfig);

	//seg_label_tensor = seg_label_tensor.permute({ ianSegModelConfig.transpose_backward[0], ianSegModelConfig.transpose_backward[1], ianSegModelConfig.transpose_backward[2] });
	//seg_label_tensor = seg_label_tensor.contiguous();

	//std::cout << "transposed seg_label_tensor size: " << seg_label_tensor.sizes() << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::loadSegModels()
{
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);

	// 加载分割模型
	try
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().

		//dentalStructureSegModule = torch::jit::load(structureSegModelConfig.model_file_name, device);
		//dentalStructureSegModule.eval();
		//dentalStructureSegModule.to(device);

		//dentalIANSegModule = torch::jit::load(ianSegModelConfig.model_file_name, device);
		//dentalIANSegModule.eval();
		//dentalIANSegModule.to(device);
		
		return DentalCbctSegAI_STATUS_SUCCESS;
	}
	catch (const c10::Error& e)
	{
		std::cerr << "error in loading the torch jit models. \n";
		return DentalCbctSegAI_LOADING_FAIED;
	}
}


AI_INT  DentalUnet::setInput(AI_DataInfo *srcData)
{
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //单位: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //单位: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //单位: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //单位: mm
	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ };

	float fovX = float(Width0) * voxelSpacingX;
	float fovY = float(Height0) * voxelSpacingY;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //输入体数据过小

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return DentalCbctSegAI_STATUS_VOLUME_LARGE; //输入体数据过大

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume过小
		return DentalCbctSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing过小
		return DentalCbctSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.f || voxelSpacingX > 1.f || voxelSpacingY > 1.f || voxelSpacingZ > 1.f)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //voxelSpacing多大

	// 拷贝输入数据到CImg对象
	////RAI 牙齿在前，后脑勺在后；耳朵在左右；下巴在上，头顶在下
	inputCbctVolume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(inputCbctVolume.data(), srcData->ptr_Data, volSize);

	//auto tensor_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
	//input_volume_tensor = torch::from_blob(srcData->ptr_Data, { Depth0, Width0, Height0 }, tensor_options);
	//std::cout << "input_volume size: " <<input_volume_tensor.sizes() << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


torch::Tensor  DentalUnet::CTNormalization(torch::Tensor input_volume, nnUNetConfig config)
{
	//HU值截断
	float min_hu4dentalCTNormalization = config.min_max_HU[0];
	float max_hu4dentalCTNormalization = config.min_max_HU[1];
	torch::Tensor clipped = torch::clamp(input_volume, min_hu4dentalCTNormalization, max_hu4dentalCTNormalization);

	//计算z-score
	float mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	float std_hu4dentalCTNormalization = config.mean_std_HU[1];
	return (clipped - mean_hu4dentalCTNormalization) / std_hu4dentalCTNormalization;
}


torch::Tensor  DentalUnet::create_3d_gaussian_kernel(const std::vector<int64_t>& window_sizes)
{
	std::vector<float> sigmas(3);
	for (int i = 0; i < 3; ++i)
		sigmas[i] = (window_sizes[i] - 1) / 6.0f; // 按W=6σ+1推导

	// 创建各维度坐标轴
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto x = torch::linspace(-(window_sizes[0] - 1) / 2.0, (window_sizes[0] - 1) / 2.0, window_sizes[0], options);
	auto y = torch::linspace(-(window_sizes[1] - 1) / 2.0, (window_sizes[1] - 1) / 2.0, window_sizes[1], options);
	auto z = torch::linspace(-(window_sizes[2] - 1) / 2.0, (window_sizes[2] - 1) / 2.0, window_sizes[2], options);

	// 计算各维度高斯分量（外积实现）
	auto gauss_x = torch::exp(-0.5 * x.pow(2) / pow(sigmas[0], 2));
	auto gauss_y = torch::exp(-0.5 * y.pow(2) / pow(sigmas[1], 2));
	auto gauss_z = torch::exp(-0.5 * z.pow(2) / pow(sigmas[2], 2));

	// 归一化各分量
	gauss_x /= gauss_x.sum();
	gauss_y /= gauss_y.sum();
	gauss_z /= gauss_z.sum();

	// 三维高斯核生成（通过外积）
	auto kernel = gauss_x.unsqueeze(-1).unsqueeze(-1)
		* gauss_y.unsqueeze(0).unsqueeze(-1)
		* gauss_z.unsqueeze(0).unsqueeze(0);


	return kernel / kernel.mean();
}


torch::Tensor DentalUnet::resize_volume(torch::Tensor& input_volume, const std::vector<int64_t>& output_size) {

	// 检查输入是否为5D张量 [batch, channel, depth, height, width]
	if (input_volume.dim() != 5) {
		throw std::runtime_error("Input tensor must be 5D");
	}

	bool align_corners = false;

	// 使用3D插值进行缩放
	auto options = torch::nn::functional::InterpolateFuncOptions().size(output_size).mode(torch::kTrilinear).align_corners(align_corners);

	return torch::nn::functional::interpolate(input_volume, options);
}


AI_INT DentalUnet::sliding_window_inference(torch::jit::script::Module& model, nnUNetConfig config)
{
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	//torch::Tensor gaussian_kernel = create_3d_gaussian_kernel(config.patch_size);
	torch::Tensor gaussian_kernel = create_3d_gaussian_kernel({ config.patch_size[2], config.patch_size[0], config.patch_size[1]});
	std::cout << "gaussian_kernel size: " << gaussian_kernel.sizes() << endl; //should be 128, 160,112
	//config.patch_size = { 192, 160, 64 };//y, x, z: height, width,  depth for tensor 

	int depth = workVolume.depth();
	int width = workVolume.width();
	int height = workVolume.height();

	std::cout << "normalized_input_volume mean: " << workVolume.mean() << endl;
	std::cout << "normalized_input_volume var: " << workVolume.variance() << endl;

	float step_size_ratio = config.step_size_ratio;
	float actualStepSize[3];
	int X_num_steps = (int)ceil(float(width - config.patch_size[1]) / (config.patch_size[1] * step_size_ratio)) + 1; //X
	if (X_num_steps > 1)
		actualStepSize[1] = float(width - config.patch_size[1]) / (X_num_steps - 1);
	else
		actualStepSize[1] = 999999.f;

	int Y_num_steps = (int)ceil(float(height - config.patch_size[0]) / (config.patch_size[0] * step_size_ratio)) + 1; //Y
	if (Y_num_steps > 1)
		actualStepSize[0] = float(height - config.patch_size[0]) / (Y_num_steps - 1);
	else
		actualStepSize[0] = 999999.f;

	int Z_num_steps = (int)ceil(float(depth - config.patch_size[2]) / (config.patch_size[2] * step_size_ratio)) + 1; //Y
	if (Z_num_steps > 1)
		actualStepSize[2] = float(depth - config.patch_size[2]) / (Z_num_steps - 1);
	else
		actualStepSize[2] = 999999.f;

	if (NETDEBUG_FLAG)
		std::cout << "Number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;

	// 初始化输出张量
	predSegProbVolume = CImg<float>(width, height, depth, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(width, height, depth, 1, 0.f);
	//std::cout << "predSegProbVolume shape: " << depth << width << height << endl;

	CImg<float> input_patch = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], 1, 0.f);

	CImg<float> win_weight = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], 1, 0.f);
	long patch_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
	std::memcpy(win_weight.data(), gaussian_kernel.data_ptr<float>(), patch_vol_sz);

	CImg<float> win_pob = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], config.num_classes, 0.f);
	patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	int patch_count = 0;
	for (int sx = 0; sx < X_num_steps; sx++)
	{
		int lb_x = (int)std::round(sx * actualStepSize[1]);
		int ub_x = lb_x + config.patch_size[1] - 1;
		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[0]);
			int ub_y = lb_y + config.patch_size[0] - 1;
			for (int sz = 0; sz < Z_num_steps; sz++)
			{
				int lb_z = (int)std::round(sz * actualStepSize[2]);
				int ub_z = lb_z + config.patch_size[2] - 1;

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << "current tile#: " << patch_count << endl;

				input_patch = workVolume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
				//torch::Tensor input_patch = input_volume.slice(0, lb_z, ub_z).slice(1, lb_y, ub_y).slice(2, lb_x, ub_x);

				std::cout << "input_patch mean: " << input_patch.mean() << endl;
				std::cout << "input_patch variance: " << input_patch.variance() << endl;

				//long input_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
				//CImg<float> input_slice_z = input_patch.get_slice(56);
				//input_slice_z.display("slice 56");

				torch::Tensor input_patch_tensor = torch::from_blob(input_patch.data(), {1, 1, config.patch_size[2], config.patch_size[0], config.patch_size[1] }, options);
				std::cout << "input_patch size: " << input_patch_tensor.sizes() << endl;

				//input_patch_tensor = input_patch_tensor.squeeze();
				//input_patch_tensor = input_patch_tensor.permute({config.transpose_forward[0], config.transpose_forward[1], config.transpose_forward[2] });
				//input_patch_tensor = input_patch_tensor.contiguous();
				//std::cout << "transposed input_patch size: " << input_patch_tensor.sizes() << endl; //should be 128, 160,112

				input_patch_tensor = input_patch_tensor.to(device);

				//std::vector<torch::jit::IValue> inputs;
				//inputs.push_back(input_patch.unsqueeze(0).unsqueeze(0)); // 添加batch和channel维度
				//auto output_patch = model.forward(inputs).toTensor();

				torch::Tensor output_patch_tensor = model.forward({ input_patch_tensor }).toTensor();

				output_patch_tensor = output_patch_tensor.to(torch::kCPU);
				std::cout << "output_patch size: "<< output_patch_tensor.sizes() << endl;

				//output_patch_tensor = output_patch_tensor.permute({ 0, config.transpose_backward[0] + 1, config.transpose_backward[1] + 1, config.transpose_backward[2] + 1 });
				//output_patch_tensor = output_patch_tensor.contiguous();
				//output_patch_tensor = output_patch_tensor.unsqueeze(0);
				//std::cout << "transposed output_patch size: " << output_patch_tensor.sizes() << endl;

				/*
				torch::Tensor predicted_label_patch = output_patch_tensor.argmax(0).squeeze();
				predicted_label_patch = predicted_label_patch.to(torch::kInt16);
				CImg<short> tmp_label_patch = CImg<short>(config.patch_size[1], config.patch_size[0], config.patch_size[2], 1, 0);
				long input_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(short);
				std::memcpy(tmp_label_patch.data(), predicted_label_patch.data_ptr<short>(), input_vol_sz);
				CImg<short> slice_z = tmp_label_patch.get_slice(56);
				slice_z.display("slice 60");
				*/
				
				torch::Tensor weighted_output_patch = output_patch_tensor * gaussian_kernel;
				std::memcpy(win_pob.data(), weighted_output_patch.data_ptr<float>(), patch_vol_sz);

				cimg_forXYZC(win_pob, x, y, z, c){
					predSegProbVolume(lb_x + x, lb_y + y, lb_z + z, c) += win_pob(x, y, z, c);
				}
				cimg_forXYZ(win_weight, x, y, z){
					count_vol(lb_x + x, lb_y + y, lb_z + z) += win_weight(x, y, z);
				}
			}
		}
	}
	// 归一化输出
	cimg_forXYZC(predSegProbVolume, x, y, z, c) {
		predSegProbVolume(x, y, z, c) /= count_vol(x, y, z);
	}
	
	//output.div(count_map);
	//torch::Tensor output = torch::from_blob(output_prob_vol.data(), { 1, config.num_classes, depth, width, height}, options);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);

	std::memcpy(dstData->ptr_Data, segMaskVolume.data(), volSize);

	//segMaskVolume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	//std::memcpy(segMaskVolume.data(), seg_label_tensor.data_ptr<short>(), volSize);
	//segMaskVolume.save_analyze("segMaskVolume.hdr");

	//std::memcpy(dstData->ptr_Data, seg_label_tensor.data_ptr<short>(), volSize);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::segModelInfer(torch::jit::script::Module& model, nnUNetConfig config)
{
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);
	auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	bool align_corners = false;

	if (input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// 计算目标尺寸
	int s_height = std::round( Height0 * input_voxel_spacing[0] / config.voxel_spacing[0]);
	int s_width =  std::round( Width0  * input_voxel_spacing[1] / config.voxel_spacing[1]);
	int s_depth  = std::round( Depth0  * input_voxel_spacing[2] / config.voxel_spacing[2] );

	//workVolume = inputCbctVolume.get_resize(s_width, s_height, s_depth, -100, 3);

	//归一化处理
	//short min_hu4dentalCTNormalization = (short)config.min_max_HU[0];
	//short max_hu4dentalCTNormalization = (short)config.min_max_HU[1];
	//workVolume.cut(min_hu4dentalCTNormalization, max_hu4dentalCTNormalization);

	//计算z-score
	//float mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	//float std_hu4dentalCTNormalization = config.mean_std_HU[1];
	//workVolume -= mean_hu4dentalCTNormalization;
	//workVolume /= std_hu4dentalCTNormalization;

	float intensity_mu = inputCbctVolume.mean();
	float intensity_std = inputCbctVolume.variance();
	intensity_std = std::sqrt(intensity_std);
	workVolume = inputCbctVolume - intensity_mu;
	workVolume /= (intensity_std + 0.00000001f);

	std::cout << "scaled_input_volume mean: " << workVolume.mean() << endl;
	std::cout << "scaled_input_volume var: " << workVolume.variance() << endl;

	//CImg<float> slice_z = workVolume.get_slice( 56);
	//slice_z.display("slice 56");

	//使用3D插值进行缩放
	//auto options = torch::nn::functional::InterpolateFuncOptions().size(output_size).mode(torch::kTrilinear).align_corners(align_corners);
	//input_volume = input_volume.to(torch::kFloat32);
	//torch::Tensor scaled_input_volume = torch::nn::functional::interpolate(input_volume.unsqueeze(0).unsqueeze(0), options);
	//scaled_input_volume = scaled_input_volume.squeeze();

	//归一化处理
	//scaled_input_volume = CTNormalization(scaled_input_volume, config);
	//std::cout << "normalized_input_volume mean: " << scaled_input_volume.mean() << endl;
	//std::cout << "normalized_input_volume std: " << scaled_input_volume.std() << endl;


	//滑动窗推理预测
	sliding_window_inference(model, config);
	//sliding_window_inference(model, scaled_input_volume, config);

	//std::cout << "predicted_output_prob size: " << predSegProbVolume.size << endl;
	//使用3D插值进行缩放
	//auto final_output_shape = input_volume.sizes().vec();
	//final_output_shape.insert(final_output_shape.begin(), config.num_classes); // 分割class_num类
	//std::vector<int64_t> final_output_size;
	//for (int i = 0; i < 3; ++i) {  // 只处理空间维度
		//final_output_size.push_back(static_cast<int64_t>(input_sizes[i]));
	//}
	//std::cout << "final_output_size: " << final_output_size << endl;

	//predicted_output_prob = predicted_output_prob.to(device);
	//options = torch::nn::functional::InterpolateFuncOptions().size(final_output_size).mode(torch::kTrilinear).align_corners(align_corners);
	//predicted_output_prob = torch::nn::functional::interpolate(predicted_output_prob, options);

	//predSegProbVolume.resize(Width0, Height0, Depth0, config.num_classes, 3);

	auto tmp_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor predicted_prob_tensor = torch::from_blob(predSegProbVolume.data(), {1, config.num_classes, Depth0, Height0, Width0}, tmp_option);

	torch::Tensor predicted_label_tensor = predicted_prob_tensor.argmax(1);
	predicted_label_tensor = predicted_label_tensor.to(torch::kInt16);
	std::cout << "final predicted_label size: " << predicted_label_tensor.sizes() << endl;

	predicted_prob_tensor.resize_(at::IntArrayRef{ 0 });
	predSegProbVolume.clear();

	segMaskVolume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long vol_sz = Depth0 * Width0 * Height0 * sizeof(short);
	std::memcpy(segMaskVolume.data(), predicted_label_tensor.data_ptr<short>(), vol_sz);

	predicted_label_tensor.resize_(at::IntArrayRef{ 0 });
	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::segModelInfer0(torch::jit::script::Module& model, nnUNetConfig config)
{
	bool is_use_gpu = torch::cuda::is_available();
	torch::Device device(is_use_gpu ? torch::kCUDA : torch::kCPU, 0);

	auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);// .device(torch::kCPU);

	bool isCropped = false;

	inputCbctVolume.load_analyze("S7_64.hdr");

	//rescale volume
	int s_width0  = inputCbctVolume.width();
	int s_height0 = inputCbctVolume.height();
	int s_depth0  = inputCbctVolume.depth();
	int s_width1  = s_width0;
	int s_height1 = s_height0;
	int s_depth1  = s_depth0;

	bool isVolumeScaled = false;

	float intensity_mu = 60.f;
	float intensity_std = 50.f;

	intensity_mu = inputCbctVolume.mean();
	intensity_std = inputCbctVolume.variance();
	intensity_std = std::sqrt(intensity_std);

	if (NETDEBUG_FLAG){
		std::cout << "mean: " << intensity_mu << endl;
		std::cout << "std: " << intensity_std << endl;
	}
	workVolume = inputCbctVolume - intensity_mu;
	workVolume /= (intensity_std + 0.00000001f);

	CImg<float> pred_prob_vol = CImg<float>(s_width1, s_height1, s_depth1, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(s_width1, s_height1, s_depth1, 1, 0.f);


	CImg<float> mr_patch = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], 1, 0.f);
	at::Tensor mr_patch_tensor, mr_patch_tensor1, pred_prob_tensor, pred_prob_tensor1;

	CImg<float> weight_win = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], 1, 1.f);
	cimg_forXYZ(weight_win, x, y, z)
	{
		if (x < 8)
			weight_win(x, y, z) = 0.5f;
		if (y < 8)
			weight_win(x, y, z) = 0.5f;
		if (z < 8)
			weight_win(x, y, z) = 0.5f;
		if (x >= (config.patch_size[0] - 9))
			weight_win(x, y, z) = 0.5f;
		if (y >= (config.patch_size[1] - 9))
			weight_win(x, y, z) = 0.5f;
		if (z >= (config.patch_size[2] - 9))
			weight_win(x, y, z) = 0.5f;
	}

	float actualStepSize[3];
	float tile_step_size = 0.f;
	tile_step_size = config.patch_size[1] * config.step_size_ratio;
	int X_num_steps = (int)ceil(float(s_width1 - config.patch_size[1]) / tile_step_size) + 1; //X
	if (X_num_steps > 1)
		actualStepSize[1] = float(s_width1 - config.patch_size[1]) / (X_num_steps - 1);
	else
		actualStepSize[1] = 999999.f;

	tile_step_size = config.patch_size[0] * config.step_size_ratio;
	int Y_num_steps = (int)ceil(float(s_height1 - config.patch_size[0]) / tile_step_size) + 1; //Y
	if (Y_num_steps > 1)
		actualStepSize[0] = float(s_height1 - config.patch_size[0]) / (Y_num_steps - 1);
	else
		actualStepSize[0] = 999999.f;

	tile_step_size = config.patch_size[2] * config.step_size_ratio;
	int Z_num_steps = (int)ceil(float(s_depth1 - config.patch_size[2]) / tile_step_size) + 1;
	if (Z_num_steps > 1)
		actualStepSize[2] = float(s_depth1 - config.patch_size[2]) / (Z_num_steps - 1);
	else
		actualStepSize[2] = 999999.f;

	if (NETDEBUG_FLAG)
		std::cout << "Number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;

	int patch_count = 0;
	for (int sx = 0; sx < X_num_steps; sx++)
	{
		int lb_x = (int)std::round(sx * actualStepSize[1]);//width
		int ub_x = lb_x + config.patch_size[1] - 1;
		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[0]);
			int ub_y = lb_y + config.patch_size[0] - 1;//height
			for (int sz = 0; sz < Z_num_steps; sz++)
			{
				int lb_z = (int)std::round(sz * actualStepSize[2]);
				int ub_z = lb_z + config.patch_size[2] - 1;

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << "current tile#: " << patch_count << endl;

				mr_patch = workVolume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
				mr_patch_tensor = torch::from_blob(mr_patch.data(), { 1, 1, config.patch_size[2], config.patch_size[0], config.patch_size[1] }, tensor_options);
				
				mr_patch_tensor = mr_patch_tensor.to(device);
				pred_prob_tensor = model.forward({ mr_patch_tensor }).toTensor();

				pred_prob_tensor = pred_prob_tensor.to(torch::kCPU);

				if (false)
				{
					mr_patch_tensor = mr_patch_tensor.flip({ 2 });
					pred_prob_tensor1 = model.forward({ mr_patch_tensor }).toTensor();

					pred_prob_tensor1 = pred_prob_tensor1.softmax(1).squeeze();
					pred_prob_tensor1 = pred_prob_tensor1.to(torch::kCPU);
					pred_prob_tensor1 = pred_prob_tensor1.flip({ 1 });

					pred_prob_tensor = 0.5f * pred_prob_tensor + 0.5f * pred_prob_tensor1;
				}

				CImg<float> pred_prob_patch = CImg<float>(config.patch_size[1], config.patch_size[0], config.patch_size[2], config.num_classes, 0.f);
				long patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
				std::memcpy(pred_prob_patch.data(), pred_prob_tensor.data_ptr<float>(), patch_vol_sz);

				cimg_forXYZC(pred_prob_patch, x, y, z, c)
				{
					pred_prob_vol(lb_x + x, lb_y + y, lb_z + z, c) += (pred_prob_patch(x, y, z, c) * weight_win(x, y, z));
				}
				cimg_forXYZ(weight_win, x, y, z)
				{
					count_vol(lb_x + x, lb_y + y, lb_z + z) += weight_win(x, y, z);
				}
			}
		}
	}

	workVolume.clear();
	mr_patch.clear();
	mr_patch_tensor.resize_(at::IntArrayRef{ 0 });
	//pred_prob_tensor.resize_(at::IntArrayRef{ 0 });

	float weight = 0.f;
	cimg_forXYZ(count_vol, x, y, z)
	{
		weight = count_vol(x, y, z);
		if (weight < 0.01f)
			pred_prob_vol(x, y, z, 0) = 10.f;
		else
		{
			for (int cc = 0; cc < config.num_classes; cc++)
			{
				pred_prob_vol(x, y, z, cc) /= weight;
			}
		}
	}

	//rescale to original voxel spacing
	if (isVolumeScaled)
		pred_prob_vol.resize(s_width0, s_height0, s_depth0, config.num_classes, 3);

	at::Tensor predProbTensor = torch::from_blob(pred_prob_vol.data(), { 1, config.num_classes, s_depth0, s_width0, s_height0 }, tensor_options);
	at::Tensor seg_label_tensor = predProbTensor.argmax(1).squeeze();
	seg_label_tensor = seg_label_tensor.to(torch::kInt16);

	pred_prob_vol.clear();
	predProbTensor.resize_(at::IntArrayRef{ 0 });

	segMaskVolume = CImg<short>(s_width0, s_height0, s_depth0, 1, 0);
	long vol_sz = s_depth0 * s_width0 * s_height0 * sizeof(short);
	std::memcpy(segMaskVolume.data(), seg_label_tensor.data_ptr<short>(), vol_sz);
	seg_label_tensor.resize_(at::IntArrayRef{ 0 });


	if (NETDEBUG_FLAG)
		std::cout << "unet processing is finished. " << std::endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}
