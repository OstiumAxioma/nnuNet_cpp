#include "DentalUnet.h"

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = false;

	use_cuda = torch::cuda::is_available();

	model_is_loaded = false;

	unetConfig.model_file_name = "..\\models\\dentalCBCT_Segmentator.pt";
	unetConfig.input_channels = 1;
	unetConfig.num_classes = 7;
	unetConfig.mandible_label = 1; 
	unetConfig.maxilla_label = 2;
	unetConfig.sinus_label = 3;
	unetConfig.ian_label = 4;
	unetConfig.uppertooth_label = 5;
	unetConfig.lowertooth_label = 6;
	unetConfig.transpose_forward  = { 0, 1, 2 };
	unetConfig.transpose_backward = { 0, 1, 2 };
	unetConfig.voxel_spacing = { 0.3f, 0.3f, 0.3f };
	unetConfig.patch_size = { 128, 128, 128 };
	unetConfig.step_size_ratio = 0.5f;
	unetConfig.normalization_type = "CTNormalization";
	unetConfig.min_max_HU = { -1000.f,  2433.f };
	unetConfig.mean_std_HU = { 667.2818175448842f, 758.9756833768081f };
	unetConfig.use_mirroring = true;

	unetConfig.remove_metal_markers = true;
	unetConfig.marker_diameter = 3.; //mm

	if (NETDEBUG_FLAG) {
		//use_cuda = false;
		std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
		std::cout << "use_cuda: " << use_cuda << endl;
		std::cout << "model file: " << unetConfig.model_file_name << endl;
	}
}


DentalUnet::~DentalUnet()
{
}


DentalUnet *DentalUnet::CreateDentalUnet()
{
	DentalUnet *segUnetModel = new DentalUnet();

	return segUnetModel;
}

void  DentalUnet::setModelFns(std::string model_fn)
{
	unetConfig.model_file_name = model_fn;

	std::cout << "updated model file: " << unetConfig.model_file_name << endl;
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

	std::cout << "updated step size ratio: " << unetConfig.step_size_ratio << endl;
}


void  DentalUnet::setMarkerBallDiameter(float diameter, bool remove_metal_markers)
{
	unetConfig.marker_diameter = diameter; //mm
	unetConfig.remove_metal_markers = remove_metal_markers;
	std::cout << "need to remove_metal_markers: " << unetConfig.remove_metal_markers << endl;
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
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);
	
	int input_status = setInput(srcData);
	if (input_status != DentalCbctSegAI_STATUS_SUCCESS)
		return input_status;

	if (!model_is_loaded) {
		AI_INT loading_status = loadSegModels();
		if (loading_status == DentalCbctSegAI_STATUS_SUCCESS)
			model_is_loaded = true;
		else
			return loading_status;
	}

	std::cout << "model loading is done. " << endl;

	auto input_shape = input_volume_tensor.sizes().vec();

	input_volume_tensor = input_volume_tensor.permute({ unetConfig.transpose_forward[0], unetConfig.transpose_forward[1], unetConfig.transpose_forward[2] });
	input_volume_tensor = input_volume_tensor.contiguous();
	if (NETDEBUG_FLAG) {
		std::cout << "transposed input_volume_tensor size: " << input_volume_tensor.sizes() << endl;
	}

	transposed_input_voxel_spacing.clear();
	for (int i = 0; i < 3; ++i) {
		transposed_input_voxel_spacing.push_back(input_voxel_spacing[unetConfig.transpose_forward[i]]);
	}

	//apply CNN
	seg_label_tensor = segModelInfer(unetModule, input_volume_tensor, unetConfig);
	std::cout << "Segmentation inference is done. " << endl;

	if (NETDEBUG_FLAG) {
		std::cout << "seg_label_tensor size: " << seg_label_tensor.sizes() << endl;
	}

	seg_label_tensor = seg_label_tensor.squeeze();
	seg_label_tensor = seg_label_tensor.permute({unetConfig.transpose_backward[0], unetConfig.transpose_backward[1], unetConfig.transpose_backward[2] });
	seg_label_tensor = seg_label_tensor.contiguous();
	if (NETDEBUG_FLAG) {
		std::cout << "transposed seg_label_tensor size: " << seg_label_tensor.sizes() << endl;
	}

	input_volume_tensor = input_volume_tensor.permute({ unetConfig.transpose_backward[0], unetConfig.transpose_backward[1], unetConfig.transpose_backward[2] });
	input_volume_tensor = input_volume_tensor.contiguous();

	if (unetConfig.remove_metal_markers)
		removeMetalBalls(unetConfig.marker_diameter, 3500);

	postProcessing();
	std::cout << "Post processing is done. " << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::loadSegModels()
{
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);

	// 加载分割模型
	try
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		unetModule = torch::jit::load(unetConfig.model_file_name);
		unetModule.eval();

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

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
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
	//RAI: 牙齿在前，后脑勺在后；耳朵在左右；下巴在上，头顶在下
	//inputCbctVolume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	//long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	//std::memcpy(inputCbctVolume.data(), srcData->ptr_Data, volSize);

	auto tensor_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
	input_volume_tensor = torch::from_blob(srcData->ptr_Data, {1, 1, Depth0, Height0, Width0}, tensor_options); //N, C, D, H, W
	input_volume_tensor = input_volume_tensor.squeeze();

	intensity_mean = input_volume_tensor.mean(torch::kFloat32).item<float>();
	intensity_std  = input_volume_tensor.to(torch::kFloat32).std().item().toFloat();
	if (intensity_std < 0.0001f)
		intensity_std = 0.0001f;

	input_voxel_spacing = { voxelSpacingZ, voxelSpacingY, voxelSpacingX }; // z Image depth, y Image height, x Image width

	if (NETDEBUG_FLAG) {
		std::cout << "input_volume size: " << input_volume_tensor.sizes() << endl;
		std::cout << "input_volume intensity_mean: " << intensity_mean << endl;
		std::cout << "input_volume intensity_std: " << intensity_std << endl;
	}
	
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
		sigmas[i] = (window_sizes[i] - 1) / 5.0f; // 按W=5σ+1推导

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


torch::Tensor  DentalUnet::compute_boundingbox(torch::Tensor mask)
{
	auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	torch::Tensor bbox = torch::zeros({3, 2}, options);

	auto sz = mask.sizes();

	torch::Tensor tmp_proj = mask.sum({ 2 }).squeeze(); //xy
	torch::Tensor tmp_profile = tmp_proj.sum({1}).squeeze();//x

	int x0 = 0;
	for (int i = 0; i < sz[0]; i++){
		if (tmp_profile[i].item().toFloat() > 0){
			x0 = i;
			break;
		}
	}
	int x1 = x0 + 127;
	for (int i = sz[0] - 1; i > x0; i--){
		if (tmp_profile[i].item().toFloat() > 0){
			x1 = i;
			break;
		}
	}

	tmp_profile = tmp_proj.sum({ 0 }).squeeze();//y
	int y0 = 0;
	for (int i = 0; i < sz[1]; i++) {
		if (tmp_profile[i].item().toFloat() > 0) {
			y0 = i;
			break;
		}
	}
	int y1 = y0 + 127;
	for (int i = sz[1] - 1; i > y0; i--) {
		if (tmp_profile[i].item().toFloat() > 0) {
			y1 = i;
			break;
		}
	}

	tmp_proj = mask.sum({ 0 }).squeeze(); //yz
	tmp_profile = tmp_proj.sum({ 0 }).squeeze();//z
	int z0 = 0;
	for (int i = 0; i < sz[2]; i++) {
		if (tmp_profile[i].item().toFloat() > 0) {
			z0 = i;
			break;
		}
	}
	int z1 = z0 + 127;
	for (int i = sz[2] - 1; i > z0; i--) {
		if (tmp_profile[i].item().toFloat() > 0) {
			z1 = i;
			break;
		}
	}

	bbox[0][0] = x0;
	bbox[0][1] = x1;
	bbox[1][0] = y0;
	bbox[1][1] = y1;
	bbox[2][0] = z0;
	bbox[2][1] = z1;

	return bbox;
}


torch::Tensor DentalUnet::sliding_window_inference(torch::jit::script::Module& model, torch::Tensor input_volume, nnUNetConfig config)
{
	torch::Tensor gaussian_kernel = create_3d_gaussian_kernel(config.patch_size);
	//std::cout << "gaussian_kernel size: " << gaussian_kernel.sizes() << endl;

	auto input_sizes = input_volume.sizes().vec();
	float step_size_ratio = config.step_size_ratio;

	//input_volume size: //N, C, D, H, W
	float actualStepSize[3];
	int Z_num_steps = (int)ceil(float(input_sizes[0] - config.patch_size[0]) / (config.patch_size[0] * step_size_ratio)) + 1; //Z
	if (Z_num_steps > 1)
		actualStepSize[0] = float(input_sizes[0] - config.patch_size[0]) / (Z_num_steps - 1);
	else
		actualStepSize[0] = 999999.f;

	int Y_num_steps = (int)ceil(float(input_sizes[1] - config.patch_size[1]) / (config.patch_size[1] * step_size_ratio)) + 1; //Y
	if (Y_num_steps > 1)
		actualStepSize[1] = float(input_sizes[1] - config.patch_size[1]) / (Y_num_steps - 1);
	else
		actualStepSize[1] = 999999.f;

	int X_num_steps = (int)ceil(float(input_sizes[2] - config.patch_size[2]) / (config.patch_size[2] * step_size_ratio)) + 1; //X
	if (X_num_steps > 1)
		actualStepSize[2] = float(input_sizes[2] - config.patch_size[2]) / (X_num_steps - 1);
	else
		actualStepSize[2] = 999999.f;

	if (NETDEBUG_FLAG)
		std::cout << "Number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;

	// 初始化输出张量
	//auto output_shape = input_volume.sizes().vec();
	//output_shape.insert(output_shape.begin(), config.num_classes); // 分割class_num类
	//torch::Tensor output = torch::zeros(output_shape, options);
	//torch::Tensor count_map = torch::zeros(output_shape, options);

	// 初始化输出张量
	pred_prob_volume = CImg<float>(input_sizes[2], input_sizes[1], input_sizes[0], config.num_classes, 0.f);

	CImg<float> count_vol = CImg<float>(input_sizes[2], input_sizes[1], input_sizes[0], 1, 0.f);
	if (NETDEBUG_FLAG) {
		std::cout << "input_volume size: " << input_volume.sizes() << endl;
	}

	CImg<float> win_weight = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], 1, 0.f);
	long patch_vol_sz = config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);
	std::memcpy(win_weight.data(), gaussian_kernel.data_ptr<float>(), patch_vol_sz);

	CImg<float> win_pob = CImg<float>(config.patch_size[2], config.patch_size[1], config.patch_size[0], config.num_classes, 0.f);
	patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	//model to device
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);
	model.to(device);

	//滑动窗推理
	int patch_count = 0;
	if (NETDEBUG_FLAG) {
		std::cout << "current tile#: ";
	}
	for (int sz = 0; sz < Z_num_steps; sz++)
	{
		int lb_z = (int)std::round(sz * actualStepSize[0]);
		int ub_z = lb_z + config.patch_size[0];

		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[1]);
			int ub_y = lb_y + config.patch_size[1];

			for (int sx = 0; sx < X_num_steps; sx++)
			{
				int lb_x = (int)std::round(sx * actualStepSize[2]);
				int ub_x = lb_x + config.patch_size[2];

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << patch_count << ", ";

				torch::Tensor input_patch = input_volume.slice(0, lb_z, ub_z).slice(1, lb_y, ub_y).slice(2, lb_x, ub_x);
				//std::cout << "input_patch mean: " << input_patch.mean() << endl;
				//std::cout << "input_patch std: " << input_patch.std() << endl;

				if (NETDEBUG_FLAG && patch_count % 10 == 0) {
					std::cout << endl;
					std::cout << "input_patch.size: " << input_patch.sizes() << endl;
				}

				input_patch = input_patch.unsqueeze(0).unsqueeze(0).to(device);

				//std::vector<torch::jit::IValue> inputs;
				//inputs.push_back(input_patch); // 添加batch和channel维度
				//auto output_patch = model.forward(inputs).toTensor();
				torch::Tensor output_patch = model.forward({ input_patch }).toTensor();

				if (config.use_mirroring && use_cuda)
				{
					input_patch = input_patch.flip({ 4 });
					torch::Tensor output_patch1 = model.forward({ input_patch }).toTensor();
					output_patch1 = output_patch1.flip({ 4 });
					output_patch = 0.5f * output_patch + 0.5f * output_patch1;
				}

				output_patch = output_patch.to(torch::kCPU);
				//std::cout << "output_patch size: "<<output_patch.sizes() << endl;
				//std::cout << "output_patch min: " << output_patch.min() << endl;
				//std::cout << "output_patch max: " << output_patch.max() << endl;
				//std::cout << "output_patch mean: " << output_patch.mean() << endl;

				torch::Tensor weighted_output_patch = output_patch * gaussian_kernel;
				std::memcpy(win_pob.data(), weighted_output_patch.data_ptr<float>(), patch_vol_sz);

				cimg_forXYZC(win_pob, x, y, z, c){
					pred_prob_volume(lb_x + x, lb_y + y, lb_z + z, c) += win_pob(x, y, z, c);
				}
				cimg_forXYZ(win_weight, x, y, z){
					count_vol(lb_x + x, lb_y + y, lb_z + z) += win_weight(x, y, z);
				}

				// 加权累加
				//output.slice(1, lb_z, ub_z).slice(2, lb_y, ub_y).slice(3, lb_x, ub_x).add_(output_patch * gaussian_kernel);
				//count_map.slice(1, lb_z, ub_z).slice(2, lb_y, ub_y).slice(3, lb_x, ub_x).add_(gaussian_kernel);
			}
		}
	}
	if (NETDEBUG_FLAG)
		std::cout << endl;
	
	model.to(torch::kCPU);

	// 归一化输出
	//output.div(count_map);
	cimg_forXYZC(pred_prob_volume, x, y, z, c) {
		pred_prob_volume(x, y, z, c) /= count_vol(x, y, z);
	}
	
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor output = torch::from_blob(pred_prob_volume.data(), { 1, config.num_classes, input_sizes[0], input_sizes[1], input_sizes[2] }, options);
	
	return output;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	long vol_sz = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(dstData->ptr_Data, seg_label_tensor.data_ptr<short>(), vol_sz);
	//std::memcpy(dstData->ptr_Data, final_seg_mask.data(), vol_sz);

	return DentalCbctSegAI_STATUS_SUCCESS;
}

AI_INT DentalUnet::getMarkerCount()
{
	if (markerPosition.empty())
		return 0;
	int markerSize = static_cast<int>(markerPosition.size() / 3);
	return markerSize;
}

AI_INT DentalUnet::getMarkerInfo(float* markerInfo)
{
	if (markerPosition.empty())
		return 0;
	int markerSize = static_cast<int>(markerPosition.size() / 3);
	std::copy(markerPosition.begin(), markerPosition.end(), markerInfo);
	return markerSize;
}


torch::Tensor  DentalUnet::segModelInfer(torch::jit::script::Module& model, torch::Tensor input_volume, nnUNetConfig config)
{
	torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU, 0);
	auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
	bool align_corners = false;

	if (transposed_input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// 计算目标尺寸
	bool is_volume_scaled = false;
	std::vector<int64_t> output_size;
	auto input_sizes = input_volume.sizes().vec();
	float scaled_factor = 1.f;
	for (int i = 0; i < 3; ++i) {  // 只处理空间维度
		scaled_factor = transposed_input_voxel_spacing[i] / config.voxel_spacing[i];
		int scaled_sz = std::round(input_sizes[i] * scaled_factor);
		if (scaled_factor < 0.9f || scaled_factor > 1.25f || scaled_sz < config.patch_size[i])
			is_volume_scaled = true;

		if (scaled_sz < config.patch_size[i])
			scaled_sz = config.patch_size[i];

		output_size.push_back(static_cast<int64_t>(scaled_sz));
	}
	if (NETDEBUG_FLAG) {
		std::cout << "expected output_size: " << output_size << endl;
	}

	torch::Tensor scaled_input_volume = input_volume.to(torch::kFloat32);
	//使用3D插值进行缩放
	if (is_volume_scaled) {
		auto options = torch::nn::functional::InterpolateFuncOptions().size(output_size).mode(torch::kTrilinear).align_corners(align_corners);
		scaled_input_volume = torch::nn::functional::interpolate(scaled_input_volume.unsqueeze(0).unsqueeze(0), options);
		scaled_input_volume = scaled_input_volume.squeeze();
	}
	if (NETDEBUG_FLAG) {
		std::cout << "scaled_input_volume size: " << scaled_input_volume.sizes() << endl;
		std::cout << "scaled_input_volume mean: " << scaled_input_volume.mean() << endl;
		std::cout << "scaled_input_volume std: " << scaled_input_volume.std() << endl;
	}
	
	//归一化处理
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

	switch (normlization_type){
	case 10:
		scaled_input_volume = CTNormalization(scaled_input_volume, config);
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
	if (NETDEBUG_FLAG) {
		std::cout << "normalized_input_volume mean: " << scaled_input_volume.mean() << endl;
		std::cout << "normalized_input_volume std: " << scaled_input_volume.std() << endl;
	}

	if (NETDEBUG_FLAG) {
		std::cout << "Normalization is done. Starting sliding window inference..." << endl;
	}

	//滑动窗推理预测
	torch::Tensor predicted_output_prob = sliding_window_inference(model, scaled_input_volume, config);
	scaled_input_volume.resize_(at::IntArrayRef{ 0 });

	if (NETDEBUG_FLAG) {
		std::cout << "Sliding window inference is done." << endl;
	}

	//使用3D插值进行缩放
	std::vector<int64_t> final_output_size;
	for (int i = 0; i < 3; ++i) {  // 只处理空间维度
		final_output_size.push_back(static_cast<int64_t>(input_sizes[i]));
	}

	if (NETDEBUG_FLAG) {
		std::cout << "predicted_output_prob size: " << predicted_output_prob.sizes() << endl;
		std::cout << "final_output_size: " << final_output_size << endl;
	}
	
	// 检查输入是否为5D张量 [batch, channel, depth, height, width]
	if (predicted_output_prob.dim() != 5) {
		throw std::runtime_error("InterpolateFuncOptions: Input tensor must be 5D");
	}

	if (is_volume_scaled) {
		auto options = torch::nn::functional::InterpolateFuncOptions().size(final_output_size).mode(torch::kTrilinear).align_corners(align_corners);
		predicted_output_prob = torch::nn::functional::interpolate(predicted_output_prob, options);
	}

	torch::Tensor predicted_label = predicted_output_prob.argmax(1).squeeze();
	predicted_label = predicted_label.to(torch::kInt16).to(torch::kCPU);

	if (NETDEBUG_FLAG) {
		std::cout << "final predicted_label size: " << predicted_label.sizes() << endl;
		//std::cout << "final predicted_label max: " << predicted_label.max() << endl;
	}
	
	predicted_output_prob.resize_(at::IntArrayRef{ 0 });
	pred_prob_volume.clear();

	if (NETDEBUG_FLAG) {
		std::cout << "Segmentaion is done. aha." << endl;
	}

	return predicted_label;
}


AI_INT  DentalUnet::removeMetalBalls(float diameter, short metal_thresh)
{
	//diameter = 3.f;
	//metal_thresh = 4500;
	// 重置之前的分割结果
	markerPosition.clear();
	auto sizes = seg_label_tensor.sizes();
	const int32_t depth = sizes[0], height = sizes[1], width = sizes[2];

	//at::Tensor bw_tensor = input_volume_tensor > metal_thresh;
	at::Tensor bw_tensor = seg_label_tensor == unetConfig.uppertooth_label;
	bw_tensor.logical_or_(seg_label_tensor == unetConfig.lowertooth_label);
	bw_tensor.logical_and_(input_volume_tensor > metal_thresh);

	bw_tensor = bw_tensor.to(torch::kInt16).contiguous();

	// 获取数据指针
	short* input_data_ptr = bw_tensor.data_ptr<short>();

	//std::cout << "cc3d processing..." << endl;
	// 调用CC3D函数
	uint32_t* cc_labels_ptr = cc3d::connected_components3d<short>(
		input_data_ptr, width, height, depth, 26);

	// 输出张量
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor cc_labels = torch::zeros({ depth, height, width }, options);
	cc_labels = torch::from_blob(cc_labels_ptr, { depth, height, width }, options);

	// 连通区域统计信息结构
	struct RegionStats {
		int32_t region_id;
		int64_t voxel_count;                // 体积(体素数)
		std::vector<double> centroid;  // 质心坐标
		std::vector<int32_t> bbox_min;  // 边界框最小坐标
		std::vector<int32_t> bbox_max;  // 边界框最大坐标
		double equivalent_diameter;     // 等效直径

		RegionStats() : region_id(0), voxel_count(0), equivalent_diameter(0.0) {
			centroid.resize(3, 0);
			bbox_min.resize(3, std::numeric_limits<int32_t>::max());
			bbox_max.resize(3, std::numeric_limits<int32_t>::min());
		}
	};

	auto accessor = cc_labels.accessor<int32_t, 3>();

	if (NETDEBUG_FLAG) {
		std::cout << "统计区域信息..." << endl;
	}
	std::unordered_map<int, RegionStats> region_data;
	// 第一遍遍历：初始化区域信息
	for (int32_t z = 0; z < depth; ++z) {
		for (int32_t y = 0; y < height; ++y) {
			for (int32_t x = 0; x < width; ++x) {
				int32_t region_id = accessor[z][y][x];
				if (region_id == 0) continue; // 跳过背景

				if (region_data.count(region_id) == 0) {
					region_data[region_id].region_id = region_id;
				}

				region_data[region_id].voxel_count++;
				region_data[region_id].centroid[0] += x;
				region_data[region_id].centroid[1] += y;
				region_data[region_id].centroid[2] += z;

				// 更新边界框
				region_data[region_id].bbox_min[0] = std::min(region_data[region_id].bbox_min[0], x);
				region_data[region_id].bbox_min[1] = std::min(region_data[region_id].bbox_min[1], y);
				region_data[region_id].bbox_min[2] = std::min(region_data[region_id].bbox_min[2], z);

				region_data[region_id].bbox_max[0] = std::max(region_data[region_id].bbox_max[0], x);
				region_data[region_id].bbox_max[1] = std::max(region_data[region_id].bbox_max[1], y);
				region_data[region_id].bbox_max[2] = std::max(region_data[region_id].bbox_max[2], z);
			}
		}
	}

	bw_tensor = bw_tensor.to(torch::kBool);
	bw_tensor.fill_(0);

	// 计算质心和等效直径
	int removed_ball_num = 0;
	for (auto elem : region_data) {
		elem.second.centroid[0] /= elem.second.voxel_count;
		elem.second.centroid[1] /= elem.second.voxel_count;
		elem.second.centroid[2] /= elem.second.voxel_count;

		// 等效直径: 与区域体积相同的球的直径
		double tmp_volume = elem.second.voxel_count * input_voxel_spacing[0] * input_voxel_spacing[1] * input_voxel_spacing[2];
		elem.second.equivalent_diameter = 2.0 * std::pow(0.75 * tmp_volume / M_PI, 1.0 / 3.0);

		//球体占边界框比例
		tmp_volume = elem.second.bbox_max[0] - elem.second.bbox_min[0] + 1.;
		tmp_volume *= elem.second.bbox_max[1] - elem.second.bbox_min[1] + 1.;
		tmp_volume *= elem.second.bbox_max[2] - elem.second.bbox_min[2] + 1.;

		double vol_ratio = elem.second.voxel_count / double(tmp_volume);
		double min_aspect_ratio = 1000.;
		double max_aspect_ratio = 0.;
		for (int j = 0; j < 2; j++)
		{
			min_aspect_ratio = std::min(min_aspect_ratio, double(elem.second.bbox_max[j] - elem.second.bbox_min[j] + 1.) / double(elem.second.bbox_max[j+1] - elem.second.bbox_min[j+1] + 1.));
			min_aspect_ratio = std::min(min_aspect_ratio, double(elem.second.bbox_max[j+1] - elem.second.bbox_min[j+1] + 1.) / double(elem.second.bbox_max[j] - elem.second.bbox_min[j] + 1.));
			max_aspect_ratio = std::max(max_aspect_ratio, double(elem.second.bbox_max[j] - elem.second.bbox_min[j] + 1.) / double(elem.second.bbox_max[j + 1] - elem.second.bbox_min[j + 1] + 1.));
			max_aspect_ratio = std::max(max_aspect_ratio, double(elem.second.bbox_max[j + 1] - elem.second.bbox_min[j + 1] + 1.) / double(elem.second.bbox_max[j] - elem.second.bbox_min[j] + 1.));
		}
		min_aspect_ratio = std::min(min_aspect_ratio, double(elem.second.bbox_max[0] - elem.second.bbox_min[0] + 1.) / double(elem.second.bbox_max[2] - elem.second.bbox_min[2] + 1.));
		min_aspect_ratio = std::min(min_aspect_ratio, double(elem.second.bbox_max[2] - elem.second.bbox_min[2] + 1.) / double(elem.second.bbox_max[0] - elem.second.bbox_min[0] + 1.));
		max_aspect_ratio = std::max(max_aspect_ratio, double(elem.second.bbox_max[0] - elem.second.bbox_min[0] + 1.) / double(elem.second.bbox_max[2] - elem.second.bbox_min[2] + 1.));
		max_aspect_ratio = std::max(max_aspect_ratio, double(elem.second.bbox_max[2] - elem.second.bbox_min[2] + 1.) / double(elem.second.bbox_max[0] - elem.second.bbox_min[0] + 1.));

		/*
		std::cout << "voxel_count:  " << elem.second.voxel_count << endl;
		std::cout << "equivalent_diameter:  " << elem.second.equivalent_diameter << endl;
		std::cout << "vol_ratio:  " << vol_ratio << endl;
		std::cout << "min_aspect_ratio:  " << min_aspect_ratio << endl;
		std::cout << "max_aspect_ratio:  " << max_aspect_ratio << endl;
		*/

		//判断是否为小球
		if (elem.second.equivalent_diameter < 1.25 * diameter && elem.second.equivalent_diameter > 0.5 * diameter
			&& vol_ratio > 0.25 && vol_ratio < 0.75
			&& min_aspect_ratio > 0.7 && max_aspect_ratio < 1.42)
		{
			removed_ball_num++;
			bw_tensor.logical_or_(cc_labels == elem.second.region_id);
			// 保存当前小球的质心
			markerPosition.push_back(elem.second.centroid[0]);
			markerPosition.push_back(elem.second.centroid[1]);
			markerPosition.push_back(elem.second.centroid[2]);
			//std::cout << "removed! " << endl;
		}
		
	}
	if (NETDEBUG_FLAG) {
		std::cout << "removed_ball_num:  " << removed_ball_num << endl;
	}

	//将小球区域往外扩展1个体素，防止可能的粘连
	/*
	bw_tensor = bw_tensor.unsqueeze(0).unsqueeze(0);
	bw_tensor = bw_tensor.to(torch::kInt8);
	bw_tensor = torch::nn::functional::max_pool3d(bw_tensor, torch::nn::functional::MaxPool3dFuncOptions({ 3, 3, 3 }).stride(1).padding(1).dilation(1));
	bw_tensor = bw_tensor.squeeze();
	*/

	//将小球区域的label重置为0
	seg_label_tensor.masked_fill_(bw_tensor > 0, 0);
}

AI_INT  DentalUnet::postProcessing()
{
	/*
	int maxilla_z0 = 0;
	int mandible_z0 = 0;

	at::Tensor bw_tensor = seg_label_tensor == unetConfig.maxilla_label;
	at::Tensor profile = bw_tensor.sum(2,false).sum(1, false);
	int peak_z = profile.argmax(0).item().toInt();
	//std::cout << "peak_z: " << peak_z << endl;
	for (int z = peak_z; z >= 0; z--) {
		if (profile[z].item() == 0) {
			maxilla_z0 = z;
			break;
		}
	}
	//std::cout << "maxilla_z0: " << maxilla_z0 << endl;
	if (maxilla_z0 > 0)
		seg_label_tensor.slice(0, 0, maxilla_z0).masked_fill_(bw_tensor.slice(0, 0, maxilla_z0), 0);

	bw_tensor = seg_label_tensor == unetConfig.mandible_label;
	profile = bw_tensor.sum(2, false).sum(1, false);
	int peak_z1 = profile.argmax(0).item().toInt();
	//std::cout << "peak_z1: " << peak_z1 << endl;
	for (int z = peak_z1; z >= 0; z--) {
		if (profile[z].item() == 0) {
			mandible_z0 = z;
			break;
		}
	}
	//std::cout << "mandible_z0: " << mandible_z0 << endl;
	if (mandible_z0 > 0)
		seg_label_tensor.slice(0, 0, mandible_z0).fill_(0);
	*/

	int min_obj_sz = (int)(unetConfig.marker_diameter / input_voxel_spacing[0] * unetConfig.marker_diameter / input_voxel_spacing[1] * unetConfig.marker_diameter / input_voxel_spacing[2] + 0.5f);
	//std::cout << "min_obj_sz:" << min_obj_sz << endl;

	//校正上下牙标签错误
	// 确保张量在CPU上且数据内存是连续的
	seg_label_tensor = seg_label_tensor.contiguous();
	auto sizes = seg_label_tensor.sizes();
	const int32_t depth = sizes[0], height = sizes[1], width = sizes[2];

	// 获取数据指针
	short* input_data_ptr  = seg_label_tensor.data_ptr<short>();

	if (NETDEBUG_FLAG) {
		std::cout << "cc3d processing..." << endl;
	}
	
	// 调用CC3D函数
	uint32_t* cc_labels_ptr = cc3d::connected_components3d<short>(
		input_data_ptr, width, height, depth, 26);

	if (NETDEBUG_FLAG) {
		std::cout << "cc3d is finished." << endl;
	}
	
	// 输出张量
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor cc_labels = torch::zeros({ depth, height, width }, options);
	cc_labels = torch::from_blob(cc_labels_ptr, { depth, height, width }, options);

	// 连通区域统计结构体
	struct RegionStats {
		int region_id;        // 原始类别标号
		short class_label;    // 组织结构类别
		size_t voxel_count;   // 体素数量
		std::unordered_set<int32_t> adjacent_region_ids; // 邻接区域ID集合
	};

	auto accessor = cc_labels.accessor<int32_t, 3>();
	auto class_label_accessor = seg_label_tensor.accessor<short, 3>();

	if (NETDEBUG_FLAG) {
		std::cout << "统计区域信息..." << endl;
	}
	
	std::unordered_map<int, RegionStats> region_data;
	// 第一遍遍历：初始化区域信息
	for (int32_t z = 0; z < depth; ++z) {
		for (int32_t y = 0; y < height; ++y) {
			for (int32_t x = 0; x < width; ++x) {
				int32_t region_id = accessor[z][y][x];
				if (region_id == 0) continue; // 跳过背景

				if (region_data.count(region_id) == 0) {
					region_data[region_id] = { region_id, class_label_accessor[z][y][x], 0, {} };
				}
				region_data[region_id].voxel_count++;
			}
		}
	}
	if (NETDEBUG_FLAG) {
		std::cout << "统计区域信息完成." << endl;
	}

	size_t maxilla_max_voxel_count = 0;
	int kept_maxilla_region_id = 0;
	size_t mandible_max_voxel_count = 0;
	int kept_mandible_region_id = 0;
	//遍历region_data，确定需保留的上颚和下颌区域标号
	if (NETDEBUG_FLAG) {
		std::cout << "遍历region_data..." << endl;
	}
	for (auto elem : region_data) {
		if (elem.second.class_label == unetConfig.maxilla_label) {
			if (elem.second.voxel_count > maxilla_max_voxel_count) {
				maxilla_max_voxel_count = elem.second.voxel_count;
				kept_maxilla_region_id = elem.second.region_id;
			}
		}
		if (elem.second.class_label == unetConfig.mandible_label) {
			if (elem.second.voxel_count > mandible_max_voxel_count) {
				mandible_max_voxel_count = elem.second.voxel_count;
				kept_mandible_region_id = elem.second.region_id;
			}
		}
	}
	
	options = torch::TensorOptions().dtype(torch::kBool);
	torch::Tensor bw_tensor = torch::zeros({ depth, height, width }, options);

	for (auto elem : region_data) {
		if (elem.second.class_label == unetConfig.maxilla_label && elem.second.region_id != kept_maxilla_region_id) {
			bw_tensor.logical_or_(cc_labels == elem.second.region_id);
			elem.second.region_id = 0;
			elem.second.class_label = 0;
			elem.second.voxel_count = 0;
		}
		if (elem.second.class_label == unetConfig.mandible_label && elem.second.region_id != kept_mandible_region_id) {
			bw_tensor.logical_or_(cc_labels == elem.second.region_id);
			elem.second.region_id = 0;
			elem.second.class_label = 0;
			elem.second.voxel_count = 0;
		}
	}
	seg_label_tensor.masked_fill_(bw_tensor, 0);
	cc_labels.masked_fill_(bw_tensor, 0);

	if (NETDEBUG_FLAG) {
		std::cout << "统计邻接区域..." << endl;
	}
	// 检测邻接关系（26连通）
	for (int32_t z = 0; z < depth; ++z) {
		for (int32_t y = 0; y < height; ++y) {
			for (int32_t x = 0; x < width; ++x) {
				int32_t current_region = accessor[z][y][x];
				if (current_region == 0) continue;

				// 检查26邻域
				for (int dz = -1; dz <= 1; ++dz) {
					for (int dy = -1; dy <= 1; ++dy) {
						for (int dx = -1; dx <= 1; ++dx) {
							if (dz == 0 && dy == 0 && dx == 0) continue;

							int32_t nz = z + dz;
							int32_t ny = y + dy;
							int32_t nx = x + dx;

							// 边界检查
							if (nz >= 0 && nz < depth &&
								ny >= 0 && ny < height &&
								nx >= 0 && nx < width) {
								int32_t neighbor_region = accessor[nz][ny][nx];
								if (neighbor_region != 0 && neighbor_region != current_region) {
									region_data[current_region].adjacent_region_ids.insert(neighbor_region);
								}
							}
						}
					}
				}
			}
		}
	}
	if (NETDEBUG_FLAG) {
		std::cout << "统计邻接区域完成." << endl;
	}
	
	bw_tensor.fill_(0);
	if (NETDEBUG_FLAG) {
		std::cout << "校正上下牙类别标号..." << endl;
	}
	for (auto elem : region_data) {
		//校正预测的上牙标签
		//与下颌邻接的牙齿都是下牙
		if (elem.second.class_label == unetConfig.uppertooth_label) {
			const auto& adjacent_regions = elem.second.adjacent_region_ids;
			int adj_id = 0;
			//如果与下颌邻接，改变区域类别标号为下牙
			if (adjacent_regions.find(kept_mandible_region_id) != adjacent_regions.end()) {
				bw_tensor = cc_labels == elem.second.region_id;
				seg_label_tensor.masked_fill_(bw_tensor, unetConfig.lowertooth_label);
				elem.second.class_label = unetConfig.lowertooth_label;
			}

			//仅与下牙邻接而不与上颚或上牙邻接的都是下牙
			bool is_only_adj_lowertooth = true;
			for (int adj_region_id : adjacent_regions) {
				if (region_data[adj_region_id].class_label != unetConfig.lowertooth_label)
					is_only_adj_lowertooth = false;
				if (region_data[adj_region_id].class_label == unetConfig.lowertooth_label)
					adj_id = adj_region_id;
			}
			if (is_only_adj_lowertooth) {
				bw_tensor = cc_labels == elem.second.region_id;
				seg_label_tensor.masked_fill_(bw_tensor, unetConfig.lowertooth_label);
				elem.second.class_label = unetConfig.lowertooth_label;
				elem.second.voxel_count += region_data[adj_id].voxel_count;
				region_data[adj_id].voxel_count = elem.second.voxel_count;
			}
		}

		if (elem.second.class_label == unetConfig.lowertooth_label) {
			const auto& adjacent_regions = elem.second.adjacent_region_ids;
			int adj_id = 0;
			//如果与上颚邻接，改变区域类别标号为上牙
			if (adjacent_regions.find(kept_maxilla_region_id) != adjacent_regions.end()) {
				bw_tensor = cc_labels == elem.second.region_id;
				seg_label_tensor.masked_fill_(bw_tensor, unetConfig.uppertooth_label);
				elem.second.class_label = unetConfig.uppertooth_label;
			}

			//仅与上牙邻接且不与下颌或下牙邻接的都是上牙
			bool is_only_adj_uppertooth = true;
			for (int adj_region_id : adjacent_regions) {
				if (region_data[adj_region_id].class_label != unetConfig.uppertooth_label)
					is_only_adj_uppertooth = false;
				if (region_data[adj_region_id].class_label == unetConfig.uppertooth_label)
					adj_id = adj_region_id;
			}
			if (is_only_adj_uppertooth) {
				bw_tensor = cc_labels == elem.second.region_id;
				seg_label_tensor.masked_fill_(bw_tensor, unetConfig.uppertooth_label);
				elem.second.class_label = unetConfig.uppertooth_label;
				elem.second.voxel_count += region_data[adj_id].voxel_count;
				region_data[adj_id].voxel_count = elem.second.voxel_count;
			}
		}
	}
	if (NETDEBUG_FLAG) {
		std::cout << "校正上下牙类别标号完成." << endl;
	}
	
	//移除细碎的区域
	bw_tensor.fill_(0);
	for(auto elem : region_data) {
		//std::cout << "voxel_count:" << elem.second.voxel_count << endl;
		if (elem.second.voxel_count <= min_obj_sz) {
			bw_tensor.logical_or_(cc_labels == elem.second.region_id);
			elem.second.region_id = 0;
			elem.second.class_label = 0;
			elem.second.voxel_count = 0;
			//std::cout << "移除小的区域..." << endl;
			continue;
		}
		if (elem.second.class_label == unetConfig.maxilla_label) {
			if (elem.second.region_id != kept_maxilla_region_id) {
				bw_tensor.logical_or_(cc_labels == elem.second.region_id);
				elem.second.region_id = 0;
				elem.second.class_label = 0;
				elem.second.voxel_count = 0;
				continue;
			}
		}
		if (elem.second.class_label == unetConfig.mandible_label) {
			if (elem.second.region_id != kept_mandible_region_id) {
				bw_tensor.logical_or_(cc_labels == elem.second.region_id);
				elem.second.region_id = 0;
				elem.second.class_label = 0;
				elem.second.voxel_count = 0;
			}
		}
	}
	seg_label_tensor.masked_fill_(bw_tensor, 0);

	return DentalCbctSegAI_STATUS_SUCCESS;
}
