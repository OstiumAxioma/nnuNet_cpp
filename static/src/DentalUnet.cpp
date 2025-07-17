#include "../header/DentalUnet.h"

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = true;

	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnUNetInference");
	std::vector<std::string> providers = Ort::GetAvailableProviders();
	use_gpu = true;

	for (const auto& provider : providers) {
		std::cout << "����Provider: " << provider << std::endl;
		if (provider == "CUDAExecutionProvider") {
			use_gpu = true;
		}
	}
	//use_gpu = false;


	unetConfig.model_file_name = L".\\models\\kneeseg_test.onnx";
	unetConfig.input_channels = 1;
	unetConfig.num_classes = 3;
	unetConfig.mandible_label = 1;
	unetConfig.maxilla_label = 2;
	unetConfig.sinus_label = 3;
	//unetConfig.ian_label = 4;
	//unetConfig.uppertooth_label = 5;
	//unetConfig.lowertooth_label = 6;
	unetConfig.cimg_transpose_forward  = "xyz";
	unetConfig.cimg_transpose_backward = "xyz";
	unetConfig.transpose_forward  = { 0, 1, 2 };
	unetConfig.transpose_backward = { 0, 1, 2 };
	unetConfig.voxel_spacing = { 0.5810545086860657f, 0.5810545086860657f, 1.0f };
	unetConfig.patch_size = { 160, 160, 96 };
	unetConfig.step_size_ratio = 0.75f;
	unetConfig.normalization_type = "CTNormalization";
	unetConfig.min_max_HU = { -172.01852416992188f,  1824.9935302734375f };
	unetConfig.mean_std_HU = { 274.2257080078125f, 366.05450439453125f };
	unetConfig.use_mirroring = false;
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


void  DentalUnet::setDnnOptions()
{
	//��������չӲ������������
}


void  DentalUnet::setAlgParameter()
{
	//�����������㷨����
}


AI_INT  DentalUnet::initializeOnnxruntimeInstances()
{
	if (use_gpu) {
		//OrtCUDAProviderOptions cuda_options;
		//cuda_options.device_id = 0;  // ָ�� GPU �豸 ID
		//session_options.AppendExecutionProvider_CUDA(cuda_options);

		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
	}
	// �����߳���
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);

	// �����Ự
	//semantic_seg_session_ptr = std::make_unique<Ort::Session>(env, unetConfig.model_file_name.c_str(), session_options);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::setInput(AI_DataInfo *srcData)
{
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //��λ: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //��λ: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //��λ: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //��λ: mm

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //���������ݹ�С

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return DentalCbctSegAI_STATUS_VOLUME_LARGE; //���������ݹ���

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume��С
		return DentalCbctSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing��С
		return DentalCbctSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.1f || voxelSpacingX > 1.1f || voxelSpacingY > 1.1f || voxelSpacingZ > 1.1f)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //voxelSpacing���

	// �����������ݵ�CImg����
	//RAI: ������ǰ���������ں󣻶��������ң��°����ϣ�ͷ������
	input_cbct_volume = CImg<short>(Width0, Height0, Depth0, 1, 0);
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(input_cbct_volume.data(), srcData->ptr_Data, volSize);

	intensity_mean = (float)input_cbct_volume.mean();
	intensity_std = (float)input_cbct_volume.variance();
	intensity_std = std::sqrt(intensity_std);
	if (intensity_std < 0.0001f)
		intensity_std = 0.0001f;

	input_voxel_spacing = { voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth

	std::cout << "input_volume intensity_mean: " << intensity_mean << endl;
	std::cout << "input_volume intensity_std: " << intensity_std << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::performInference(AI_DataInfo *srcData)
{
	int input_status = setInput(srcData);
	std::cout << "input_status: " << input_status << endl;
	if (input_status != DentalCbctSegAI_STATUS_SUCCESS)
		return input_status;

	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_forward);
	transposed_input_voxel_spacing.clear();
	for (int i = 0; i < 3; ++i) {
		transposed_input_voxel_spacing.push_back(input_voxel_spacing[unetConfig.transpose_forward[i]]);
	}

	//apply CNN
	int infer_status = segModelInfer(unetConfig, input_cbct_volume);
	std::cout << "infer_status: " << infer_status << endl;

	output_seg_mask.permute_axes(unetConfig.cimg_transpose_backward);
	input_cbct_volume.permute_axes(unetConfig.cimg_transpose_backward);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::segModelInfer(nnUNetConfig config, CImg<short> input_volume)
{

	if (transposed_input_voxel_spacing.size() != config.voxel_spacing.size()) {
		throw std::runtime_error("Spacing dimensions mismatch");
	}

	// ����Ŀ��ߴ�
	bool is_volume_scaled = false;
	////input_voxel_spacing = {voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth 
	std::vector<int64_t> input_size = { input_volume.width(), input_volume.height(), input_volume.depth()};
	std::vector<int64_t> output_size;
	float scaled_factor = 1.f;
	for (int i = 0; i < 3; ++i) {  // ֻ�����ռ�ά��
		scaled_factor = transposed_input_voxel_spacing[i] / config.voxel_spacing[i];
		int scaled_sz = std::round(input_size[i] * scaled_factor);
		if (scaled_factor < 0.9f || scaled_factor > 1.1f || scaled_sz < config.patch_size[i])
			is_volume_scaled = true;

		if (scaled_sz < config.patch_size[3])
			scaled_sz = config.patch_size[3];

		output_size.push_back(static_cast<int64_t>(scaled_sz));
	}

	CImg<float> scaled_input_volume;
	if (is_volume_scaled)
		scaled_input_volume = input_volume.get_resize(output_size[0], output_size[1], output_size[2], -100, 3);
	else
		scaled_input_volume.assign(input_volume);

	std::cout << "scaled_input_volume depth: " << scaled_input_volume.depth() << endl;
	std::cout << "scaled_input_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "scaled_input_volume variance: " << scaled_input_volume.variance() << endl;

	//��һ������
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
		CTNormalization(scaled_input_volume, config);
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
	std::cout << "normalized_input_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "normalized_input_volume variance: " << scaled_input_volume.variance() << endl;

	//����������Ԥ��
	AI_INT is_ok = slidingWindowInfer(config, scaled_input_volume);
	std::cout << "slidingWindowInfer: " << is_ok << endl;

	//ʹ��3D��ֵ��������
	if (is_volume_scaled)
		predicted_output_prob.resize(input_size[0], input_size[1], input_size[2], config.num_classes, 3);

	output_seg_mask = argmax_spectrum(predicted_output_prob);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume)
{
	if (use_gpu) {
		OrtCUDAProviderOptions cuda_options;
		//cuda_options.gpu_mem_limit = 6 * 1024 * 1024 * 1024;  // ����Ϊ6GB�Դ�[6,12](@ref)
		cuda_options.device_id = 0;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
	}

	std::cout << "env setting is done: " << endl;

	// �����Ự
	Ort::AllocatorWithDefaultOptions allocator;
	//std::unique_ptr<Ort::Session> session_ptr = std::make_unique<Ort::Session>(env, config.model_file_name, session_options);
	// ��ȡ���������Ϣ
	//const char* input_name  = session_ptr->GetInputNameAllocated(0, allocator).get();
	//const char* output_name = session_ptr->GetOutputNameAllocated(0, allocator).get();
	// ��ȡ������״
	//auto input_shape = session_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	Ort::Session session(env, config.model_file_name, session_options);
	
	const char* input_name  = session.GetInputNameAllocated(0, allocator).get();
	const char* output_name = session.GetOutputNameAllocated(0, allocator).get();

	std::cout << "Session loading is done: " << endl;
	std::cout << "input_name: " << input_name << endl;
	std::cout << "output_name: " << output_name << endl;
	auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	if (input_shape.size() != 5) {
		throw std::runtime_error("Expected 5D input (batch, channels, depth, height, width)");
	}

	// ��֤ͼ����С
	if (config.patch_size.size() != 3) {
		throw std::runtime_error("Patch size should be 3D (depth, height, width)");
	}

	// ׼������������״ (1, 1, D, H, W)
	std::vector<int64_t> input_tensor_shape = { 1, 1, config.patch_size[2], config.patch_size[1], config.patch_size[0] };

	int depth = normalized_volume.depth();
	int width = normalized_volume.width();
	int height = normalized_volume.height();

	// x Image width, y Image height, z Image depth
	float step_size_ratio = config.step_size_ratio;
	float actualStepSize[3];
	int X_num_steps = (int)ceil(float(width - config.patch_size[0]) / (config.patch_size[0] * step_size_ratio)) + 1; //X
	if (X_num_steps > 1)
		actualStepSize[0] = float(width - config.patch_size[0]) / (X_num_steps - 1);
	else
		actualStepSize[0] = 999999.f;

	int Y_num_steps = (int)ceil(float(height - config.patch_size[1]) / (config.patch_size[1] * step_size_ratio)) + 1; //Y
	if (Y_num_steps > 1)
		actualStepSize[1] = float(height - config.patch_size[1]) / (Y_num_steps - 1);
	else
		actualStepSize[1] = 999999.f;

	int Z_num_steps = (int)ceil(float(depth - config.patch_size[2]) / (config.patch_size[2] * step_size_ratio)) + 1; //Y
	if (Z_num_steps > 1)
		actualStepSize[2] = float(depth - config.patch_size[2]) / (Z_num_steps - 1);
	else
		actualStepSize[2] = 999999.f;

	if (NETDEBUG_FLAG)
		std::cout << "Number of tiles: " << X_num_steps * Y_num_steps * Z_num_steps << endl;

	// ��ʼ���������
	predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(width, height, depth, 1, 0.f);
	//std::cout << "predSegProbVolume shape: " << depth << width << height << endl;

	//CImg<float> input_patch = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.f);
	CImg<float> win_pob = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], config.num_classes, 0.f);
	CImg<float> gaussisan_weight = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.f);
	create_3d_gaussian_kernel(gaussisan_weight, config.patch_size);

	size_t input_patch_voxel_numel = config.patch_size[0] * config.patch_size[1] * config.patch_size[2];
	size_t output_patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	//����������
	int patch_count = 0;
	for (int sz = 0; sz < Z_num_steps; sz++)
	{
		int lb_z = (int)std::round(sz * actualStepSize[2]);
		int ub_z = lb_z + config.patch_size[2] - 1;

		for (int sy = 0; sy < Y_num_steps; sy++)
		{
			int lb_y = (int)std::round(sy * actualStepSize[1]);
			int ub_y = lb_y + config.patch_size[1] - 1;

			for (int sx = 0; sx < X_num_steps; sx++)
			{
				int lb_x = (int)std::round(sx * actualStepSize[0]);
				int ub_x = lb_x + config.patch_size[0] - 1;

				patch_count += 1;
				if (NETDEBUG_FLAG)
					std::cout << "current tile#: " << patch_count << endl;

				CImg<float> input_patch = normalized_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
				//std::cout << "input_patch mean: " << input_patch.mean() << endl;
				//std::cout << "input_patch variance: " << input_patch.variance() << endl;
				std::cout << "input_patch width: " << input_patch.width() << endl;
				std::cout << "input_patch height: " << input_patch.height() << endl;
				std::cout << "input_patch depth: " << input_patch.depth() << endl;

				//std::vector<float> input_tensor_data;
				//const float* input_patch_ptr = input_patch.data(0, 0, 0, 0);
				//input_tensor_data.insert(input_tensor_data.end(), input_patch_ptr, input_patch_ptr + input_patch_voxel_numel);

				float* input_data_ptr = input_patch.data();

				// ������������
				Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
					OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
					input_data_ptr,
					input_patch_voxel_numel,
					input_tensor_shape.data(),
					input_tensor_shape.size());

				// ��������
				std::cout << "Run onnx session." << endl;

				//session_ptr = std::make_unique<Ort::Session>(env, config.model_file_name, session_options);

				//auto output_tensors = session_ptr->Run(
				auto output_tensors = session.Run(
					Ort::RunOptions{ nullptr },
					&input_name,
					&input_tensor,
					1,
					&output_name,
					1
				);

				if (config.use_mirroring && use_gpu)
				{
					input_patch = input_patch.mirror('x');
				}

				std::cout << "onnx session running is done." << endl;

				// ��ȡ���
				float* output_data = output_tensors[0].GetTensorMutableData<float>();

				// ת��ΪCImg
				std::memcpy(win_pob.data(), output_data, output_patch_vol_sz);
				output_tensors.clear();
				//input_tensor.release();

				std::cout << "output_patch min: " << win_pob.min() << endl;
				std::cout << "output_patch max: " << win_pob.max() << endl;
				std::cout << "output_patch mean: " << win_pob.mean() << endl;

				cimg_forXYZC(win_pob, x, y, z, c) {
					predicted_output_prob(lb_x + x, lb_y + y, lb_z + z, c) += (win_pob(x, y, z, c) * gaussisan_weight(x, y, z));
				}
				cimg_forXYZ(gaussisan_weight, x, y, z) {
					count_vol(lb_x + x, lb_y + y, lb_z + z) += gaussisan_weight(x, y, z);
				}
			}
		}
	}

	// ��һ�����
	cimg_forXYZC(predicted_output_prob, x, y, z, c) {
		predicted_output_prob(x, y, z, c) /= count_vol(x, y, z);
	}
	std::cout << "Sliding window inference is done." << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
}


void  DentalUnet::CTNormalization(CImg<float>& input_volume, nnUNetConfig config)
{
	//HUֵ�ض�
	float min_hu4dentalCTNormalization = config.min_max_HU[0];
	float max_hu4dentalCTNormalization = config.min_max_HU[1];
	input_volume.cut(min_hu4dentalCTNormalization, max_hu4dentalCTNormalization);

	//����z-score
	float mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	float std_hu4dentalCTNormalization = config.mean_std_HU[1];
	input_volume -= mean_hu4dentalCTNormalization;
	input_volume /= std_hu4dentalCTNormalization;
}


void  DentalUnet::create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes)
{
	std::vector<float> sigmas(3);
	for (int i = 0; i < 3; ++i)
		sigmas[i] = (patch_sizes[i] - 1) / 5.0f; // ��W=5��+1�Ƶ�

	int64_t depth  = patch_sizes[0];
	int64_t height = patch_sizes[1]; 
	int64_t width  = patch_sizes[2];

	// ����ÿ��ά�ȵ�����
	float z_center = (depth - 1)  / 2.0f;
	float y_center = (height - 1) / 2.0f;
	float x_center = (width - 1)  / 2.0f;

	// �����׼���������������
	float z_sigma = depth  / 4.0f;
	float y_sigma = height / 4.0f;
	float x_sigma = width  / 4.0f;

	float z_part = 0.f;
	float y_part = 0.f;
	float x_part = 0.f;
	cimg_forXYZ(gaussisan_weight, x, y, z) {
		z_part = std::exp(-0.5f * std::pow((z - z_center) / z_sigma, 2));
		y_part = std::exp(-0.5f * std::pow((y - y_center) / y_sigma, 2));
		x_part = std::exp(-0.5f * std::pow((x - x_center) / x_sigma, 2));
		gaussisan_weight(x, y, z) = z_part * y_part * x_part;
	}

	gaussisan_weight /= gaussisan_weight.mean();
}


CImg<short> DentalUnet::argmax_spectrum(const CImg<float>& input) {
	if (input.is_empty() || input.spectrum() == 0) {
		throw std::invalid_argument("Input must be a non-empty 4D CImg with spectrum dimension.");
	}

	// ��ʼ�����ͼ����ά�ռ䣬��spectrumά��
	CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

	// ����ÿ���ռ�λ�� (x,y,z)
	cimg_forXYZ(input, x, y, z) {
		short max_idx = 0;
		float max_val = input(x, y, z, 0);

		// ����spectrumά��
		for (short s = 1; s < input.spectrum(); ++s) {
			const float current_val = input(x, y, z, s);
			if (current_val > max_val) {
				max_val = current_val;
				max_idx = s;
			}
		}
		result(x, y, z) = max_idx; // �洢�������
	}
	return result;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(dstData->ptr_Data, output_seg_mask.data(), volSize);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


