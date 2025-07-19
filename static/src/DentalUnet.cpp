#include "DentalUnet.h"

DentalUnet::DentalUnet()
{
	NETDEBUG_FLAG = true;

	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnUNetInference");
	std::vector<std::string> providers = Ort::GetAvailableProviders();
	use_gpu = true;

	for (const auto& provider : providers) {
		std::cout << "????Provider: " << provider << std::endl;
		if (provider == "CUDAExecutionProvider") {
			use_gpu = true;
		}
	}
	//use_gpu = false;


	// ģ��·��Ӧ����������ͨ�� setModelFns ����
	unetConfig.model_file_name = nullptr;  // ��ʼ��Ϊ�գ��ȴ��ⲿ����
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
	std::cout << "[DEBUG] DentalUnet::setModelFns() called" << endl;
	
	if (model_fn == nullptr) {
		std::cerr << "[ERROR] Model filename is NULL!" << endl;
		return;
	}
	
	// Print model path for debugging
	std::wcout << L"[DEBUG] Model path: " << model_fn << endl;
	
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
	//??????????????????????
}


void  DentalUnet::setAlgParameter()
{
	//????????????????
}


AI_INT  DentalUnet::initializeOnnxruntimeInstances()
{
	std::cout << "[DEBUG] Initializing ONNX Runtime instances..." << endl;
	
	if (use_gpu) {
		std::cout << "[DEBUG] GPU mode enabled, configuring CUDA provider..." << endl;
		try {
			//OrtCUDAProviderOptions cuda_options;
			//cuda_options.device_id = 0;  // ??? GPU ?�� ID
			//session_options.AppendExecutionProvider_CUDA(cuda_options);

			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
			std::cout << "[DEBUG] CUDA provider added successfully" << endl;
		} catch (const Ort::Exception& e) {
			std::cerr << "[WARNING] Failed to add CUDA provider: " << e.what() << endl;
			std::cerr << "[WARNING] Falling back to CPU" << endl;
			use_gpu = false;
		}
	} else {
		std::cout << "[DEBUG] Using CPU mode" << endl;
	}
	
	// ?????????
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	std::cout << "[DEBUG] Thread settings: IntraOp=1, InterOp=1" << endl;

	// ??????
	//semantic_seg_session_ptr = std::make_unique<Ort::Session>(env, unetConfig.model_file_name.c_str(), session_options);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::setInput(AI_DataInfo *srcData)
{
	std::cout << "[DEBUG] DentalUnet::setInput() called" << endl;
	
	// Validate input
	if (srcData == nullptr) {
		std::cerr << "[ERROR] srcData is NULL!" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	
	if (srcData->ptr_Data == nullptr) {
		std::cerr << "[ERROR] srcData->ptr_Data is NULL!" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
	
	//check size of input volume
	Width0 = srcData->Width;
	Height0 = srcData->Height;
	Depth0 = srcData->Depth;
	float voxelSpacing = srcData->VoxelSpacing; //??��: mm
	float voxelSpacingX = srcData->VoxelSpacingX; //??��: mm
	float voxelSpacingY = srcData->VoxelSpacingY; //??��: mm
	float voxelSpacingZ = srcData->VoxelSpacingZ; //??��: mm
	
	std::cout << "[DEBUG] Input volume dimensions: " << Width0 << "x" << Height0 << "x" << Depth0 << endl;
	std::cout << "[DEBUG] Voxel spacing: X=" << voxelSpacingX << ", Y=" << voxelSpacingY << ", Z=" << voxelSpacingZ << endl;

	float fovX = float(Width0) * voxelSpacingY;
	float fovY = float(Height0) * voxelSpacingX;
	float fovZ = float(Depth0) * voxelSpacingZ;

	if (Height0 < 64 || Width0 < 64 || Depth0 < 64)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //???????????��

	if (Height0 > 4096 || Width0 > 4096 || Depth0 > 2048)
		return DentalCbctSegAI_STATUS_VOLUME_LARGE; //?????????????

	if (fovX < 30.f || fovY < 30.f || fovZ < 30.f) //volume??��
		return DentalCbctSegAI_STATUS_VOLUME_SMALL;

	if (voxelSpacing < 0.04f || voxelSpacingX < 0.04f || voxelSpacingY < 0.04f || voxelSpacingZ < 0.04f) //voxelSpacing??��
		return DentalCbctSegAI_STATUS_VOLUME_LARGE;

	if (voxelSpacing > 1.1f || voxelSpacingX > 1.1f || voxelSpacingY > 1.1f || voxelSpacingZ > 1.1f)
		return DentalCbctSegAI_STATUS_VOLUME_SMALL; //voxelSpacing???

	// ?????????????CImg????
	//RAI: ?????????????????????????????��?????????????
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

	// ?????????
	bool is_volume_scaled = false;
	////input_voxel_spacing = {voxelSpacingX, voxelSpacingY, voxelSpacingZ }; // x Image width, y Image height, z Image depth 
	std::vector<int64_t> input_size = { input_volume.width(), input_volume.height(), input_volume.depth()};
	std::vector<int64_t> output_size;
	float scaled_factor = 1.f;
	for (int i = 0; i < 3; ++i) {  // ???????????
		scaled_factor = transposed_input_voxel_spacing[i] / config.voxel_spacing[i];
		int scaled_sz = std::round(input_size[i] * scaled_factor);
		if (scaled_factor < 0.9f || scaled_factor > 1.1f || scaled_sz < config.patch_size[i])
			is_volume_scaled = true;

		if (scaled_sz < config.patch_size[i])
			scaled_sz = config.patch_size[i];

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

	//?????????
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
		std::cout << "[DEBUG] Using CT Normalization" << endl;
		CTNormalization(scaled_input_volume, config);
		break;
	case 20:
		std::cout << "[DEBUG] Using Z-Score Normalization" << endl;
		std::cout << "[DEBUG] intensity_mean: " << intensity_mean << ", intensity_std: " << intensity_std << endl;
		scaled_input_volume -= intensity_mean;
		scaled_input_volume /= intensity_std;
		break;
	default:
		std::cout << "[DEBUG] Using default Z-Score Normalization" << endl;
		scaled_input_volume -= intensity_mean;
		scaled_input_volume /= intensity_std;
		break;
	}
	std::cout << "normalized_input_volume mean: " << scaled_input_volume.mean() << endl;
	std::cout << "normalized_input_volume variance: " << scaled_input_volume.variance() << endl;

	//?????????????
	std::cout << "[DEBUG] Calling slidingWindowInfer..." << endl;
	try {
		AI_INT is_ok = slidingWindowInfer(config, scaled_input_volume);
		std::cout << "slidingWindowInfer returned: " << is_ok << endl;
		if (is_ok != DentalCbctSegAI_STATUS_SUCCESS) {
			std::cerr << "[ERROR] slidingWindowInfer failed with code: " << is_ok << endl;
			return is_ok;
		}
	} catch (const std::exception& e) {
		std::cerr << "[ERROR] Exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	} catch (...) {
		std::cerr << "[ERROR] Unknown exception in slidingWindowInfer" << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}

	//???3D???????????
	if (is_volume_scaled)
		predicted_output_prob.resize(input_size[0], input_size[1], input_size[2], config.num_classes, 3);

	output_seg_mask = argmax_spectrum(predicted_output_prob);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


AI_INT  DentalUnet::slidingWindowInfer(nnUNetConfig config, CImg<float> normalized_volume)
{
	if (use_gpu) {
		std::cout << "[DEBUG] Configuring CUDA provider..." << endl;
		try {
			OrtCUDAProviderOptions cuda_options;
			//cuda_options.gpu_mem_limit = 6 * 1024 * 1024 * 1024;  // ?????6GB???[6,12](@ref)
			cuda_options.device_id = 0;
			session_options.AppendExecutionProvider_CUDA(cuda_options);
			std::cout << "[DEBUG] CUDA provider configured successfully" << endl;
		} catch (const Ort::Exception& e) {
			std::cerr << "[WARNING] Failed to configure CUDA provider: " << e.what() << endl;
			std::cerr << "[WARNING] Falling back to CPU" << endl;
		}
	}

	std::cout << "env setting is done" << endl;

	// ??????
	Ort::AllocatorWithDefaultOptions allocator;
	
	// ???????????
	if (config.model_file_name == nullptr) {
		std::cerr << "ERROR: Model file name is NULL!" << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	}
	
	//try-catch?ONNX Runtime?
	try {
		std::cout << "Creating ONNX session..." << endl;
		
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

	// ??��
	if (config.patch_size.size() != 3) {
		throw std::runtime_error("Patch size should be 3D (depth, height, width)");
	}

	// ?? (1, 1, D, H, W)
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

	//?
	predicted_output_prob = CImg<float>(width, height, depth, config.num_classes, 0.f);
	CImg<float> count_vol = CImg<float>(width, height, depth, 1, 0.f);
	//std::cout << "predSegProbVolume shape: " << depth << width << height << endl;

	//CImg<float> input_patch = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.f);
	CImg<float> win_pob = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], config.num_classes, 0.f);
	CImg<float> gaussisan_weight = CImg<float>(config.patch_size[0], config.patch_size[1], config.patch_size[2], 1, 0.f);
	create_3d_gaussian_kernel(gaussisan_weight, config.patch_size);

	size_t input_patch_voxel_numel = config.patch_size[0] * config.patch_size[1] * config.patch_size[2];
	size_t output_patch_vol_sz = config.num_classes * config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * sizeof(float);

	//
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

				std::cout << "[DEBUG] Patch bounds - X: [" << lb_x << ", " << ub_x << "], Y: [" << lb_y << ", " << ub_y << "], Z: [" << lb_z << ", " << ub_z << "]" << endl;
				
				// Validate bounds
				if (ub_x >= width || ub_y >= height || ub_z >= depth) {
					std::cerr << "[ERROR] Patch bounds exceed volume dimensions!" << endl;
					std::cerr << "Volume dimensions: " << width << "x" << height << "x" << depth << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				CImg<float> input_patch;
				try {
					input_patch = normalized_volume.get_crop(lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, 0);
					//std::cout << "input_patch mean: " << input_patch.mean() << endl;
					//std::cout << "input_patch variance: " << input_patch.variance() << endl;
					std::cout << "input_patch dimensions: " << input_patch.width() << "x" << input_patch.height() << "x" << input_patch.depth() << endl;
					if (input_patch.width() != config.patch_size[0] || input_patch.height() != config.patch_size[1] || input_patch.depth() != config.patch_size[2]) {
						std::cerr << "[ERROR] Patch size mismatch! Expected: " << config.patch_size[0] << "x" << config.patch_size[1] << "x" << config.patch_size[2] << endl;
						return DentalCbctSegAI_STATUS_FAIED;
					}
				} catch (const CImgException& e) {
					std::cerr << "[ERROR] CImg exception during cropping: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				//std::vector<float> input_tensor_data;
				//const float* input_patch_ptr = input_patch.data(0, 0, 0, 0);
				//input_tensor_data.insert(input_tensor_data.end(), input_patch_ptr, input_patch_ptr + input_patch_voxel_numel);

				float* input_data_ptr = input_patch.data();

				// 
				Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
					OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
					input_data_ptr,
					input_patch_voxel_numel,
					input_tensor_shape.data(),
					input_tensor_shape.size());

				// 
				std::cout << "[DEBUG] Running ONNX inference for patch #" << patch_count << endl;
				
				// Validate input tensor
				if (input_data_ptr == nullptr) {
					std::cerr << "[ERROR] Input data pointer is null!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// Declare output_tensors outside try block
				std::vector<Ort::Value> output_tensors;
				
				try {
					//session_ptr = std::make_unique<Ort::Session>(env, config.model_file_name, session_options);

					//auto output_tensors = session_ptr->Run(
					std::cout << "[DEBUG] Calling session.Run()..." << endl;
					output_tensors = session.Run(
						Ort::RunOptions{ nullptr },
						&input_name,
						&input_tensor,
						1,
						&output_name,
						1
					);
					std::cout << "[DEBUG] session.Run() completed successfully" << endl;

					if (config.use_mirroring && use_gpu)
					{
						input_patch = input_patch.mirror('x');
					}

					std::cout << "[DEBUG] ONNX inference completed for patch #" << patch_count << endl;
				} catch (const Ort::Exception& e) {
					std::cerr << "[ERROR] ONNX Runtime exception during inference: " << e.what() << endl;
					std::cerr << "[ERROR] Error code: " << e.GetOrtErrorCode() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				} catch (const std::exception& e) {
					std::cerr << "[ERROR] Standard exception during inference: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				} catch (...) {
					std::cerr << "[ERROR] Unknown exception during inference" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// ?
				std::cout << "[DEBUG] Processing output tensor..." << endl;
				
				if (output_tensors.empty()) {
					std::cerr << "[ERROR] No output tensors returned from inference!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}
				
				float* output_data = output_tensors[0].GetTensorMutableData<float>();
				
				if (output_data == nullptr) {
					std::cerr << "[ERROR] Output data pointer is null!" << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}

				// ??CImg
				std::cout << "[DEBUG] Copying output data to win_pob..." << endl;
				std::memcpy(win_pob.data(), output_data, output_patch_vol_sz);
				output_tensors.clear();
				//input_tensor.release();

				std::cout << "[DEBUG] Output patch statistics:" << endl;
				std::cout << "  - Min: " << win_pob.min() << endl;
				std::cout << "  - Max: " << win_pob.max() << endl;
				std::cout << "  - Mean: " << win_pob.mean() << endl;
				std::cout << "  - Dimensions: " << win_pob.width() << "x" << win_pob.height() << "x" << win_pob.depth() << "x" << win_pob.spectrum() << endl;

				std::cout << "[DEBUG] Accumulating patch results..." << endl;
				try {
					cimg_forXYZC(win_pob, x, y, z, c) {
						int gx = lb_x + x;
						int gy = lb_y + y;
						int gz = lb_z + z;
						
						// Validate bounds before writing
						if (gx < 0 || gx >= width || gy < 0 || gy >= height || gz < 0 || gz >= depth) {
							std::cerr << "[ERROR] Out of bounds write attempt: (" << gx << ", " << gy << ", " << gz << ")" << endl;
							return DentalCbctSegAI_STATUS_FAIED;
						}
						
						predicted_output_prob(gx, gy, gz, c) += (win_pob(x, y, z, c) * gaussisan_weight(x, y, z));
					}
					cimg_forXYZ(gaussisan_weight, x, y, z) {
						count_vol(lb_x + x, lb_y + y, lb_z + z) += gaussisan_weight(x, y, z);
					}
				} catch (const std::exception& e) {
					std::cerr << "[ERROR] Exception during patch accumulation: " << e.what() << endl;
					return DentalCbctSegAI_STATUS_FAIED;
				}
			}
		}
	}

	//?
	cimg_forXYZC(predicted_output_prob, x, y, z, c) {
		predicted_output_prob(x, y, z, c) /= count_vol(x, y, z);
	}
	std::cout << "Sliding window inference is done." << endl;

	return DentalCbctSegAI_STATUS_SUCCESS;
	} catch (const Ort::Exception& e) {
		std::cerr << "ONNX Runtime exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_LOADING_FAIED;
	} catch (const std::exception& e) {
		std::cerr << "Standard exception in slidingWindowInfer: " << e.what() << endl;
		return DentalCbctSegAI_STATUS_FAIED;
	}
}


void DentalUnet::CTNormalization(CImg<float>& input_volume, nnUNetConfig config)
{
	//HU????
	float min_hu4dentalCTNormalization = config.min_max_HU[0];
	float max_hu4dentalCTNormalization = config.min_max_HU[1];
	input_volume.cut(min_hu4dentalCTNormalization, max_hu4dentalCTNormalization);

	//????z-score
	float mean_hu4dentalCTNormalization = config.mean_std_HU[0];
	float std_hu4dentalCTNormalization = config.mean_std_HU[1];
	input_volume -= mean_hu4dentalCTNormalization;
	input_volume /= std_hu4dentalCTNormalization;
}


void DentalUnet::create_3d_gaussian_kernel(CImg<float>& gaussisan_weight, const std::vector<int64_t>& patch_sizes)
{
	std::vector<float> sigmas(3);
	for (int i = 0; i < 3; ++i)
		sigmas[i] = (patch_sizes[i] - 1) / 5.0f; // ??W=5??+1???

	int64_t depth  = patch_sizes[0];
	int64_t height = patch_sizes[1]; 
	int64_t width  = patch_sizes[2];

	// ???????????????
	float z_center = (depth - 1)  / 2.0f;
	float y_center = (height - 1) / 2.0f;
	float x_center = (width - 1)  / 2.0f;

	// ??????????????????????
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

	// ???????????????????spectrum???
	CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);

	// ??????????��?? (x,y,z)
	cimg_forXYZ(input, x, y, z) {
		short max_idx = 0;
		float max_val = input(x, y, z, 0);

		// ????spectrum???
		for (short s = 1; s < input.spectrum(); ++s) {
			const float current_val = input(x, y, z, s);
			if (current_val > max_val) {
				max_val = current_val;
				max_idx = s;
			}
		}
		result(x, y, z) = max_idx; // ?��???????
	}
	return result;
}


AI_INT  DentalUnet::getSegMask(AI_DataInfo *dstData)
{
	long volSize = Width0 * Height0 * Depth0 * sizeof(short);
	std::memcpy(dstData->ptr_Data, output_seg_mask.data(), volSize);

	return DentalCbctSegAI_STATUS_SUCCESS;
}


