#include "UnetInference.h"
#include "DentalCbctSegAI_API.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace cimg_library;

AI_INT UnetInference::runSlidingWindow(DentalUnet* parent,
                                      const nnUNetConfig& config,
                                      const CImg<float>& input,
                                      CImg<float>& output,
                                      Ort::Env& env,
                                      Ort::SessionOptions& session_options,
                                      bool use_gpu)
{
    return DentalCbctSegAI_STATUS_SUCCESS;
}

void UnetInference::createGaussianKernel(CImg<float>& kernel, const std::vector<int64_t>& patch_sizes)
{
}

AI_INT UnetInference::inferPatch(Ort::Session& session,
                                const CImg<float>& patch,
                                CImg<float>& output,
                                const std::vector<int64_t>& input_shape,
                                const char* input_name,
                                const char* output_name)
{
    return DentalCbctSegAI_STATUS_SUCCESS;
}