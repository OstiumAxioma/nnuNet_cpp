#include "UnetPostprocessor.h"
#include "DentalCbctSegAI_API.h"
#include <iostream>
#include <cstring>

using namespace std;
using namespace cimg_library;

AI_INT UnetPostprocessor::processSegmentationMask(DentalUnet* parent,
                                                 CImg<float>& prob_volume,
                                                 AI_DataInfo* dstData)
{
    return DentalCbctSegAI_STATUS_SUCCESS;
}

CImg<short> UnetPostprocessor::argmaxSpectrum(const CImg<float>& input)
{
    CImg<short> result(input.width(), input.height(), input.depth(), 1, 0);
    return result;
}

void UnetPostprocessor::restoreOriginalSize(const CImg<short>& input, CImg<short>& output, const CropBBox& bbox, int width0, int height0, int depth0)
{
}

void UnetPostprocessor::revertTranspose(CImg<short>& volume, const char* transpose_backward)
{
}