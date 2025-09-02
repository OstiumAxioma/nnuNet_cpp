#include "UnetIO.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <direct.h>

using namespace std;
using namespace cimg_library;

void UnetIO::savePreprocessedData(const CImg<float>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
    // TODO: 从DentalUnet::savePreprocessedData迁移代码（行1441-1524）
}

void UnetIO::saveModelOutput(const CImg<float>& data, const std::wstring& path, const std::wstring& filename)
{
    // TODO: 从DentalUnet::saveModelOutput迁移代码（行1525-1617）
}

void UnetIO::savePostprocessedData(const CImg<short>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
    // TODO: 从DentalUnet::savePostprocessedData迁移代码（行1618-1701）
}

void UnetIO::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z, const std::wstring& path)
{
    // TODO: 从DentalUnet::saveTile迁移代码（行1702-1741）
}

void UnetIO::ensureDirectoryExists(const std::wstring& path)
{
    if (!filesystem::exists(path)) {
        filesystem::create_directories(path);
    }
}