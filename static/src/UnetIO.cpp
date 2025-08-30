#include "UnetIO.h"
#include <iostream>
#include <fstream>
#include <direct.h>
#include <windows.h>

using namespace std;
using namespace cimg_library;

void UnetIO::savePreprocessedData(const CImg<float>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
}

void UnetIO::saveModelOutput(const CImg<float>& data, const std::wstring& path, const std::wstring& filename)
{
}

void UnetIO::savePostprocessedData(const CImg<short>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
}

void UnetIO::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z, const std::wstring& path)
{
}

void UnetIO::ensureDirectoryExists(const std::wstring& path)
{
    CreateDirectoryW(path.c_str(), NULL);
}