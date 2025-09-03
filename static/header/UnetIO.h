#ifndef _UNET_IO_H_
#define _UNET_IO_H_
#pragma once

#include <string>
#include "CImg.h"

// ITK headers for image I/O
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

// Forward declarations
class UnetMain;

struct ImageMetadata {
    double origin[3];
    double spacing[3];
    double direction[9];
};

class UnetIO {
public:
    // 保存预处理数据
    static void savePreprocessedData(const cimg_library::CImg<float>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata);
    
    // 保存模型输出
    static void saveModelOutput(const cimg_library::CImg<float>& data, const std::wstring& path, const std::wstring& filename);
    
    // 保存后处理数据
    static void savePostprocessedData(const cimg_library::CImg<short>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata);
    
    // 保存单个tile
    static void saveTile(const cimg_library::CImg<float>& tile, int tileIndex, int x, int y, int z, const std::wstring& path);
    
private:
    // 辅助函数：创建目录
    static void ensureDirectoryExists(const std::wstring& path);
};

#endif // _UNET_IO_H_