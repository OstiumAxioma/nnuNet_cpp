#ifndef _UNET_IO_H_
#define _UNET_IO_H_
#pragma once

#include <string>
#include "CImg.h"

// Forward declarations
class UnetMain;

class UnetIO {
public:
    // 保存预处理数据
    static void savePreprocessedData(UnetMain* parent, const cimg_library::CImg<float>& data, const std::wstring& path, const std::wstring& filename);
    
    // 保存模型输出
    static void saveModelOutput(const cimg_library::CImg<float>& data, const std::wstring& path, const std::wstring& filename);
    
    // 保存后处理数据
    static void savePostprocessedData(UnetMain* parent, const cimg_library::CImg<short>& data, const std::wstring& path, const std::wstring& filename);
    
    // 保存单个tile
    static void saveTile(const cimg_library::CImg<float>& tile, int tileIndex, int x, int y, int z, const std::wstring& path);
    
private:
    // 辅助函数：创建目录
    static void ensureDirectoryExists(const std::wstring& path);
};

#endif // _UNET_IO_H_