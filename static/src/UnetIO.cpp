#include "UnetIO.h"
#include "UnetMain.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <direct.h>

// ITK headers for image I/O
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

using namespace std;
using namespace cimg_library;

void UnetIO::savePreprocessedData(UnetMain* parent, const CImg<float>& data, const std::wstring& path, const std::wstring& filename)
{
    if (path.empty()) return;
    
    // 使用ITK保存为NIfTI格式以保留origin信息
    std::wstring niftiPath = path + L"\\" + filename + L".nii.gz";
    std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
    
    // 定义ITK类型
    using FloatImageType = itk::Image<float, 3>;
    using WriterType = itk::ImageFileWriter<FloatImageType>;
    
    // 创建ITK图像
    FloatImageType::Pointer image = FloatImageType::New();
    
    // 设置图像大小
    FloatImageType::SizeType size;
    size[0] = data.width();
    size[1] = data.height();
    size[2] = data.depth();
    
    FloatImageType::IndexType start;
    start.Fill(0);
    
    FloatImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    
    image->SetRegions(region);
    image->Allocate();
    
    // 设置元数据
    FloatImageType::PointType origin;
    origin[0] = parent->imageMetadata.origin[0];
    origin[1] = parent->imageMetadata.origin[1];
    origin[2] = parent->imageMetadata.origin[2];
    image->SetOrigin(origin);
    
    FloatImageType::SpacingType spacing;
    spacing[0] = parent->imageMetadata.spacing[0];
    spacing[1] = parent->imageMetadata.spacing[1];
    spacing[2] = parent->imageMetadata.spacing[2];
    image->SetSpacing(spacing);
    
    // 复制数据
    itk::ImageRegionIterator<FloatImageType> it(image, region);
    const float* cimg_data = data.data();
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(*cimg_data++);
    }
    
    // 写入图像
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(narrowNiftiPath);
    writer->SetInput(image);
    
    try {
        writer->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << "Error writing NIfTI file: " << e.GetDescription() << std::endl;
    }
    
    // 保存为二进制格式供numpy使用
    std::wstring rawPath = path + L"\\" + filename + L".raw";
    std::wstring metaPath = path + L"\\" + filename + L"_meta.txt";
    
    std::string narrowRawPath(rawPath.begin(), rawPath.end());
    std::string narrowMetaPath(metaPath.begin(), metaPath.end());
    
    // 保存原始数据
    std::ofstream rawFile(narrowRawPath, std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    rawFile.close();
    
    // 保存Python使用的元数据
    std::ofstream metaFile(narrowMetaPath);
    metaFile << "dtype: float32" << std::endl;
    metaFile << "shape: (" << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
    metaFile << "order: C" << std::endl;
    metaFile << "description: Preprocessed normalized volume" << std::endl;
    metaFile.close();
}

void UnetIO::saveModelOutput(const CImg<float>& data, const std::wstring& path, const std::wstring& filename)
{
    if (path.empty()) return;
    
    // 保存为二进制格式供numpy使用
    std::wstring rawPath = path + L"\\" + filename + L".raw";
    std::wstring metaPath = path + L"\\" + filename + L"_meta.txt";
    
    std::string narrowRawPath(rawPath.begin(), rawPath.end());
    std::string narrowMetaPath(metaPath.begin(), metaPath.end());
    
    // 保存原始数据
    std::ofstream rawFile(narrowRawPath, std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    rawFile.close();
    
    // 保存Python使用的元数据
    std::ofstream metaFile(narrowMetaPath);
    metaFile << "dtype: float32" << std::endl;
    metaFile << "shape: (" << data.spectrum() << ", " << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
    metaFile << "order: C" << std::endl;
    metaFile << "description: Model output probability volume (channels, depth, height, width)" << std::endl;
    metaFile.close();
}

void UnetIO::savePostprocessedData(UnetMain* parent, const CImg<short>& data, const std::wstring& path, const std::wstring& filename)
{
    if (path.empty()) return;
    
    // 使用ITK保存为NIfTI格式以保留origin信息
    std::wstring niftiPath = path + L"\\" + filename + L".nii.gz";
    std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
    
    // 定义ITK类型
    using ShortImageType = itk::Image<short, 3>;
    using WriterType = itk::ImageFileWriter<ShortImageType>;
    
    // 创建ITK图像
    ShortImageType::Pointer image = ShortImageType::New();
    
    // 设置图像大小
    ShortImageType::SizeType size;
    size[0] = data.width();
    size[1] = data.height();
    size[2] = data.depth();
    
    ShortImageType::IndexType start;
    start.Fill(0);
    
    ShortImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    
    image->SetRegions(region);
    image->Allocate();
    
    // 设置元数据
    ShortImageType::PointType origin;
    origin[0] = parent->imageMetadata.origin[0];
    origin[1] = parent->imageMetadata.origin[1];
    origin[2] = parent->imageMetadata.origin[2];
    image->SetOrigin(origin);
    
    ShortImageType::SpacingType spacing;
    spacing[0] = parent->imageMetadata.spacing[0];
    spacing[1] = parent->imageMetadata.spacing[1];
    spacing[2] = parent->imageMetadata.spacing[2];
    image->SetSpacing(spacing);
    
    // 复制数据
    itk::ImageRegionIterator<ShortImageType> it(image, region);
    const short* cimg_data = data.data();
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(*cimg_data++);
    }
    
    // 写入图像
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(narrowNiftiPath);
    writer->SetInput(image);
    
    try {
        writer->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << "Error writing NIfTI file: " << e.GetDescription() << std::endl;
    }
    
    // 保存为二进制格式供numpy使用
    std::wstring rawPath = path + L"\\" + filename + L".raw";
    std::wstring metaPath = path + L"\\" + filename + L"_meta.txt";
    
    std::string narrowRawPath(rawPath.begin(), rawPath.end());
    std::string narrowMetaPath(metaPath.begin(), metaPath.end());
    
    // 保存原始数据
    std::ofstream rawFile(narrowRawPath, std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
    rawFile.close();
    
    // 保存Python使用的元数据
    std::ofstream metaFile(narrowMetaPath);
    metaFile << "dtype: int16" << std::endl;
    metaFile << "shape: (" << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
    metaFile << "order: C" << std::endl;
    metaFile << "description: Postprocessed segmentation mask" << std::endl;
    metaFile.close();
}

void UnetIO::saveTile(const CImg<float>& tile, int tileIndex, int x, int y, int z, const std::wstring& path)
{
    if (path.empty()) return;
    
    // 如果不存在则创建tiles子目录
    ensureDirectoryExists(path + L"\\tiles");
    
    std::wstringstream ss;
    ss << L"tile_" << std::setfill(L'0') << std::setw(4) << tileIndex 
       << L"_x" << x << L"_y" << y << L"_z" << z;
    
    // 保存为二进制格式
    std::wstring rawPath = path + L"\\tiles\\" + ss.str() + L".raw";
    std::wstring metaPath = path + L"\\tiles\\" + ss.str() + L"_meta.txt";
    
    std::string narrowRawPath(rawPath.begin(), rawPath.end());
    std::string narrowMetaPath(metaPath.begin(), metaPath.end());
    
    // 保存原始数据
    std::ofstream rawFile(narrowRawPath, std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(tile.data()), tile.size() * sizeof(float));
    rawFile.close();
    
    // 保存元数据
    std::ofstream metaFile(narrowMetaPath);
    metaFile << "dtype: float32" << std::endl;
    metaFile << "shape: (" << tile.spectrum() << ", " << tile.depth() << ", " << tile.height() << ", " << tile.width() << ")" << std::endl;
    metaFile << "order: C" << std::endl;
    metaFile << "position: (" << x << ", " << y << ", " << z << ")" << std::endl;
    metaFile << "tile_index: " << tileIndex << std::endl;
    metaFile.close();
}

void UnetIO::ensureDirectoryExists(const std::wstring& path)
{
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
}