#include "UnetIO.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <direct.h>

using namespace std;
using namespace cimg_library;

void UnetIO::savePreprocessedData(const CImg<float>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
    if (path.empty()) return;
    
    ensureDirectoryExists(path);
    
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
    origin[0] = metadata.origin[0];
    origin[1] = metadata.origin[1];
    origin[2] = metadata.origin[2];
    image->SetOrigin(origin);
    
    FloatImageType::SpacingType spacing;
    spacing[0] = metadata.spacing[0];
    spacing[1] = metadata.spacing[1];
    spacing[2] = metadata.spacing[2];
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
        std::cerr << "Error writing NIfTI file: " << e.what() << std::endl;
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
    
    ensureDirectoryExists(path);
    
    // 使用ITK保存为NIfTI格式
    std::wstring niftiPath = path + L"\\" + filename + L".nii.gz";
    std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
    
    // 定义ITK类型
    using FloatImageType = itk::Image<float, 3>;
    using WriterType = itk::ImageFileWriter<FloatImageType>;
    
    // 对于多通道数据，我们可能需要分别保存每个通道
    // 目前，如果是概率图则保存第一个通道
    CImg<float> dataToSave;
    if (data.spectrum() > 1) {
        dataToSave = data.get_channel(0);
    } else {
        dataToSave = data;
    }
    
    // 创建ITK图像
    FloatImageType::Pointer image = FloatImageType::New();
    
    // 设置图像大小
    FloatImageType::SizeType size;
    size[0] = dataToSave.width();
    size[1] = dataToSave.height();
    size[2] = dataToSave.depth();
    
    FloatImageType::IndexType start;
    start.Fill(0);
    
    FloatImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    
    image->SetRegions(region);
    image->Allocate();
    
    // 注意：模型输出可能没有origin信息，使用默认值
    FloatImageType::PointType origin;
    origin.Fill(0.0);
    image->SetOrigin(origin);
    
    FloatImageType::SpacingType spacing;
    spacing.Fill(1.0);
    image->SetSpacing(spacing);
    
    // 复制数据
    itk::ImageRegionIterator<FloatImageType> it(image, region);
    const float* cimg_data = dataToSave.data();
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
        std::cerr << "Error writing NIfTI file: " << e.what() << std::endl;
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
    metaFile << "shape: (" << data.spectrum() << ", " << data.depth() << ", " << data.height() << ", " << data.width() << ")" << std::endl;
    metaFile << "order: C" << std::endl;
    metaFile << "description: Model output probability volume (channels, depth, height, width)" << std::endl;
    metaFile.close();
}

void UnetIO::savePostprocessedData(const CImg<short>& data, const std::wstring& path, const std::wstring& filename, const ImageMetadata& metadata)
{
    if (path.empty()) return;
    
    ensureDirectoryExists(path);
    
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
    origin[0] = metadata.origin[0];
    origin[1] = metadata.origin[1];
    origin[2] = metadata.origin[2];
    image->SetOrigin(origin);
    
    ShortImageType::SpacingType spacing;
    spacing[0] = metadata.spacing[0];
    spacing[1] = metadata.spacing[1];
    spacing[2] = metadata.spacing[2];
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
        std::cerr << "Error writing NIfTI file: " << e.what() << std::endl;
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
    
    // 创建tiles子目录
    std::wstring tilesPath = path + L"\\tiles";
    ensureDirectoryExists(tilesPath);
    
    std::wstringstream ss;
    ss << L"tile_" << std::setfill(L'0') << std::setw(4) << tileIndex 
       << L"_x" << x << L"_y" << y << L"_z" << z;
    
    // 保存为NIfTI格式（未压缩）
    std::wstring niftiPath = tilesPath + L"\\" + ss.str() + L".nii";
    std::string narrowNiftiPath(niftiPath.begin(), niftiPath.end());
    tile.save(narrowNiftiPath.c_str());
    
    // 保存为二进制格式
    std::wstring rawPath = tilesPath + L"\\" + ss.str() + L".raw";
    std::wstring metaPath = tilesPath + L"\\" + ss.str() + L"_meta.txt";
    
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
    if (!filesystem::exists(path)) {
        filesystem::create_directories(path);
    }
}