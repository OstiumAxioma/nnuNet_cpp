#!/usr/bin/env python3
"""
简化版本：对比 C++ 和 Python nnUNet 结果
只需要 numpy
"""

import numpy as np
import os
import struct
import gzip

def read_hdr_img(hdr_path):
    """读取Analyze格式的医学图像"""
    # 读取header文件
    with open(hdr_path, 'rb') as f:
        # 跳过不需要的header信息
        f.seek(40)  # 跳到dimensions部分
        dims = struct.unpack('8h', f.read(16))
        ndims = dims[0]
        nx, ny, nz = dims[1], dims[2], dims[3]
        
        # 读取数据类型
        f.seek(70)
        datatype = struct.unpack('h', f.read(2))[0]
        
        # 读取voxel dimensions
        f.seek(76)
        voxel_dims = struct.unpack('8f', f.read(32))
    
    # 根据数据类型确定dtype
    dtype_map = {
        2: np.uint8,    # unsigned char
        4: np.int16,    # signed short
        8: np.int32,    # signed int
        16: np.float32, # float
        64: np.float64  # double
    }
    dtype = dtype_map.get(datatype, np.float32)
    
    # 读取img文件
    img_path = hdr_path.replace('.hdr', '.img')
    data = np.fromfile(img_path, dtype=dtype)
    
    # reshape数据
    if ndims == 3:
        data = data.reshape((nz, ny, nx))
        data = np.transpose(data, (2, 1, 0))  # 转换为 (x, y, z)
    elif ndims == 4:
        nt = dims[4]
        data = data.reshape((nt, nz, ny, nx))
        data = np.transpose(data, (3, 2, 1, 0))  # 转换为 (x, y, z, t)
    
    return data

def compare_basic_stats(py_data, cpp_data, name):
    """比较基本统计信息"""
    print(f"\n{'='*50}")
    print(f"{name} 对比分析")
    print(f"{'='*50}")
    
    print(f"Python shape: {py_data.shape}")
    print(f"C++ shape: {cpp_data.shape}")
    
    if py_data.shape != cpp_data.shape:
        print("警告: 数据形状不匹配！")
        return
    
    # 基本统计
    print(f"\nPython 统计:")
    print(f"  均值: {np.mean(py_data):.6f}")
    print(f"  标准差: {np.std(py_data):.6f}")
    print(f"  最小值: {np.min(py_data):.6f}")
    print(f"  最大值: {np.max(py_data):.6f}")
    print(f"  非零元素: {np.count_nonzero(py_data)}")
    
    print(f"\nC++ 统计:")
    print(f"  均值: {np.mean(cpp_data):.6f}")
    print(f"  标准差: {np.std(cpp_data):.6f}")
    print(f"  最小值: {np.min(cpp_data):.6f}")
    print(f"  最大值: {np.max(cpp_data):.6f}")
    print(f"  非零元素: {np.count_nonzero(cpp_data)}")
    
    # 差异分析
    diff = cpp_data - py_data
    abs_diff = np.abs(diff)
    
    print(f"\n差异分析:")
    print(f"  绝对差异均值: {np.mean(abs_diff):.6f}")
    print(f"  绝对差异最大值: {np.max(abs_diff):.6f}")
    print(f"  相对差异均值: {np.mean(abs_diff / (np.abs(py_data) + 1e-8)):.6f}")
    
    # 相关性
    correlation = np.corrcoef(py_data.flatten(), cpp_data.flatten())[0, 1]
    print(f"  相关系数: {correlation:.6f}")
    
    # 一致性分析（对于分割mask）
    if np.all(np.isin(py_data, [0, 1, 2, 3])) and np.all(np.isin(cpp_data, [0, 1, 2, 3])):
        agreement = np.mean(py_data == cpp_data)
        print(f"\n分割一致性: {agreement:.4%}")
        
        # 每个类别的分析
        unique_labels = np.unique(np.concatenate([py_data.flatten(), cpp_data.flatten()]))
        for label in unique_labels:
            py_mask = py_data == label
            cpp_mask = cpp_data == label
            
            overlap = np.sum(py_mask & cpp_mask)
            py_count = np.sum(py_mask)
            cpp_count = np.sum(cpp_mask)
            
            dice = 2 * overlap / (py_count + cpp_count) if (py_count + cpp_count) > 0 else 0
            print(f"  类别 {label} - Dice系数: {dice:.4f}, Python体素: {py_count}, C++体素: {cpp_count}")

def analyze_probability_slices(py_prob, cpp_prob):
    """分析概率图的几个切片"""
    if len(py_prob.shape) != 4:
        print("跳过概率切片分析（需要4D数据）")
        return
    
    print(f"\n概率图切片分析（中间切片）:")
    z_mid = py_prob.shape[2] // 2
    
    # 分析每个类别
    for c in range(py_prob.shape[3]):
        py_slice = py_prob[:, :, z_mid, c]
        cpp_slice = cpp_prob[:, :, z_mid, c]
        
        print(f"\n类别 {c}:")
        print(f"  Python - 均值: {np.mean(py_slice):.4f}, 范围: [{np.min(py_slice):.4f}, {np.max(py_slice):.4f}]")
        print(f"  C++    - 均值: {np.mean(cpp_slice):.4f}, 范围: [{np.min(cpp_slice):.4f}, {np.max(cpp_slice):.4f}]")
        
        # 检查概率和是否为1
        if c == 0:
            py_prob_sum = np.sum(py_prob[:, :, z_mid, :], axis=2)
            cpp_prob_sum = np.sum(cpp_prob[:, :, z_mid, :], axis=2)
            print(f"\n概率和检查（应该接近1）:")
            print(f"  Python - 均值: {np.mean(py_prob_sum):.6f}, 范围: [{np.min(py_prob_sum):.6f}, {np.max(py_prob_sum):.6f}]")
            print(f"  C++    - 均值: {np.mean(cpp_prob_sum):.6f}, 范围: [{np.min(cpp_prob_sum):.6f}, {np.max(cpp_prob_sum):.6f}]")

def read_nifti_simple(nifti_path):
    """简单读取NIfTI格式文件（.nii.gz）"""
    # 解压并读取前352字节的header
    with gzip.open(nifti_path, 'rb') as f:
        # 读取关键的header信息
        f.seek(40)  # 跳到dim字段
        dim = struct.unpack('8h', f.read(16))
        nx, ny, nz = dim[1], dim[2], dim[3]
        
        f.seek(70)  # 跳到datatype字段  
        datatype = struct.unpack('h', f.read(2))[0]
        
        f.seek(108) # 跳到vox_offset
        vox_offset = struct.unpack('f', f.read(4))[0]
        
        # 确定数据类型
        dtype_map = {
            2: np.uint8,    # unsigned char
            4: np.int16,    # signed short  
            8: np.int32,    # signed int
            16: np.float32, # float
            64: np.float64, # double
            256: np.int8,   # signed char
            512: np.uint16, # unsigned short
            768: np.uint32  # unsigned int
        }
        dtype = dtype_map.get(datatype, np.float32)
        
        # 读取数据
        f.seek(int(vox_offset))
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((nx, ny, nz))
        
    return data

def main():
    # 路径设置
    compare_dir = r"D:\Project\nnuNet_cpp\compare"
    result_dir = r"D:\Project\nnuNet_cpp\result"
    
    print("nnUNet C++ vs Python 最终结果对比分析")
    print("="*60)
    
    # 只对比最终的分割结果
    print("\n加载最终分割结果...")
    try:
        # 加载Python的最终结果（NIfTI格式）
        py_final_path = os.path.join(compare_dir, "Series_5_Acq_2.nii.gz")
        py_final = read_nifti_simple(py_final_path)
        print(f"Python最终结果形状: {py_final.shape}")
        print(f"Python唯一标签值: {np.unique(py_final)}")
        print(f"Python非零体素数: {np.count_nonzero(py_final)}")
        
        # 加载C++的最终结果
        cpp_final = read_hdr_img(os.path.join(result_dir, "finalLabelMask.hdr"))
        # 如果C++结果是4D的，取第一个通道
        if len(cpp_final.shape) == 4:
            cpp_final = cpp_final[..., 0]
        print(f"\nC++最终结果形状: {cpp_final.shape}")
        print(f"C++唯一标签值: {np.unique(cpp_final)}")
        print(f"C++非零体素数: {np.count_nonzero(cpp_final)}")
        
        # 比较两个结果
        compare_basic_stats(py_final, cpp_final, "最终分割结果")
        
    except Exception as e:
        print(f"加载最终结果失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("说明：")
    print("- 如果两个结果在Slicer中看起来一样，但数值统计不同，")
    print("  可能是因为坐标系或者数据排列顺序的差异")
    print("- 重要的是分割的体素数量和分布是否一致")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()