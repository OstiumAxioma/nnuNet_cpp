#!/usr/bin/env python3
"""
将分割结果转换为npy格式并进行对比
"""

import numpy as np
import os
import struct
import gzip

def read_hdr_img(hdr_path):
    """读取HDR/IMG格式并保存为npy"""
    with open(hdr_path, 'rb') as f:
        f.seek(40)
        dims = struct.unpack('8h', f.read(16))
        ndims = dims[0]
        nx, ny, nz = dims[1], dims[2], dims[3]
        
        f.seek(70)
        datatype = struct.unpack('h', f.read(2))[0]
    
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32}
    dtype = dtype_map.get(datatype, np.float32)
    
    img_path = hdr_path.replace('.hdr', '.img')
    data = np.fromfile(img_path, dtype=dtype)
    
    if ndims == 3:
        data = data.reshape((nz, ny, nx))
        data = np.transpose(data, (2, 1, 0))  # 转换为 (X, Y, Z)
    elif ndims == 4:
        nt = dims[4]
        data = data.reshape((nt, nz, ny, nx))
        data = np.transpose(data, (3, 2, 1, 0))  # 转换为 (X, Y, Z, T)
        data = data[..., 0]  # 取第一个通道
    
    return data

def read_nifti_gz(nifti_path):
    """读取.nii.gz格式并保存为npy"""
    with gzip.open(nifti_path, 'rb') as f:
        # 读取header
        header = f.read(352)
        
        # 解析维度
        dim = struct.unpack('8h', header[40:56])
        nx, ny, nz = dim[1], dim[2], dim[3]
        
        # 数据类型
        datatype = struct.unpack('h', header[70:72])[0]
        dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32}
        dtype = dtype_map.get(datatype, np.float32)
        
        # 读取数据
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((nx, ny, nz))
    
    return data

def calculate_dice(mask1, mask2, label):
    """计算特定标签的Dice系数"""
    m1 = mask1 == label
    m2 = mask2 == label
    
    intersection = np.sum(m1 & m2)
    sum_total = np.sum(m1) + np.sum(m2)
    
    if sum_total == 0:
        return 1.0
    return 2.0 * intersection / sum_total

def compare_masks(py_data, cpp_data):
    """详细对比两个分割mask"""
    print("\n分割结果对比:")
    print("-"*50)
    
    # 基本信息
    print(f"Python形状: {py_data.shape}")
    print(f"C++形状: {cpp_data.shape}")
    
    if py_data.shape != cpp_data.shape:
        print("警告：形状不匹配！无法进行逐像素对比。")
        return
    
    # 获取唯一标签
    py_labels = np.unique(py_data)
    cpp_labels = np.unique(cpp_data)
    all_labels = np.unique(np.concatenate([py_labels, cpp_labels]))
    
    print(f"\nPython标签: {py_labels}")
    print(f"C++标签: {cpp_labels}")
    
    # 整体一致性
    agreement = np.sum(py_data == cpp_data) / py_data.size
    print(f"\n整体一致率: {agreement:.2%}")
    
    # 每个标签的分析
    print("\n各标签Dice系数:")
    dice_scores = {}
    
    for label in all_labels:
        py_count = np.sum(py_data == label)
        cpp_count = np.sum(cpp_data == label)
        dice = calculate_dice(py_data, cpp_data, label)
        dice_scores[label] = dice
        
        print(f"标签 {label}: Dice={dice:.4f}, Python={py_count:,}, C++={cpp_count:,}")
    
    # 前景平均Dice
    fg_dice = [dice_scores[l] for l in all_labels if l > 0]
    if fg_dice:
        avg_dice = np.mean(fg_dice)
        print(f"\n前景平均Dice: {avg_dice:.4f}")

def main():
    result_dir = r"D:\Project\nnuNet_cpp\result"
    compare_dir = r"D:\Project\nnuNet_cpp\compare"
    
    print("转换并对比分割结果")
    print("="*60)
    
    # 1. 转换C++结果
    print("\n1. 读取C++结果 (HDR/IMG)...")
    cpp_path = os.path.join(result_dir, "finalLabelMask.hdr")
    cpp_data = read_hdr_img(cpp_path)
    cpp_npy_path = os.path.join(result_dir, "finalLabelMask_cpp.npy")
    np.save(cpp_npy_path, cpp_data)
    print(f"   保存为: {cpp_npy_path}")
    print(f"   形状: {cpp_data.shape}, 类型: {cpp_data.dtype}")
    
    # 2. 转换Python结果
    print("\n2. 读取Python结果 (NIfTI)...")
    py_path = os.path.join(compare_dir, "Series_5_Acq_2.nii.gz")
    py_data = read_nifti_gz(py_path)
    py_npy_path = os.path.join(compare_dir, "Series_5_Acq_2_python.npy")
    np.save(py_npy_path, py_data)
    print(f"   保存为: {py_npy_path}")
    print(f"   形状: {py_data.shape}, 类型: {py_data.dtype}")
    
    # 3. 对比
    compare_masks(py_data, cpp_data)
    
    # 4. 保存差异图
    if py_data.shape == cpp_data.shape:
        diff = (py_data != cpp_data).astype(np.uint8) * 255
        diff_path = os.path.join(result_dir, "difference_map.npy")
        np.save(diff_path, diff)
        print(f"\n差异图保存为: {diff_path}")
        print("  (255=不同, 0=相同)")
    
    print("\n完成！")
    
    # 5. 可视化分析
    print("\n" + "="*60)
    print("进行可视化分析...")
    visualize_comparison(py_data, cpp_data)

def visualize_comparison(py_data, cpp_data):
    """可视化对比分析"""
    # 选择中间切片
    z_mid = py_data.shape[2] // 2
    
    print(f"\n查看第 {z_mid} 层切片:")
    
    # Python切片
    py_slice = py_data[:, :, z_mid]
    print(f"Python切片 - 唯一值: {np.unique(py_slice)}")
    print(f"  标签1像素数: {np.sum(py_slice == 1)}")
    print(f"  标签2像素数: {np.sum(py_slice == 2)}")
    
    # C++切片
    cpp_slice = cpp_data[:, :, z_mid]
    print(f"C++切片 - 唯一值: {np.unique(cpp_slice)}")
    print(f"  标签1像素数: {np.sum(cpp_slice == 1)}")
    print(f"  标签2像素数: {np.sum(cpp_slice == 2)}")
    
    # 差异统计
    diff_slice = py_slice != cpp_slice
    diff_count = np.sum(diff_slice)
    total_pixels = py_slice.size
    print(f"\n切片差异: {diff_count}/{total_pixels} ({diff_count/total_pixels*100:.1f}%)")
    
    # 试试简单的坐标变换
    print("\n尝试简单的坐标变换...")
    
    # 1. 尝试转置
    cpp_transposed = np.transpose(cpp_data, (1, 0, 2))
    dice_transposed = calculate_dice(py_data, cpp_transposed, 1)
    print(f"转置后Dice(标签1): {dice_transposed:.4f}")
    
    # 2. 尝试翻转
    cpp_flipped_x = np.flip(cpp_data, axis=0)
    dice_flipped_x = calculate_dice(py_data, cpp_flipped_x, 1)
    print(f"X轴翻转后Dice(标签1): {dice_flipped_x:.4f}")
    
    cpp_flipped_y = np.flip(cpp_data, axis=1)
    dice_flipped_y = calculate_dice(py_data, cpp_flipped_y, 1)
    print(f"Y轴翻转后Dice(标签1): {dice_flipped_y:.4f}")
    
    cpp_flipped_z = np.flip(cpp_data, axis=2)
    dice_flipped_z = calculate_dice(py_data, cpp_flipped_z, 1)
    print(f"Z轴翻转后Dice(标签1): {dice_flipped_z:.4f}")
    
    # 3. 组合变换
    cpp_rot = np.rot90(cpp_data, k=1, axes=(0, 1))
    dice_rot = calculate_dice(py_data, cpp_rot, 1)
    print(f"XY平面旋转90度后Dice(标签1): {dice_rot:.4f}")

if __name__ == "__main__":
    main()