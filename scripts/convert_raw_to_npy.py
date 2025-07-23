#!/usr/bin/env python3
"""
Convert raw binary files with metadata to numpy .npy format
用于将C++输出的原始二进制文件转换为numpy格式
"""

import numpy as np
import os
import glob
import argparse
from pathlib import Path


def parse_metadata(meta_file):
    """解析元数据文件"""
    metadata = {}
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    
    # 解析dtype
    dtype = metadata.get('dtype', 'float32')
    
    # 解析shape
    shape_str = metadata.get('shape', '(1,)')
    # 移除括号并分割
    shape_str = shape_str.strip('()')
    shape = tuple(int(x.strip()) for x in shape_str.split(','))
    
    return dtype, shape, metadata


def convert_single_file(raw_file, output_dir=None):
    """转换单个raw文件为npy格式"""
    # 构建文件路径
    raw_path = Path(raw_file)
    meta_file = raw_path.parent / (raw_path.stem + '_meta.txt')
    
    if not meta_file.exists():
        print(f"Warning: Metadata file not found for {raw_file}")
        return False
    
    try:
        # 解析元数据
        dtype, shape, metadata = parse_metadata(meta_file)
        
        # 读取原始数据
        data = np.fromfile(raw_file, dtype=dtype)
        
        # 重塑数据
        data = data.reshape(shape)
        
        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir) / (raw_path.stem + '.npy')
        else:
            output_path = raw_path.parent / (raw_path.stem + '.npy')
        
        # 保存为npy格式
        np.save(output_path, data)
        
        print(f"Converted: {raw_file} -> {output_path}")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")
        if 'description' in metadata:
            print(f"  Description: {metadata['description']}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {raw_file}: {str(e)}")
        return False


def convert_directory(input_dir, output_dir=None, recursive=True):
    """转换目录中的所有raw文件"""
    pattern = '**/*.raw' if recursive else '*.raw'
    raw_files = list(Path(input_dir).glob(pattern))
    
    if not raw_files:
        print(f"No .raw files found in {input_dir}")
        return
    
    print(f"Found {len(raw_files)} raw files to convert")
    
    success_count = 0
    for raw_file in raw_files:
        if convert_single_file(str(raw_file), output_dir):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(raw_files)} files converted successfully")


def verify_conversion(npy_file, raw_file, meta_file):
    """验证转换结果"""
    # 解析元数据
    dtype, shape, _ = parse_metadata(meta_file)
    
    # 加载npy文件
    npy_data = np.load(npy_file)
    
    # 加载原始数据
    raw_data = np.fromfile(raw_file, dtype=dtype).reshape(shape)
    
    # 比较
    if np.array_equal(npy_data, raw_data):
        print(f"✓ Verification passed for {npy_file}")
        return True
    else:
        print(f"✗ Verification failed for {npy_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert raw binary files to numpy format')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('-o', '--output', help='Output directory (default: same as input)')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Process directories recursively')
    parser.add_argument('-v', '--verify', action='store_true',
                       help='Verify conversion results')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 转换单个文件
        if input_path.suffix == '.raw':
            convert_single_file(str(input_path), args.output)
            
            # 验证转换
            if args.verify:
                npy_path = input_path.parent / (input_path.stem + '.npy')
                meta_path = input_path.parent / (input_path.stem + '_meta.txt')
                if npy_path.exists() and meta_path.exists():
                    verify_conversion(str(npy_path), str(input_path), str(meta_path))
        else:
            print(f"Error: {input_path} is not a .raw file")
    
    elif input_path.is_dir():
        # 转换整个目录
        convert_directory(str(input_path), args.output, args.recursive)
    
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()