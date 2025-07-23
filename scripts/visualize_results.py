#!/usr/bin/env python3
"""
Visualize and compare intermediate results from C++ segmentation pipeline
可视化和比较C++分割管线的中间结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(file_path):
    """加载npy或raw文件"""
    path = Path(file_path)
    
    if path.suffix == '.npy':
        return np.load(file_path)
    
    elif path.suffix == '.raw':
        # 尝试加载元数据
        meta_file = path.parent / (path.stem + '_meta.txt')
        if meta_file.exists():
            metadata = {}
            with open(meta_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            
            dtype = metadata.get('dtype', 'float32')
            shape_str = metadata.get('shape', '(1,)')
            shape = eval(shape_str)
            
            data = np.fromfile(file_path, dtype=dtype)
            return data.reshape(shape)
    
    raise ValueError(f"Unsupported file format: {path.suffix}")


def plot_slice_comparison(volumes, titles, slice_idx=None, axis=0):
    """比较多个体积数据的切片"""
    n_volumes = len(volumes)
    fig, axes = plt.subplots(1, n_volumes, figsize=(5*n_volumes, 5))
    
    if n_volumes == 1:
        axes = [axes]
    
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        # 选择切片
        if slice_idx is None:
            slice_idx = vol.shape[axis] // 2
        
        # 获取切片
        if axis == 0:
            slice_data = vol[slice_idx, :, :]
        elif axis == 1:
            slice_data = vol[:, slice_idx, :]
        else:
            slice_data = vol[:, :, slice_idx]
        
        # 显示
        im = axes[i].imshow(slice_data, cmap='gray')
        axes[i].set_title(f'{title}\nSlice {slice_idx} (axis={axis})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_histogram_comparison(volumes, titles):
    """比较多个体积数据的直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for vol, title in zip(volumes, titles):
        # 展平数据并计算直方图
        data_flat = vol.flatten()
        ax.hist(data_flat, bins=100, alpha=0.5, label=title, density=True)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Data Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def analyze_volume(volume, name):
    """分析体积数据的统计信息"""
    print(f"\n=== {name} ===")
    print(f"Shape: {volume.shape}")
    print(f"Dtype: {volume.dtype}")
    print(f"Range: [{volume.min():.4f}, {volume.max():.4f}]")
    print(f"Mean: {volume.mean():.4f}")
    print(f"Std: {volume.std():.4f}")
    
    # 对于多通道数据
    if len(volume.shape) == 4:
        print(f"Channels: {volume.shape[0]}")
        for c in range(volume.shape[0]):
            print(f"  Channel {c}: mean={volume[c].mean():.4f}, std={volume[c].std():.4f}")


def compare_results(result_dir):
    """比较三个阶段的结果"""
    result_path = Path(result_dir)
    
    # 定义要查找的文件
    stages = [
        ('preprocess/preprocessed_normalized_volume', 'Preprocessed'),
        ('model_output/model_output_probability', 'Model Output'),
        ('postprocess/postprocessed_segmentation_mask', 'Postprocessed')
    ]
    
    volumes = []
    titles = []
    
    for file_pattern, title in stages:
        # 尝试npy格式
        npy_path = result_path / f"{file_pattern}.npy"
        raw_path = result_path / f"{file_pattern}.raw"
        
        if npy_path.exists():
            vol = load_data(npy_path)
        elif raw_path.exists():
            vol = load_data(raw_path)
        else:
            print(f"Warning: {file_pattern} not found")
            continue
        
        volumes.append(vol)
        titles.append(title)
        analyze_volume(vol, title)
    
    if len(volumes) >= 2:
        # 绘制切片比较
        fig1 = plot_slice_comparison(volumes[:2], titles[:2])  # 只比较前两个（float类型）
        plt.savefig(result_path / 'slice_comparison.png', dpi=150, bbox_inches='tight')
        
        # 绘制直方图比较
        fig2 = plot_histogram_comparison(volumes[:2], titles[:2])
        plt.savefig(result_path / 'histogram_comparison.png', dpi=150, bbox_inches='tight')
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('input', help='Input file or result directory')
    parser.add_argument('--slice', type=int, help='Slice index to visualize')
    parser.add_argument('--axis', type=int, default=0, choices=[0, 1, 2],
                       help='Axis for slicing (0=Z, 1=Y, 2=X)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 可视化单个文件
        data = load_data(input_path)
        analyze_volume(data, input_path.stem)
        
        # 显示切片
        fig = plot_slice_comparison([data], [input_path.stem], 
                                  slice_idx=args.slice, axis=args.axis)
        plt.show()
    
    elif input_path.is_dir():
        # 比较整个结果目录
        compare_results(input_path)
    
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()