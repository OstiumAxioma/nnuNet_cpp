#!/usr/bin/env python3
"""
Compare results between C++ implementation and Python implementation
比较C++实现和Python实现的结果
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


def compute_metrics(data1, data2):
    """计算两个数据之间的差异指标"""
    # 确保形状相同
    if data1.shape != data2.shape:
        raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}")
    
    # 计算各种指标
    metrics = {}
    
    # 基本统计
    metrics['shape'] = data1.shape
    metrics['dtype1'] = str(data1.dtype)
    metrics['dtype2'] = str(data2.dtype)
    
    # 转换为相同类型进行比较
    if data1.dtype != data2.dtype:
        data1 = data1.astype(np.float64)
        data2 = data2.astype(np.float64)
    
    # 差异指标
    diff = data1 - data2
    abs_diff = np.abs(diff)
    
    metrics['max_abs_diff'] = np.max(abs_diff)
    metrics['mean_abs_diff'] = np.mean(abs_diff)
    metrics['std_diff'] = np.std(diff)
    metrics['rmse'] = np.sqrt(np.mean(diff**2))
    
    # 相对误差（避免除零）
    mask = data2 != 0
    if np.any(mask):
        rel_diff = abs_diff[mask] / np.abs(data2[mask])
        metrics['max_rel_diff'] = np.max(rel_diff)
        metrics['mean_rel_diff'] = np.mean(rel_diff)
    else:
        metrics['max_rel_diff'] = 0
        metrics['mean_rel_diff'] = 0
    
    # 相关性
    metrics['correlation'] = np.corrcoef(data1.flatten(), data2.flatten())[0, 1]
    
    # 完全相等的元素比例
    metrics['exact_match_ratio'] = np.sum(data1 == data2) / data1.size
    
    # 在容差范围内相等的元素比例
    metrics['close_match_ratio_1e-6'] = np.sum(np.isclose(data1, data2, rtol=1e-6)) / data1.size
    metrics['close_match_ratio_1e-3'] = np.sum(np.isclose(data1, data2, rtol=1e-3)) / data1.size
    
    return metrics, diff


def plot_comparison(data1, data2, diff, name1="Data 1", name2="Data 2", slice_idx=None):
    """绘制比较图"""
    # 选择要显示的切片
    if len(data1.shape) == 3:
        if slice_idx is None:
            slice_idx = data1.shape[0] // 2
        slice1 = data1[slice_idx, :, :]
        slice2 = data2[slice_idx, :, :]
        slice_diff = diff[slice_idx, :, :]
    elif len(data1.shape) == 4:
        # 对于4D数据，选择第一个通道
        if slice_idx is None:
            slice_idx = data1.shape[1] // 2
        slice1 = data1[0, slice_idx, :, :]
        slice2 = data2[0, slice_idx, :, :]
        slice_diff = diff[0, slice_idx, :, :]
    else:
        raise ValueError(f"Unsupported data shape: {data1.shape}")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：原始数据和差异
    im1 = axes[0, 0].imshow(slice1, cmap='gray')
    axes[0, 0].set_title(f'{name1}\nSlice {slice_idx}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(slice2, cmap='gray')
    axes[0, 1].set_title(f'{name2}\nSlice {slice_idx}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(slice_diff, cmap='RdBu_r', 
                           vmin=-np.abs(slice_diff).max(), 
                           vmax=np.abs(slice_diff).max())
    axes[0, 2].set_title(f'Difference\n({name1} - {name2})')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：直方图和散点图
    # 值分布直方图
    axes[1, 0].hist(data1.flatten(), bins=100, alpha=0.5, label=name1, density=True)
    axes[1, 0].hist(data2.flatten(), bins=100, alpha=0.5, label=name2, density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Value Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 差异直方图
    axes[1, 1].hist(diff.flatten(), bins=100, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Difference')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Difference Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 散点图（采样以避免过多点）
    sample_size = min(10000, data1.size)
    idx = np.random.choice(data1.size, sample_size, replace=False)
    axes[1, 2].scatter(data1.flatten()[idx], data2.flatten()[idx], 
                      alpha=0.5, s=1)
    axes[1, 2].plot([data1.min(), data1.max()], [data1.min(), data1.max()], 
                   'r--', label='y=x')
    axes[1, 2].set_xlabel(name1)
    axes[1, 2].set_ylabel(name2)
    axes[1, 2].set_title('Scatter Plot (sampled)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_metrics(metrics, name="Comparison"):
    """打印比较指标"""
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Shape: {metrics['shape']}")
    print(f"Data types: {metrics['dtype1']} vs {metrics['dtype2']}")
    print(f"\nDifference Metrics:")
    print(f"  Max absolute difference: {metrics['max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {metrics['mean_abs_diff']:.6e}")
    print(f"  Standard deviation of difference: {metrics['std_diff']:.6e}")
    print(f"  RMSE: {metrics['rmse']:.6e}")
    print(f"  Max relative difference: {metrics['max_rel_diff']:.6e}")
    print(f"  Mean relative difference: {metrics['mean_rel_diff']:.6e}")
    print(f"\nSimilarity Metrics:")
    print(f"  Correlation coefficient: {metrics['correlation']:.6f}")
    print(f"  Exact match ratio: {metrics['exact_match_ratio']:.6f}")
    print(f"  Close match ratio (rtol=1e-6): {metrics['close_match_ratio_1e-6']:.6f}")
    print(f"  Close match ratio (rtol=1e-3): {metrics['close_match_ratio_1e-3']:.6f}")
    print(f"{'='*50}\n")


def compare_stages(cpp_dir, python_dir, output_dir=None):
    """比较C++和Python实现的各个阶段"""
    cpp_path = Path(cpp_dir)
    python_path = Path(python_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # 定义要比较的阶段
    stages = [
        ('preprocessed_normalized_volume', 'Preprocessing'),
        ('model_output_probability', 'Model Output'),
        ('postprocessed_segmentation_mask', 'Postprocessing')
    ]
    
    for filename, stage_name in stages:
        print(f"\n{'#'*60}")
        print(f"Comparing {stage_name}")
        print(f"{'#'*60}")
        
        # 查找文件
        cpp_files = list(cpp_path.rglob(f"{filename}*"))
        python_files = list(python_path.rglob(f"{filename}*"))
        
        cpp_file = None
        python_file = None
        
        # 找到对应的文件
        for f in cpp_files:
            if f.suffix in ['.npy', '.raw']:
                cpp_file = f
                break
        
        for f in python_files:
            if f.suffix in ['.npy', '.raw']:
                python_file = f
                break
        
        if not cpp_file or not python_file:
            print(f"Warning: Could not find matching files for {filename}")
            if not cpp_file:
                print(f"  C++ file not found in {cpp_path}")
            if not python_file:
                print(f"  Python file not found in {python_path}")
            continue
        
        print(f"C++ file: {cpp_file}")
        print(f"Python file: {python_file}")
        
        try:
            # 加载数据
            cpp_data = load_data(cpp_file)
            python_data = load_data(python_file)
            
            # 计算指标
            metrics, diff = compute_metrics(cpp_data, python_data)
            
            # 打印指标
            print_metrics(metrics, f"{stage_name} Comparison")
            
            # 绘制比较图
            fig = plot_comparison(cpp_data, python_data, diff, 
                                "C++ Result", "Python Result")
            
            if output_dir:
                fig_path = output_path / f"{filename}_comparison.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"Saved comparison plot to {fig_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error comparing {filename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare C++ and Python segmentation results')
    
    subparsers = parser.add_subparsers(dest='mode', help='Comparison mode')
    
    # 比较单个文件
    file_parser = subparsers.add_parser('files', help='Compare two files')
    file_parser.add_argument('file1', help='First file (C++ result)')
    file_parser.add_argument('file2', help='Second file (Python result)')
    file_parser.add_argument('-o', '--output', help='Output plot file')
    file_parser.add_argument('--slice', type=int, help='Slice index')
    
    # 比较整个pipeline
    pipeline_parser = subparsers.add_parser('pipeline', 
                                          help='Compare entire pipeline results')
    pipeline_parser.add_argument('cpp_dir', help='C++ results directory')
    pipeline_parser.add_argument('python_dir', help='Python results directory')
    pipeline_parser.add_argument('-o', '--output', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.mode == 'files':
        # 比较两个文件
        data1 = load_data(args.file1)
        data2 = load_data(args.file2)
        
        metrics, diff = compute_metrics(data1, data2)
        print_metrics(metrics)
        
        fig = plot_comparison(data1, data2, diff, 
                            Path(args.file1).stem, 
                            Path(args.file2).stem,
                            slice_idx=args.slice)
        
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    elif args.mode == 'pipeline':
        # 比较整个pipeline
        compare_stages(args.cpp_dir, args.python_dir, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()