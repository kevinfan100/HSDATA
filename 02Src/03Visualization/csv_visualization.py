#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 資料視覺化工具
從檔案名稱自動提取頻率，並視覺化 VM、VD、DA 資料
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import re
import os
import argparse
from pathlib import Path

def extract_frequency_from_filename(filename: str) -> float:
    """
    從檔案名稱中提取頻率數字
    
    Args:
        filename: 檔案名稱 (例如: ndc100.csv, dc50.csv, pi200.csv)
        
    Returns:
        提取的頻率值 (Hz)
    """
    # 移除副檔名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 使用正則表達式提取數字
    match = re.search(r'(\d+)', name_without_ext)
    if match:
        frequency = float(match.group(1))
        print(f"Extracted frequency from filename '{filename}': {frequency}Hz")
        return frequency
    else:
        print(f"Warning: Cannot extract frequency from filename '{filename}', using default 10Hz")
        return 10.0

def convert_da_to_current(da_data: np.ndarray) -> np.ndarray:
    """
    將DA數據轉換為電流
    
    Args:
        da_data: DA數據 (16bit, 0-65535)
        
    Returns:
        電流數據 (A)
    """
    # DA轉換參數
    da_max = 65535  # 16bit最大值
    voltage_range = 20  # ±10V = 20V總範圍
    current_ratio = 0.6  # 電壓到電流的轉換比例
    
    # 轉換步驟：
    # 1. DA值 -> 電壓 (-10V to +10V)
    voltage = (da_data / da_max - 0.5) * voltage_range
    
    # 2. 電壓 -> 電流
    current = voltage * current_ratio
    
    return current

def visualize_csv_data(csv_file, start_index=0, num_samples=None, sampling_freq=100000, target_freq=None):
    """
    讀取 CSV 檔案並視覺化 VM、VD、DA 資料
    
    Parameters:
    csv_file: CSV 檔案路徑
    start_index: 起始索引
    num_samples: 要提取的樣本數
    sampling_freq: 採樣頻率 (Hz)
    target_freq: 目標頻率 (Hz)，如果為 None 則從檔案名稱提取
    """
    # 如果沒有指定目標頻率，從檔案名稱提取
    if target_freq is None:
        filename = os.path.basename(csv_file)
        target_freq = extract_frequency_from_filename(filename)
    
    print(f"Target frequency: {target_freq} Hz")
    
    # 讀取 CSV 檔案
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read CSV file, total {len(df)} records")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # 檢查索引範圍
    if start_index >= len(df):
        print(f"Error: Starting index {start_index} is out of range (0-{len(df)-1})")
        return None
    
    # 如果沒有指定樣本數，使用全部資料
    if num_samples is None:
        num_samples = len(df) - start_index
    
    end_index = min(start_index + num_samples, len(df))
    actual_samples = end_index - start_index
        
    # 提取指定範圍的資料
    data_subset = df.iloc[start_index:end_index]
    
    # 分離 VM、VD、DA 資料
    vm_columns = [col for col in df.columns if col.startswith('vm_')]
    vd_columns = [col for col in df.columns if col.startswith('vd_')]
    da_columns = [col for col in df.columns if col.startswith('da_')]

    
    # 建立時間軸 (秒)
    time_axis = np.arange(actual_samples) / sampling_freq
    
    # 設定 matplotlib 顯示參數
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    
    # 建立三個子圖
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{target_freq}Hz (Index {start_index} to {end_index-1})', 
                 fontsize=16, fontweight='bold')
    
    # 繪製 VM 資料
    ax1.set_title('VM Data', fontsize=14, fontweight='bold')
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, col in enumerate(vm_columns):
        color = colors[i % len(colors)]
        ax1.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=1.5, color=color)
    ax1.set_ylabel('V')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_axis[-1])
    
    # 繪製 VD 資料
    ax2.set_title('VD Data', fontsize=14, fontweight='bold')
    for i, col in enumerate(vd_columns):
        color = colors[i % len(colors)]
        ax2.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=1.5, color=color)
    ax2.set_ylabel('V')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_axis[-1])
    
    # 繪製 DA 資料（轉換為電流）
    ax3.set_title('DA Current', fontsize=14, fontweight='bold')
    for i, col in enumerate(da_columns):
        color = colors[i % len(colors)]
        # 將 DA 數據轉換為電流
        da_current = convert_da_to_current(data_subset[col].values)
        ax3.plot(time_axis, da_current, label=f'ch{i+1}', linewidth=1.5, color=color)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('A')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, time_axis[-1])
    
    plt.tight_layout()
    plt.show()
    
    # 計算並顯示時間長度
    duration = actual_samples / sampling_freq
    print(f"Time duration: {duration:.6f} seconds ({duration*1000:.3f} ms)")
    
    return data_subset

def create_individual_plots(csv_file, start_index=0, num_samples=None, sampling_freq=100000, target_freq=None):
    """
    為 VM、VD 和 DA 資料建立個別視窗的圖表
    """
    
    print(f"\nCreating individual plots...")
    
    # 如果沒有指定目標頻率，從檔案名稱提取
    if target_freq is None:
        filename = os.path.basename(csv_file)
        target_freq = extract_frequency_from_filename(filename)
    
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file)
    
    # 如果沒有指定樣本數，使用全部資料
    if num_samples is None:
        num_samples = len(df) - start_index
    
    data_subset = df.iloc[start_index:start_index+num_samples]
    
    # 分離資料
    vm_columns = [col for col in df.columns if col.startswith('vm_')]
    vd_columns = [col for col in df.columns if col.startswith('vd_')]
    da_columns = [col for col in df.columns if col.startswith('da_')]
    
    time_axis = np.arange(num_samples) / sampling_freq
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 建立個別圖表
    for data_type, columns in [('VM', vm_columns), ('VD', vd_columns), ('DA', da_columns)]:
        plt.figure(figsize=(12, 8))
        
        # 特殊處理 DA 資料（轉換為電流）
        if data_type == 'DA':
            plt.title(f'DA Current - {target_freq}Hz (Index {start_index} to {start_index+num_samples-1})', 
                     fontsize=16, fontweight='bold')
            ylabel = 'Current (A)'
        else:
            plt.title(f'{data_type} Data - {target_freq}Hz (Index {start_index} to {start_index+num_samples-1})', 
                     fontsize=16, fontweight='bold')
            ylabel = f'{data_type} Values'
        
        for i, col in enumerate(columns):
            color = colors[i % len(colors)]
            if data_type == 'DA':
                # 將 DA 數據轉換為電流
                da_current = convert_da_to_current(data_subset[col].values)
                plt.plot(time_axis, da_current, label=f'ch{i+1}', linewidth=2, color=color)
            else:
                plt.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=2, color=color)
        
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='CSV Data Visualization')
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='CSV file path (e.g., 01Data/02Processed_csv/0716_hsdata_ndc/ndc100.csv)')
    parser.add_argument('--start-index', '-s', type=int, default=0,
                       help='Start index (default: 0)')
    parser.add_argument('--num-samples', '-n', type=int, default=None,
                       help='Number of samples (default: all data)')
    parser.add_argument('--sampling-freq', type=int, default=100000,
                       help='Sampling frequency (default: 100000 Hz)')
    parser.add_argument('--target-freq', type=float, default=None,
                       help='Target frequency (default: auto-extract from filename)')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Create individual plots')
    
    args = parser.parse_args()
    
    print("CSV Data Visualization")
    print("=" * 50)
    print(f"CSV file: {args.file}")
    print(f"Start index: {args.start_index}")
    if args.num_samples is None:
        print("Number of samples: All data")
    else:
        print(f"Number of samples: {args.num_samples}")
    print(f"Sampling frequency: {args.sampling_freq} Hz")
    if args.target_freq:
        print(f"Specified target frequency: {args.target_freq} Hz")
    else:
        print("Target frequency: Auto-extract from filename")
    # 檢查檔案是否存在
    if not os.path.exists(args.file):
        print(f"Error: File not found {args.file}")
        print("Please check the file path")
        print("Recommended to run from project root, for example:")
        print("python 02Src\\03Visualization\\csv_visualization_improved.py --file 01Data\\02Processed_csv\\0716_hsdata_ndc\\ndc100.csv")
        return
    
    try:
        # 執行視覺化
        data = visualize_csv_data(
            args.file, 
            args.start_index, 
            args.num_samples, 
            args.sampling_freq,
            args.target_freq
        )
        
        # 如果需要，建立個別圖表
        if args.individual:
            create_individual_plots(
                args.file, 
                args.start_index, 
                args.num_samples, 
                args.sampling_freq,
                args.target_freq
            )
        
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 