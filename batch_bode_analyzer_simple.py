#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版批量Bode分析系統
"""
from hsdata_reader import HSDataReader
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def extract_frequency_from_filename(filename):
    """從檔名提取頻率"""
    match = re.search(r'ndc(\d+)\.dat', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"無法從檔名 {filename} 提取頻率")

def auto_detect_input_channel(records, target_freq, sampling_freq):
    """自動檢測能量最大的輸入信號通道"""
    best_channel = None
    max_energy = 0
    
    for vd_channel in range(6):
        signal_data = np.array([record['vd'][vd_channel] for record in records])
        fft_result = np.fft.fft(signal_data)
        freq_axis = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        
        # 尋找目標頻率附近的峰值
        target_bin = int(target_freq / (sampling_freq / len(signal_data)))
        if target_bin < len(fft_result) // 2:
            energy = np.abs(fft_result[target_bin])
            if energy > max_energy:
                max_energy = energy
                best_channel = vd_channel
    
    return best_channel

def analyze_single_frequency(records, input_channel, target_freq):
    """單頻率Bode分析"""
    sampling_freq = 100000
    
    # 提取輸入信號
    input_signal = np.array([record['vd'][input_channel] for record in records])
    
    # 輸入信號FFT
    fft_input = np.fft.fft(input_signal)
    
    # 頻率軸和目標bin
    freq_axis = np.fft.fftfreq(len(input_signal), 1/sampling_freq)
    freq_resolution = sampling_freq / len(input_signal)
    target_bin = int(target_freq / freq_resolution)
    
    # 計算所有VM通道的FRF
    results = {}
    for vm_channel in range(6):
        # 提取輸出信號
        output_signal = np.array([record['vm'][vm_channel] for record in records])
        
        # 輸出信號FFT
        fft_output = np.fft.fft(output_signal)
        
        # FRF計算
        input_magnitude = np.abs(fft_input[target_bin])
        if input_magnitude < 1e-10:
            frf_magnitude = 0
            frf_phase = 0
        else:
            frf = fft_output[target_bin] / fft_input[target_bin]
            frf_magnitude = np.abs(frf)
            frf_phase = np.angle(frf, deg=True)
        
        results[f'VM{vm_channel}'] = {
            'magnitude': frf_magnitude,
            'phase': frf_phase
        }
    
    return results

def batch_bode_analysis(data_folder="0715_ndc_data"):
    """批量Bode分析主函數"""
    results = {f'VM{i}': {'freqs': [], 'magnitudes': [], 'phases': []} 
               for i in range(6)}
    
    # 獲取所有.dat檔案
    dat_files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]
    dat_files.sort(key=lambda x: extract_frequency_from_filename(x))
    
    print(f"發現 {len(dat_files)} 個數據檔案")
    
    success_count = 0
    
    for i, filename in enumerate(dat_files, 1):
        file_path = os.path.join(data_folder, filename)
        freq = extract_frequency_from_filename(filename)
        
        print(f"\n [{i}/{len(dat_files)}] 處理 {filename} ({freq} Hz)")
        
        try:
            # 讀取數據
            reader = HSDataReader(file_path)
            records = reader.read_data_records()
            
            # 自動檢測輸入通道
            input_channel = auto_detect_input_channel(records, freq, 100000)
            print(f"   檢測到輸入通道: VD{input_channel}")
            
            # 執行Bode分析
            frf_results = analyze_single_frequency(records, input_channel, freq)
            
            # 收集結果
            for vm_channel in range(6):
                results[f'VM{vm_channel}']['freqs'].append(freq)
                results[f'VM{vm_channel}']['magnitudes'].append(frf_results[f'VM{vm_channel}']['magnitude'])
                results[f'VM{vm_channel}']['phases'].append(frf_results[f'VM{vm_channel}']['phase'])
            
            success_count += 1
            print(f"   ✓ 分析完成")
                
        except Exception as e:
            print(f"   ✗ 處理失敗: {str(e)}")
    
    print(f"\n批量分析完成")
    print(f"   成功處理: {success_count}/{len(dat_files)} 檔案")
    
    return results

def plot_bode_diagram(results):
    """繪製波德圖"""
    # 創建雙子圖佈局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (channel, data) in enumerate(results.items()):
        if len(data['freqs']) > 0:
            # 幅度響應
            ax1.semilogx(data['freqs'], data['magnitudes'], 
                        color=colors[i], linewidth=2, marker='o', 
                        markersize=4, label=f'CH{i+1}')
            
            # 相位響應
            ax2.semilogx(data['freqs'], data['phases'], 
                        color=colors[i], linewidth=2, marker='o', 
                        markersize=4, label=f'CH{i+1}')
    
    ax1.set_title('Magnitude Response')
    ax1.set_ylabel('Magnitude (linear)')
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    ax2.set_title('Phase Response')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """主程式"""
    print("簡化版批量Bode分析系統")
    print("=" * 50)
    
    # 檢查數據資料夾是否存在
    if not os.path.exists("0715_ndc_data"):
        print("❌ 數據資料夾 '0715_ndc_data' 不存在")
        return
    
    # 執行批量分析
    results = batch_bode_analysis("0715_ndc_data")
    
    # 檢查是否有有效結果
    valid_channels = [ch for ch, data in results.items() if len(data['freqs']) > 0]
    
    if valid_channels:
        print(f"\n生成波德圖")
        print(f"有效通道: {', '.join(valid_channels)}")
        plot_bode_diagram(results)
    else:
        print("沒有有效的分析結果")
    
    print("分析完成")

if __name__ == "__main__":
    main() 