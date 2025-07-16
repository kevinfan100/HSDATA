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
from scipy import signal

# ============================================================================
# 可調整的參數設定區塊
# ============================================================================

# 週期性檢測參數
PERIODICITY_CONFIG = {
    'min_periods': 2,           # 最少週期數 (建議: 2-5) - 降低要求
    'period_freq_tolerance': 0.15,  # 週期頻率容差 15% (建議: 0.05-0.15) - 放寬
    'peak_distance_tolerance': 0.30,  # 峰值距離容差 30% (建議: 0.15-0.25) - 放寬
    'max_autocorr_length': 200000,  # 自相關最大長度 (防止低頻卡住) - 增加
}

# 主頻率檢測參數
FREQUENCY_CONFIG = {
    'dominant_freq_tolerance': 0.10,  # 主頻率容差 10% (建議: 0.03-0.10) - 放寬
    'min_cycles_for_fft': 4,     # FFT最少週期數 (建議: 4-16) - 降低
    'max_fft_points': 65536,     # FFT最大點數 (防止低頻卡住)
}

# 系統參數
SYSTEM_CONFIG = {
    'sampling_freq': 100000,     # 採樣頻率 (Hz)
    'autocorr_height_ratio': 0.3,  # 自相關峰值高度比例 (建議: 0.3-0.7) - 降低
    'low_freq_threshold': 50,    # 低頻閾值 (Hz)，低於此值使用特殊處理 - 提高
}

# ============================================================================

def print_current_config():
    """顯示當前配置"""
    print("當前配置設定:")
    print("=" * 50)
    print("週期性檢測參數:")
    for key, value in PERIODICITY_CONFIG.items():
        print(f"  {key}: {value}")
    print("\n主頻率檢測參數:")
    for key, value in FREQUENCY_CONFIG.items():
        print(f"  {key}: {value}")
    print("\n系統參數:")
    for key, value in SYSTEM_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 50)

def extract_frequency_from_filename(filename):
    """從檔名提取頻率 - 支援dc、ndc、pi格式"""
    # 嘗試匹配 dc{頻率}.dat 格式
    match = re.search(r'dc([0-9]+\.?[0-9]*)\.dat', filename)
    if match:
        return float(match.group(1))
    else:
        # 嘗試匹配 ndc{頻率}.dat 格式
        match = re.search(r'ndc(\d+)\.dat', filename)
        if match:
            return int(match.group(1))
        else:
            # 嘗試匹配 pi{頻率}.dat 格式
            match = re.search(r'pi([0-9]+\.?[0-9]*)\.dat', filename)
            if match:
                return float(match.group(1))
            else:
                raise ValueError(f"無法從檔名 {filename} 提取頻率")

def check_periodicity(signal_data, sampling_freq, target_freq):
    """檢測信號的週期性，使用FFT方法統一檢測"""
    data_length = len(signal_data)
    
    # 從配置中獲取參數
    freq_tolerance = PERIODICITY_CONFIG['period_freq_tolerance']
    low_freq_threshold = SYSTEM_CONFIG['low_freq_threshold']
    
    # 計算理論週期長度（採樣點數）
    theoretical_period = sampling_freq / target_freq
    
    # 低頻特殊處理
    if target_freq < low_freq_threshold:
        # 對於低頻，限制數據長度
        max_corr_length = min(data_length, PERIODICITY_CONFIG['max_autocorr_length'])
        # 確保至少包含2個週期
        min_corr_length = int(theoretical_period * 2)
        if max_corr_length < min_corr_length:
            max_corr_length = min(min_corr_length, data_length)
        signal_data = signal_data[:max_corr_length]
        data_length = len(signal_data)
    
    # 確保數據長度足夠包含至少2個週期
    min_required_length = theoretical_period * 2
    if data_length < min_required_length:
        return False, 0, f"數據長度不足: {data_length} < {min_required_length:.0f}"
    
    # 使用FFT檢測主頻率
    fft_result = np.fft.fft(signal_data)
    freq_axis = np.fft.fftfreq(data_length, 1/sampling_freq)
    
    # 只考慮正頻率部分
    positive_freqs = freq_axis[:len(freq_axis)//2]
    positive_fft = np.abs(fft_result[:len(fft_result)//2])
    
    # 找到最大幅值對應的頻率
    max_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[max_idx]
    
    # 檢查主頻率是否與目標頻率接近
    if abs(dominant_freq - target_freq) <= target_freq * freq_tolerance:
        # 計算週期長度
        period_length = sampling_freq / dominant_freq
        return True, period_length, f"週期性確認: 主頻率{dominant_freq:.1f}Hz"
    else:
        return False, 0, f"週期頻率不匹配: {dominant_freq:.1f}Hz vs {target_freq}Hz"

def get_dominant_frequency(signal_data, sampling_freq, target_freq=None):
    """獲取信號的主頻率，可選優化FFT點數"""
    data_length = len(signal_data)
    
    # 從配置中獲取參數
    min_cycles_for_fft = FREQUENCY_CONFIG['min_cycles_for_fft']
    max_fft_points = FREQUENCY_CONFIG['max_fft_points']
    low_freq_threshold = SYSTEM_CONFIG['low_freq_threshold']
    
    # 如果指定了目標頻率，優化FFT點數
    if target_freq is not None:
        # 計算理論週期
        theoretical_period = sampling_freq / target_freq
        
        # 低頻特殊處理
        if target_freq < low_freq_threshold:
            # 對於低頻，減少FFT週期數要求
            min_cycles_for_fft = max(4, min_cycles_for_fft // 2)
        
        # 確保FFT點數是2的冪次，且包含足夠的週期
        min_fft_points = int(theoretical_period * min_cycles_for_fft)
        optimal_fft_points = 2 ** int(np.log2(min_fft_points))
        
        # 限制最大FFT點數
        optimal_fft_points = min(optimal_fft_points, max_fft_points)
        
        # 如果數據長度不足，使用零填充
        if data_length < optimal_fft_points:
            padded_signal = np.pad(signal_data, (0, optimal_fft_points - data_length), 'constant')
            fft_result = np.fft.fft(padded_signal)
            freq_axis = np.fft.fftfreq(optimal_fft_points, 1/sampling_freq)
        else:
            # 如果數據足夠長，使用實際長度
            fft_result = np.fft.fft(signal_data)
            freq_axis = np.fft.fftfreq(data_length, 1/sampling_freq)
    else:
        # 使用原始數據長度
        fft_result = np.fft.fft(signal_data)
        freq_axis = np.fft.fftfreq(data_length, 1/sampling_freq)
    
    # 只考慮正頻率部分
    positive_freqs = freq_axis[:len(freq_axis)//2]
    positive_fft = np.abs(fft_result[:len(freq_axis)//2])
    
    # 找到最大幅值對應的頻率
    max_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[max_idx]
    
    return dominant_freq, positive_fft[max_idx]

def validate_vm_channel(records, vm_channel, target_freq, sampling_freq=None):
    """驗證單一VM通道的週期性和主頻率匹配"""
    if sampling_freq is None:
        sampling_freq = SYSTEM_CONFIG['sampling_freq']
    
    vm_signal = np.array([record['vm'][vm_channel] for record in records])
    
    # 檢測週期性（現在使用FFT方法）
    is_periodic, period_length, period_msg = check_periodicity(vm_signal, sampling_freq, target_freq)
    
    if not is_periodic:
        return False, period_msg
    
    # 週期性檢測已經確認了主頻率，直接返回成功
    return True, period_msg

def auto_detect_input_channel(records, target_freq, sampling_freq=None):
    """自動檢測能量最大的輸入信號通道"""
    if sampling_freq is None:
        sampling_freq = SYSTEM_CONFIG['sampling_freq']
    
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

def analyze_single_frequency(records, input_channel, target_freq, sampling_freq=None):
    """單頻率Bode分析"""
    if sampling_freq is None:
        sampling_freq = SYSTEM_CONFIG['sampling_freq']
    
    # 提取輸入信號
    input_signal = np.array([record['vd'][input_channel] for record in records])
    
    # 輸入信號FFT
    fft_input = np.fft.fft(input_signal)
    
    # 頻率軸和目標bin
    freq_axis = np.fft.fftfreq(len(input_signal), 1/sampling_freq)
    freq_resolution = sampling_freq / len(input_signal)
    target_bin = int(target_freq / freq_resolution)
    
    # 分析所有VM通道
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

def batch_bode_analysis(data_folder="0716_hsdata_pi"):
    """批量Bode分析主函數"""
    results = {f'VM{i}': {'freqs': [], 'magnitudes': [], 'phases': []} 
               for i in range(6)}
    
    # 記錄VD通道信息
    vd_info = {}
    
    # 獲取所有.dat檔案
    dat_files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]
    dat_files.sort(key=lambda x: extract_frequency_from_filename(x))
    
    print(f"Found {len(dat_files)} data files")
    
    success_count = 0
    
    for i, filename in enumerate(dat_files, 1):
        file_path = os.path.join(data_folder, filename)
        freq = extract_frequency_from_filename(filename)
        
        print(f"[{i}/{len(dat_files)}] {filename} ({freq}Hz)")
        
        try:
            # 讀取數據
            print(f"  Reading data...")
            reader = HSDataReader(file_path)
            records = reader.read_data_records()
            print(f"  Data length: {len(records)} records")
            
            # 自動檢測輸入通道
            print(f"  Detecting input channel...")
            input_channel = auto_detect_input_channel(records, freq)
            print(f"  Detected: VD{input_channel}")
            
            # 驗證對應的VM通道
            print(f"  Validating VM channel...")
            is_valid, validation_msg = validate_vm_channel(records, input_channel, freq)
            if not is_valid:
                print(f"  ✗ VM{input_channel}: {validation_msg}")
                # 顯示調試信息
                if "週期數不足" in validation_msg:
                    vm_signal = np.array([record['vm'][input_channel] for record in records])
                    theoretical_period = SYSTEM_CONFIG['sampling_freq'] / freq
                    print(f"    Debug: Data length={len(vm_signal)}, Theoretical period={theoretical_period:.0f} points")
                    print(f"    Debug: Required periods={PERIODICITY_CONFIG['min_periods']}")
                continue
            
            # 執行Bode分析
            print(f"  Executing Bode analysis...")
            frf_results = analyze_single_frequency(records, input_channel, freq)
            
            # 收集所有VM通道的結果
            for vm_channel in range(6):
                results[f'VM{vm_channel}']['freqs'].append(freq)
                results[f'VM{vm_channel}']['magnitudes'].append(frf_results[f'VM{vm_channel}']['magnitude'])
                results[f'VM{vm_channel}']['phases'].append(frf_results[f'VM{vm_channel}']['phase'])
            
            # 記錄VD通道信息
            vd_info[freq] = input_channel
            
            success_count += 1
            print(f"  ✓ VD{input_channel} → All VM channels")
                
        except Exception as e:
            print(f"  ✗ {str(e)}")
    
    print(f"\nCompleted: {success_count}/{len(dat_files)} files")
    
    # 顯示VD通道信息
    if vd_info:
        print("\nVD Channel Mapping:")
        for freq in sorted(vd_info.keys()):
            print(f"  {freq}Hz → VD{vd_info[freq]}")
    
    return results, vd_info

def plot_bode_diagram(results, vd_info):
    """繪製波德圖"""
    # 創建雙子圖佈局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # 找出有數據的通道
    valid_channels = []
    for i, (channel, data) in enumerate(results.items()):
        if len(data['freqs']) > 0:
            valid_channels.append((i, channel, data))
    
    # 繪製每個有效通道
    for i, channel, data in valid_channels:
        # 幅度響應 - 所有VM通道
        ax1.semilogx(data['freqs'], data['magnitudes'], 
                    color=colors[i], linewidth=2, marker='o', 
                    markersize=4, label=f'ch{i+1}')
        
        # 相位響應 - 只顯示有輸入VD的通道
        if i in vd_info.values():  # 只顯示有輸入的VD通道對應的VM通道
            ax2.semilogx(data['freqs'], data['phases'], 
                        color=colors[i], linewidth=2, marker='o', 
                        markersize=4, label=f'ch{i+1}')
    
    ax1.set_title('Magnitude Response')
    ax1.set_ylabel('Magnitude')
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
    print("Simplified Batch Bode Analysis System")
    print("=" * 50)
    
    # 顯示當前配置
    print_current_config()
    
    
    # 執行批量分析
    results, vd_info = batch_bode_analysis("0716_hsdata_ndc")
    
    # 檢查是否有有效結果
    valid_channels = [ch for ch, data in results.items() if len(data['freqs']) > 0]
    
    if valid_channels:
        print(f"\nGenerating Bode Diagram")
        print(f"Valid Channels: {', '.join(valid_channels)}")
        plot_bode_diagram(results, vd_info)
    else:
        print("No valid analysis results")
    
    print("Analysis Complete")

if __name__ == "__main__":
    main() 