#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量Bode分析系統
基於現有Bode分析架構的自動化批量處理系統
"""
from hsdata_reader import HSDataReader
from test_bode_analysis import (
    detect_periodicity, check_stability, average_periods, 
    plot_extracted_periods
)

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy import signal

def extract_frequency_from_filename(filename):
    """從檔名提取頻率"""
    # 例如: ndc100.dat -> 100
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

def validate_data_quality(records, input_channel, target_freq):
    """驗證數據品質"""
    sampling_freq = 100000
    
    # 1. 週期性檢測
    vm_signal = np.array([record['vm'][input_channel] for record in records])
    _, _, _, is_periodic = detect_periodicity(vm_signal, target_freq, sampling_freq)
    
    if not is_periodic:
        return False
    
    # 2. 主頻驗證
    fft_result = np.fft.fft(vm_signal)
    freq_axis = np.fft.fftfreq(len(vm_signal), 1/sampling_freq)
    magnitude = np.abs(fft_result)
    
    # 尋找主頻
    pos_mask = freq_axis > 0
    freq_pos = freq_axis[pos_mask]
    mag_pos = magnitude[pos_mask]
    max_idx = np.argmax(mag_pos)
    main_freq = freq_pos[max_idx]
    
    # 驗證標準
    freq_diff = abs(main_freq - target_freq)
    energy_ratio = mag_pos[max_idx] / np.max(mag_pos)
    
    return freq_diff < 2.0 and energy_ratio > 0.01

def auto_configure_fft_points(signal_length, target_freq, sampling_freq):
    """自動配置FFT點數以達到所需解析度"""
    min_resolution = 0.5  # 比1 Hz更好
    min_points = int(sampling_freq / min_resolution)
    
    if signal_length >= min_points:
        return signal_length
    else:
        # 使用零填充達到所需解析度
        return min_points

def apply_periodicity_to_channels(records, input_channel, target_freq, sampling_freq):
    """將週期性參數應用到所有VM通道"""
    # 對對應VM通道進行週期性檢測
    vm_signal = np.array([record['vm'][input_channel] for record in records])
    
    # 穩定性檢查
    stability_idx = check_stability(vm_signal)
    stable_signal = vm_signal[stability_idx:]
    
    # 週期性檢測
    period_length, start_idx, _, _ = detect_periodicity(
        stable_signal, target_freq, sampling_freq, min_periods=3
    )
    
    # 應用到所有VM通道
    processed_channels = {}
    
    for vm_channel in range(6):
        output_signal = np.array([record['vm'][vm_channel] for record in records])
        
        # 提取週期
        periods = []
        for i in range(3):
            period_start = start_idx + i * period_length
            period_end = period_start + period_length
            
            if period_end <= len(output_signal):
                period_data = output_signal[period_start:period_end]
                periods.append(period_data)
        
        # 週期平均
        if len(periods) >= 3:
            averaged_period = average_periods(periods)
            if averaged_period is not None:
                num_repeats = max(3, int(len(output_signal) / len(averaged_period)))
                processed_signal = np.tile(averaged_period, num_repeats)
                processed_channels[f'VM{vm_channel}'] = processed_signal
            else:
                processed_channels[f'VM{vm_channel}'] = output_signal[start_idx:]
        else:
            processed_channels[f'VM{vm_channel}'] = output_signal[start_idx:]
    
    return processed_channels

def analyze_single_frequency(records, input_channel, target_freq):
    """單頻率Bode分析"""
    sampling_freq = 100000
    
    # 提取輸入信號（直接使用，不預處理）
    input_signal = np.array([record['vd'][input_channel] for record in records])
    
    # 自動配置FFT點數
    n_points = auto_configure_fft_points(len(input_signal), target_freq, sampling_freq)
    
    # 輸入信號FFT
    if n_points > len(input_signal):
        # 零填充
        padded_input = np.zeros(n_points)
        padded_input[:len(input_signal)] = input_signal
        fft_input = np.fft.fft(padded_input)
    else:
        fft_input = np.fft.fft(input_signal)
    
    # 頻率軸和目標bin
    freq_axis = np.fft.fftfreq(n_points, 1/sampling_freq)
    freq_resolution = sampling_freq / n_points
    target_bin = int(target_freq / freq_resolution)
    
    # 週期性處理（僅對VM通道）
    processed_channels = apply_periodicity_to_channels(
        records, input_channel, target_freq, sampling_freq
    )
    
    # 計算所有VM通道的FRF
    results = {}
    for vm_channel in range(6):
        processed_output = processed_channels[f'VM{vm_channel}']
        
        # 輸出信號FFT
        if n_points > len(processed_output):
            padded_output = np.zeros(n_points)
            padded_output[:len(processed_output)] = processed_output
            fft_output = np.fft.fft(padded_output)
        else:
            fft_output = np.fft.fft(processed_output)
        
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
    
    print(f" 發現 {len(dat_files)} 個數據檔案")
    
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
            
            # 驗證數據品質
            if validate_data_quality(records, input_channel, freq):
                # 執行Bode分析
                frf_results = analyze_single_frequency(records, input_channel, freq)
                
                # 收集結果
                for vm_channel in range(6):
                    results[f'VM{vm_channel}']['freqs'].append(freq)
                    results[f'VM{vm_channel}']['magnitudes'].append(frf_results[f'VM{vm_channel}']['magnitude'])
                    results[f'VM{vm_channel}']['phases'].append(frf_results[f'VM{vm_channel}']['phase'])
                
                success_count += 1
                print(f"   ✓ 分析完成")
            else:
                print(f"   ✗ 數據品質不佳，跳過")
                
        except Exception as e:
            print(f"   ✗ 處理失敗: {str(e)}")
    
    print(f"\n 批量分析完成")
    print(f"   成功處理: {success_count}/{len(dat_files)} 檔案")
    
    return results

def plot_interactive_bode(results):
    """互動式波德圖"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("   ⚠ plotly未安裝，使用matplotlib替代")
        plot_matplotlib_bode(results)
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude Response', 'Phase Response'),
        vertical_spacing=0.1
    )
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (channel, data) in enumerate(results.items()):
        if len(data['freqs']) > 0:  # 確保有數據
            # 幅度響應（線性，0-1）
            fig.add_trace(
                go.Scatter(
                    x=data['freqs'], y=data['magnitudes'],
                    mode='lines+markers',
                    name=channel,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 相位響應（度）
            fig.add_trace(
                go.Scatter(
                    x=data['freqs'], y=data['phases'],
                    mode='lines+markers',
                    name=channel,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 設置對數頻率軸
    fig.update_xaxes(type="log", title="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(type="log", title="Frequency (Hz)", row=2, col=1)
    
    # 設置Y軸
    fig.update_yaxes(title="Magnitude (linear)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Phase (deg)", row=2, col=1)
    
    fig.update_layout(
        title="Complete Bode Plot - All Channels",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    # 嘗試在瀏覽器中顯示圖表
    try:
        print("   🌐 在瀏覽器中顯示互動式圖表")
        fig.show()
    except Exception as e:
        print(f"   ⚠ 無法在瀏覽器中顯示: {str(e)}")
        print("   📊 使用matplotlib版本顯示")
        plot_matplotlib_bode(results)
        return
    
    # 同時顯示matplotlib版本
    print("   📊 同時顯示matplotlib版本")
    plot_matplotlib_bode(results)

def plot_matplotlib_bode(results):
    """使用matplotlib繪製波德圖（備用方案）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (channel, data) in enumerate(results.items()):
        if len(data['freqs']) > 0:
            # 幅度響應
            ax1.semilogx(data['freqs'], data['magnitudes'], 
                        color=colors[i], linewidth=2, marker='o', 
                        markersize=6, label=channel)
            
            # 相位響應
            ax2.semilogx(data['freqs'], data['phases'], 
                        color=colors[i], linewidth=2, marker='o', 
                        markersize=6, label=channel)
    
    ax1.set_title('Magnitude Response')
    ax1.set_ylabel('Magnitude (linear)')
    ax1.set_ylim(0, 1)
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
    print("批量Bode分析系統")
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
        print(f"\n 生成互動式波德圖")
        print(f"   有效通道: {', '.join(valid_channels)}")
        plot_interactive_bode(results)
    else:
        print(f"\n 沒有有效的分析結果")
    
    print(f"\n 分析完成")

if __name__ == "__main__":
    main() 