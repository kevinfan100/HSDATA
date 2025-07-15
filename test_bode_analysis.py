#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波德圖分析測試腳本
用於驗證 HSData 檔案結構和 FFT 分析需求
"""
from hsdata_reader import HSDataReader

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.optimize import curve_fit

def test_data_structure():
    """測試數據結構"""
    print("=== 測試數據結構 ===")
    
    # 讀取數據檔案
    file_path = "dc100_0715.dat"
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return None
    
    reader = HSDataReader(file_path)
    
    # 顯示檔案信息
    info = reader.get_file_info()
    print(f"檔案大小: {info['file_size_mb']} MB")
    print(f"記錄數量: {info['record_count']:,}")
    print(f"創建時間: {info['creation_time']}")
    
    # 讀取數據記錄
    print("\n正在讀取數據記錄...")
    records = reader.read_data_records()
    
    return reader, records

def detect_periodicity(signal_data, target_freq, sampling_freq, min_periods=3):
    """
    檢測信號的週期性並找出連續的週期
    
    Args:
        signal_data: 輸入信號
        target_freq: 目標頻率
        sampling_freq: 採樣頻率
        min_periods: 最少需要的週期數
    
    Returns:
        period_length: 週期長度（採樣點數）
        start_idx: 穩定開始的索引
        periods: 週期列表
        is_periodic: 是否檢測到週期性
    """
    print(f"\n=== Periodicity Detection ===")
    print(f"Target frequency: {target_freq} Hz")
    print(f"Sampling frequency: {sampling_freq} Hz")
    
    # 計算理論週期長度
    theoretical_period = int(sampling_freq / target_freq)
    print(f"Theoretical period length: {theoretical_period} samples")
    
    # 使用自相關函數檢測週期
    autocorr = signal.correlate(signal_data, signal_data, mode='full')
    autocorr = autocorr[len(signal_data)-1:]
    
    # 週期性判斷的裕值設定
    prominence_threshold = np.std(autocorr) * 0.1  # 峰值突出度閾值
    distance_threshold = theoretical_period // 2    # 峰值間距閾值
    
    print(f"Autocorrelation std: {np.std(autocorr):.6f}")
    print(f"Prominence threshold: {prominence_threshold:.6f}")
    print(f"Distance threshold: {distance_threshold} samples")
    
    # 尋找自相關峰值（排除零延遲）
    peaks, properties = signal.find_peaks(autocorr[theoretical_period//2:], 
                                        distance=distance_threshold,
                                        prominence=prominence_threshold)
    peaks += theoretical_period//2
    
    print(f"Found {len(peaks)} peaks in autocorrelation")
    
    # 週期性判斷標準
    is_periodic = False
    if len(peaks) >= 2:  # 至少需要2個峰值才能判斷週期性
        # 計算峰值間距
        peak_distances = np.diff(peaks)
        mean_distance = np.mean(peak_distances)
        std_distance = np.std(peak_distances)
        
        print(f"Peak distances: {peak_distances}")
        print(f"Mean distance: {mean_distance:.1f} samples")
        print(f"Std distance: {std_distance:.1f} samples")
        
        # 週期性判斷：峰值間距的變異係數小於20%
        cv_threshold = 0.2
        cv = std_distance / mean_distance if mean_distance > 0 else float('inf')
        print(f"Coefficient of variation: {cv:.3f} (threshold: {cv_threshold})")
        
        if cv < cv_threshold:
            is_periodic = True
            period_length = int(mean_distance)
            print(f"✓ Periodicity detected! Period length: {period_length} samples")
        else:
            print("✗ Periodicity not detected - peak distances too variable")
            period_length = theoretical_period
    else:
        print("✗ Periodicity not detected - insufficient peaks")
        period_length = theoretical_period
    
    # 尋找穩定開始點（等待幾個週期後）
    stability_start = theoretical_period * 2  # 等待2個週期
    if stability_start >= len(signal_data):
        print("Warning: Signal too short for stability detection")
        stability_start = 0
    
    # 提取連續的週期
    periods = []
    start_idx = stability_start
    
    for i in range(min_periods):
        period_start = start_idx + i * period_length
        period_end = period_start + period_length
        
        if period_end <= len(signal_data):
            period_data = signal_data[period_start:period_end]
            periods.append(period_data)
            print(f"Period {i+1}: samples {period_start}-{period_end-1}")
        else:
            print(f"Warning: Cannot extract period {i+1}")
            break
    
    print(f"Extracted {len(periods)} periods")
    return period_length, start_idx, periods, is_periodic

def average_periods(periods):
    """
    對多個週期進行平均
    
    Args:
        periods: 週期列表，每個元素是一個週期的數據
    
    Returns:
        averaged_period: 平均後的週期數據
    """
    if not periods:
        return None
    
    # 確保所有週期長度相同
    min_length = min(len(p) for p in periods)
    periods_truncated = [p[:min_length] for p in periods]
    
    # 計算平均
    averaged_period = np.mean(periods_truncated, axis=0)
    
    print(f"Average period length: {len(averaged_period)} samples")
    print(f"Number of periods averaged: {len(periods)}")
    
    return averaged_period

def check_stability(signal_data, window_size=1000):
    """
    檢查信號穩定性
    
    Args:
        signal_data: 輸入信號
        window_size: 檢查窗口大小
    
    Returns:
        stability_index: 穩定開始的索引
    """
    print(f"\n=== Stability Check ===")
    
    # 計算移動平均和標準差
    if len(signal_data) < window_size * 2:
        print("Warning: Signal too short for stability analysis")
        return 0
    
    # 計算前半部分的統計特性
    first_half = signal_data[:len(signal_data)//2]
    second_half = signal_data[len(signal_data)//2:]
    
    first_mean = np.mean(first_half)
    first_std = np.std(first_half)
    second_mean = np.mean(second_half)
    second_std = np.std(second_half)
    
    print(f"First half - Mean: {first_mean:.6f}, Std: {first_std:.6f}")
    print(f"Second half - Mean: {second_mean:.6f}, Std: {second_std:.6f}")
    
    # 如果後半部分的標準差明顯小於前半部分，認為是穩定的
    stability_threshold = 0.5
    if second_std < first_std * stability_threshold:
        stability_index = len(signal_data) // 2
        print(f"Signal appears stable after sample {stability_index}")
    else:
        stability_index = 0
        print("Signal stability unclear")
    
    return stability_index

def plot_extracted_periods(periods, channel_name):
    """
    Plots the extracted periods overlaid to show repetition.
    """
    if not periods:
        return
        
    print(f"\n--- Visualizing extracted periods for {channel_name} ---")
    plt.figure(figsize=(10, 6))
    
    for i, p in enumerate(periods):
        plt.plot(p, label=f'Period {i+1}', alpha=0.8)
        
    plt.title(f'Overlaid Extracted Periods for {channel_name}')
    plt.xlabel('Sample in Period')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Sanitize filename
    safe_channel_name = channel_name.replace(" ", "_").replace("/", "_")
    filename = f'periods_visualization_{safe_channel_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Periods visualization saved as: {filename}")
    plt.close()  # Close plot to avoid showing it interactively during the run

def apply_periodicity_to_channels(records, input_channel, target_freq, sampling_freq, 
                                 period_length, start_idx, min_periods=3):
    """
    將檢測到的週期性參數應用到所有VM通道
    
    Args:
        records: 數據記錄
        input_channel: 輸入通道編號
        target_freq: 目標頻率
        sampling_freq: 採樣頻率
        period_length: 週期長度
        start_idx: 開始索引
        min_periods: 最少週期數
    
    Returns:
        processed_channels: 處理後的通道數據字典
    """
    print(f"\n=== Applying Periodicity to All VM Channels ===")
    print(f"Using period length: {period_length} samples")
    print(f"Using start index: {start_idx}")
    
    processed_channels = {}
    
    for output_channel in range(6):
        print(f"\nProcessing VM{output_channel}...")
        
        # 提取輸出信號
        output_signal = np.array([record['vm'][output_channel] for record in records])
        
        # 提取連續的週期
        periods = []
        for i in range(min_periods):
            period_start = start_idx + i * period_length
            period_end = period_start + period_length
            
            if period_end <= len(output_signal):
                period_data = output_signal[period_start:period_end]
                periods.append(period_data)
                print(f"  Period {i+1}: samples {period_start}-{period_end-1}")
            else:
                print(f"  Warning: Cannot extract period {i+1}")
                break
        
        # 週期平均
        if len(periods) >= 3:
            averaged_period = average_periods(periods)
            if averaged_period is not None:
                # 重複平均後的週期以達到足夠長度
                num_repeats = max(3, int(len(output_signal) / len(averaged_period)))
                processed_signal = np.tile(averaged_period, num_repeats)
                print(f"  Processed signal length: {len(processed_signal)} samples")
                processed_channels[f'VM{output_channel}'] = processed_signal
            else:
                print(f"  Using stable signal without period averaging")
                processed_channels[f'VM{output_channel}'] = output_signal[start_idx:]
        else:
            print(f"  Using stable signal without period averaging")
            processed_channels[f'VM{output_channel}'] = output_signal[start_idx:]
    
    return processed_channels

def preprocess_signal(signal_data, target_freq, sampling_freq, is_input_signal=False, channel_name=""):
    """
    信號預處理：穩定性檢查、週期檢測、週期平均
    
    Args:
        signal_data: 原始信號
        target_freq: 目標頻率
        sampling_freq: 採樣頻率
        is_input_signal: 是否為輸入信號（輸入信號不進行週期平均）
    
    Returns:
        processed_signal: 處理後的信號
        period_length: 週期長度
    """
    print(f"\n=== Signal Preprocessing ===")
    
    # 1. 穩定性檢查
    stability_idx = check_stability(signal_data)
    stable_signal = signal_data[stability_idx:]
    
    if len(stable_signal) < sampling_freq / target_freq * 3:
        print("Warning: Stable signal too short, using original signal")
        stable_signal = signal_data
    
    # 2. 週期檢測
    period_length, start_idx, periods, is_periodic = detect_periodicity(
        stable_signal, target_freq, sampling_freq, min_periods=3
    )
    
    # 3. 週期平均（僅對輸出信號進行）
    if is_periodic and len(periods) >= 3:
        plot_extracted_periods(periods, channel_name)  # VISUALIZATION
        if not is_input_signal:
            averaged_period = average_periods(periods)
            if averaged_period is not None:
                # 重複平均後的週期以達到足夠長度
                num_repeats = max(3, int(len(stable_signal) / len(averaged_period)))
                processed_signal = np.tile(averaged_period, num_repeats)
                print(f"Processed signal length: {len(processed_signal)} samples")
                return processed_signal, period_length, start_idx, is_periodic
    
    # 如果週期檢測失敗或是輸入信號，使用原始穩定信號
    if is_input_signal:
        print("Using stable input signal without period averaging")
    else:
        print("Using stable signal without period averaging")
    return stable_signal, period_length, start_idx, is_periodic

def analyze_frequency_response(records, input_channel=4, target_freq=100, sampling_freq=100000):
    """分析特定頻率的響應"""
    print(f"\n=== 頻率響應分析 ===")
    print(f"輸入通道: VD{input_channel}")
    print(f"目標頻率: {target_freq} Hz")
    print(f"採樣頻率: {sampling_freq} Hz")
    
    # 提取輸入和輸出信號
    input_signal = np.array([record['vd'][input_channel] for record in records])
    
    print(f"原始信號長度: {len(input_signal)} 點")
    print(f"輸入信號範圍: [{np.min(input_signal):.6f}, {np.max(input_signal):.6f}]")
    print(f"輸入信號平均值: {np.mean(input_signal):.6f}")
    print(f"輸入信號標準差: {np.std(input_signal):.6f}")
    
    # 直接使用原始輸入信號進行FFT，不做任何預處理
    print("\nUsing original input signal for FFT analysis.")
    processed_input = input_signal
    
    # 計算 FFT
    n_points = len(processed_input)
    fft_input = np.fft.fft(processed_input)
    
    # 計算頻率軸
    freq_axis = np.fft.fftfreq(n_points, 1/sampling_freq)
    freq_resolution = sampling_freq / n_points
    print(f"頻率解析度: {freq_resolution:.6f} Hz")
    
    # 找出目標頻率對應的 bin
    target_bin = int(target_freq / freq_resolution)
    if target_bin >= n_points // 2:
        print(f"警告: 目標頻率 {target_freq} Hz 超出奈奎斯特頻率 {sampling_freq/2} Hz")
        return None
    
    print(f"目標頻率 {target_freq} Hz 對應的 bin: {target_bin}")
    print(f"實際頻率: {freq_axis[target_bin]:.6f} Hz")
    
    # 只對對應輸入通道的VM進行週期性檢測
    main_output_channel = input_channel
    print(f"\n=== 週期性檢測 (僅對 VM{main_output_channel}) ===")
    
    main_output_signal = np.array([record['vm'][main_output_channel] for record in records])
    print(f"VM{main_output_channel} 原始信號範圍: [{np.min(main_output_signal):.6f}, {np.max(main_output_signal):.6f}]")
    
    # 對主要輸出通道進行週期性檢測
    _, _, _, is_periodic = preprocess_signal(
        main_output_signal, target_freq, sampling_freq, is_input_signal=False, channel_name=f'VM{main_output_channel}'
    )
    
    # 如果檢測到週期性，重新進行完整的週期性檢測以獲取參數
    period_length = None
    start_idx = None
    
    if is_periodic:
        print(f"\n週期性檢測成功！將參數應用到所有VM通道")
        # 重新檢測以獲取週期參數
        stability_idx = check_stability(main_output_signal)
        stable_signal = main_output_signal[stability_idx:]
        period_length, start_idx, _, _ = detect_periodicity(
            stable_signal, target_freq, sampling_freq, min_periods=3
        )
        
        # 將週期性參數應用到所有VM通道
        processed_channels = apply_periodicity_to_channels(
            records, input_channel, target_freq, sampling_freq, 
            period_length, start_idx, min_periods=3
        )
    else:
        print(f"\n週期性檢測失敗，使用穩定信號進行分析")
        # 對所有通道使用穩定信號
        processed_channels = {}
        stability_idx = check_stability(main_output_signal)
        
        for output_channel in range(6):
            output_signal = np.array([record['vm'][output_channel] for record in records])
            processed_channels[f'VM{output_channel}'] = output_signal[stability_idx:]
    
    # 分析所有輸出通道
    results = {}
    main_output_freq_check_done = False

    for output_channel in range(6):
        print(f"\n分析 VM{output_channel} 通道...")
        
        # 使用預處理後的輸出信號
        processed_output = processed_channels[f'VM{output_channel}']
        print(f"  VM{output_channel} 處理後信號範圍: [{np.min(processed_output):.6f}, {np.max(processed_output):.6f}]")
        print(f"  VM{output_channel} 處理後平均值: {np.mean(processed_output):.6f}")
        print(f"  VM{output_channel} 處理後標準差: {np.std(processed_output):.6f}")
        
        # 主頻檢查（只對對應輸入通道的VM）
        if output_channel == main_output_channel and not main_output_freq_check_done:
            print(f"\n--- 主頻檢查 (VM{output_channel}) ---")
            n_points_out = len(processed_output)
            fft_out_full = np.fft.fft(processed_output)
            freq_axis_out = np.fft.fftfreq(n_points_out, 1/sampling_freq)
            magnitude_out = np.abs(fft_out_full)
            # 只看正頻率
            pos_mask = freq_axis_out > 0
            freq_pos = freq_axis_out[pos_mask]
            mag_pos = magnitude_out[pos_mask]
            max_idx = np.argmax(mag_pos)
            main_freq = freq_pos[max_idx]
            main_mag = mag_pos[max_idx]
            print(f"  主頻: {main_freq:.2f} Hz, 主頻大小: {main_mag:.3f}")
            freq_diff = abs(main_freq - target_freq)
            if freq_diff < 2.0 and main_mag > 0.01 * np.max(mag_pos):
                print(f"  ✓ 主頻與目標頻率相符，數據可採信")
            else:
                print(f"  ⚠ 主頻與目標頻率不符或能量過低，數據不建議採信")
            main_output_freq_check_done = True
        
        # 計算輸出信號的 FFT
        fft_output = np.fft.fft(processed_output)
        
        # 計算頻率響應函數
        # 避免除零
        input_magnitude = np.abs(fft_input[target_bin])
        if input_magnitude < 1e-10:
            print(f"  警告: 輸入信號在目標頻率處太小 ({input_magnitude:.2e})")
            frf_magnitude = 0
            frf_phase = 0
        else:
            frf = fft_output[target_bin] / fft_input[target_bin]
            frf_magnitude = np.abs(frf)
            frf_phase = np.angle(frf, deg=True)
        
        # 計算 dB 值
        frf_db = 20 * np.log10(frf_magnitude) if frf_magnitude > 0 else -200
        
        results[f'VM{output_channel}'] = {
            'magnitude': frf_magnitude,
            'magnitude_db': frf_db,
            'phase': frf_phase,
            'input_magnitude': input_magnitude,
            'output_magnitude': np.abs(fft_output[target_bin])
        }
        
        print(f"  VM{output_channel} 響應:")
        print(f"    大小: {frf_magnitude:.6f} ({frf_db:.2f} dB)")
        print(f"    相位: {frf_phase:.2f} 度")
        print(f"    輸入大小: {input_magnitude:.6f}")
        print(f"    輸出大小: {np.abs(fft_output[target_bin]):.6f}")
    
    return results, freq_axis, fft_input

def plot_frequency_spectrum(freq_axis, fft_input, target_freq, sampling_freq):
    """Plot input signal spectrum (English)"""
    print(f"\n=== Plotting Input Signal Spectrum ===")
    
    # Only positive frequencies
    positive_freq_mask = freq_axis > 0
    freq_positive = freq_axis[positive_freq_mask]
    magnitude_positive = np.abs(fft_input[positive_freq_mask])
    
    # Range around target frequency
    freq_range = 50  # Hz
    target_mask = (freq_positive >= target_freq - freq_range) & (freq_positive <= target_freq + freq_range)
    
    plt.figure(figsize=(12, 8))
    
    # Full spectrum
    plt.subplot(2, 1, 1)
    plt.semilogy(freq_positive, magnitude_positive)
    plt.axvline(x=target_freq, color='red', linestyle='--', label=f'Target Frequency {target_freq} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Input Signal Spectrum (VD4)')
    plt.legend()
    plt.grid(True)
    
    # Zoomed-in around target frequency
    plt.subplot(2, 1, 2)
    plt.plot(freq_positive[target_mask], magnitude_positive[target_mask])
    plt.axvline(x=target_freq, color='red', linestyle='--', label=f'Target Frequency {target_freq} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Zoomed Spectrum ({target_freq-freq_range}-{target_freq+freq_range} Hz)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('input_spectrum.png', dpi=300, bbox_inches='tight')
    print("Input signal spectrum saved as: input_spectrum.png")
    plt.show()

def plot_bode_results(results, target_freq):
    """Plot Bode results in English, linear magnitude"""
    print(f"\n=== Plotting Bode Results ===")
    
    channels = list(results.keys())
    magnitudes = [results[ch]['magnitude'] for ch in channels]
    phases = [results[ch]['phase'] for ch in channels]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude (linear)
    x_pos = np.arange(len(channels))
    bars1 = ax1.bar(x_pos, magnitudes, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Magnitude (linear)')
    ax1.set_title(f'Frequency Response - Magnitude (Linear) @ {target_freq} Hz')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(channels)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Phase
    bars2 = ax2.bar(x_pos, phases, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Phase (deg)')
    ax2.set_title(f'Frequency Response - Phase @ {target_freq} Hz')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(channels)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}°', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('bode_results.png', dpi=300, bbox_inches='tight')
    print("Bode results saved as: bode_results.png")
    plt.show()

def print_summary_table(results):
    """Print summary table in English, linear magnitude"""
    print(f"\n=== Summary Table ===")
    print(f"{'Channel':<8} {'Magnitude':<12} {'Phase (deg)':<12} {'Input Mag':<12} {'Output Mag':<12}")
    print("-" * 70)
    
    for channel, data in results.items():
        print(f"{channel:<8} {data['magnitude']:<12.6f} {data['phase']:<12.2f} "
              f"{data['input_magnitude']:<12.6f} {data['output_magnitude']:<12.6f}")

def main():
    """主函數"""
    print("HSData 波德圖分析測試 - 修正版")
    print("=" * 50)
    
    # 測試數據結構
    result = test_data_structure()
    if result is None:
        return
    
    reader, records = result
    
    # 分析頻率響應
    results, freq_axis, fft_input = analyze_frequency_response(
        records, 
        input_channel=4,  # VD4
        target_freq=100,  # 100 Hz
        sampling_freq=100000  # 100 kHz
    )
    
    if results is None:
        print("分析失敗")
        return
    
    # 繪製輸入信號頻譜
    plot_frequency_spectrum(freq_axis, fft_input, 100, 100000)
    
    # 繪製波德圖結果
    plot_bode_results(results, 100)
    
    # 打印摘要表格
    print_summary_table(results)
    
    print(f"\n=== 測試完成 ===")


if __name__ == "__main__":
    main() 