import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def dac_to_voltage(dac_value):
    """16-bit DAC code to voltage conversion"""
    return (dac_value - 32768) * (20.0 / 65536)

# def dac_voltage_to_current(dac_value):
#     """voltage to apms current"""
#     return dac_value * 0.3

def load_single_test(csv_file):
    """load single data from CSV file"""
    df = pd.read_csv(csv_file)
    vm = df[[f'vm_{i}' for i in range(6)]].values.T
    da = df[[f'da_{i}' for i in range(6)]].values.T
    
    # Apply negative sign to DAC data for specific files
    import os
    filename = os.path.basename(csv_file)
    if filename in ['1_1.csv', '3_1.csv', '6_1.csv']:
        da = -da
    
    return {'vm': vm, 'da': da}

def find_excitation_channel(da_data):
    return np.argmax(np.sqrt(np.mean(da_data**2, axis=1)))

def detect_steady_state_by_periods(csv_file, target_freq, sampling_rate=100000,
                                 start_period=1, consecutive_periods=3, 
                                 check_points=10, threshold=1e-3):
    
    """detect steady state by checking consecutive periods"""
    
    df = pd.read_csv(csv_file)
    exclude_indices = np.arange(0, len(df), 10000)
    valid_mask = np.ones(len(df), dtype=bool)
    valid_mask[exclude_indices] = False
    
    valid_df = df[valid_mask].reset_index(drop=True)
    vm_data = valid_df[[f'vm_{i}' for i in range(6)]].values.T
    original_indices = np.where(valid_mask)[0]
    
    period_samples = int(sampling_rate / target_freq)
    max_periods = len(valid_df) // period_samples
    check_positions = np.linspace(0, period_samples-1, check_points, dtype=int)
    
    steady_periods = []
    
    for vm_ch in range(6):
        signal = vm_data[vm_ch]
        
        for test_period in range(start_period, max_periods - consecutive_periods):
            all_stable = True
            
            for i in range(consecutive_periods):
                current_period = test_period + i
                next_period = current_period + 1
                
                current_start = current_period * period_samples
                next_start = next_period * period_samples
                
                max_diff = 0
                for pos in check_positions:
                    current_val = signal[current_start + pos]
                    next_val = signal[next_start + pos]
                    diff = abs(current_val - next_val)
                    max_diff = max(max_diff, diff)
                
                if max_diff >= threshold:
                    all_stable = False
                    break
            
            if all_stable:
                steady_periods.append(test_period)
                break
    
    if steady_periods:
        recommended_period = max(steady_periods)
        valid_index = recommended_period * period_samples
        original_index = original_indices[valid_index]
        
        return {
            'period': recommended_period,
            'index': original_index,
            'max_periods': max_periods
        }
    
    return None

def calculate_transfer_function_with_validation(input_signal, output_signal, fs, target_freq, 
                                              tolerance_percent=5):
    """
    正確的邏輯：在容差範圍內找最大能量點，並使用該點計算傳遞函數
    """
    
    # 去除DC分量
    input_clean = input_signal #- np.mean(input_signal)
    output_clean = output_signal #- np.mean(output_signal)
    
    # FFT計算
    input_fft = fft(input_clean)
    output_fft = fft(output_clean)
    freqs = fftfreq(len(input_clean), 1/fs)
    
    # 正頻率範圍
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_input_fft = input_fft[positive_mask]
    positive_output_fft = output_fft[positive_mask]
    
    # === 步驟1: 定義容差範圍 ===
    tolerance = target_freq * tolerance_percent / 100
    freq_range_min = target_freq - tolerance
    freq_range_max = target_freq + tolerance
    
    # === 步驟2: 在容差範圍內找VM響應的最大能量點 ===
    freq_mask = (positive_freqs >= freq_range_min) & (positive_freqs <= freq_range_max)
    
    if not np.any(freq_mask):
        print(f"    ✗ 容差範圍[{freq_range_min:.2f}, {freq_range_max:.2f}]Hz內無頻率點")
        return 0, 0, 0, False
    
    # 在容差範圍內找最大能量點
    vm_power = np.abs(positive_output_fft)**2
    masked_indices = np.where(freq_mask)[0]
    masked_powers = vm_power[masked_indices]
    
    # 找到範圍內最大能量點的索引
    local_max_idx = np.argmax(masked_powers)
    global_max_idx = masked_indices[local_max_idx]
    
    # === 步驟3: 獲取最大能量點的信息 ===
    actual_freq = positive_freqs[global_max_idx]
    max_power = masked_powers[local_max_idx]
    
    # === 步驟4: 驗證這就是全局最大值 ===
    global_max_power = np.max(vm_power)
    power_ratio = max_power / global_max_power
    
    # === 步驟5: 使用最大能量點計算傳遞函數 ===
    # 注意：使用actual_freq對應的索引，而不是強制的target_freq
    input_complex = positive_input_fft[global_max_idx]
    output_complex = positive_output_fft[global_max_idx]
    H_complex = output_complex / input_complex
    
    signed_gain = np.real(H_complex)
    magnitude = np.abs(H_complex)
    phase_deg = np.angle(H_complex) * 180 / np.pi
    
    # === 步驟6: 驗證結果 ===
    freq_error = abs(actual_freq - target_freq)
    freq_error_percent = freq_error / target_freq * 100
    
    # 驗證條件
    is_valid = (
        freq_error <= tolerance and      # 頻率在容差內
        power_ratio > 0.5               # 是主要能量成分
    )
    
    print(f"    目標{target_freq}Hz±{tolerance_percent}% → 最大能量@{actual_freq:.2f}Hz")
    print(f"    誤差{freq_error_percent:.1f}%, 能量佔比{power_ratio:.3f}")
    
    return signed_gain, magnitude, phase_deg, is_valid

def plot_frequency_analysis(csv_file, target_freq, steady_info, tolerance_percent=5):
    """繪製頻域分析圖"""
    
    data = load_single_test(csv_file)
    da_voltage = dac_to_voltage(data['da'])
    excited_ch = find_excitation_channel(da_voltage)
    
    # 計算穩態後的完整週期數據
    period_samples = int(100000 / target_freq)
    steady_period = steady_info['period']
    max_periods = steady_info['max_periods']
    available_periods = max_periods - steady_period
    
    start_idx = steady_info['index']
    end_idx = start_idx + available_periods * period_samples
    
    input_signal = da_voltage[excited_ch][start_idx:end_idx]
    
    print(f"繪圖使用: 第{steady_period}週期後的{available_periods}個完整週期")
    
    # 預處理和FFT
    input_clean = input_signal# - np.mean(input_signal)
    input_fft = fft(input_clean)
    freqs = fftfreq(len(input_clean), 1/100000)
    
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    input_fft_pos = input_fft[positive_mask]
    
    # === 輸入信號分析圖 ===
    fig_input, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
    fig_input.suptitle(f'Input Signal Analysis - DA_{excited_ch} (Target: {target_freq} Hz)', fontsize=14)
    
    # 時域圖
    time_axis = np.arange(len(input_signal)) / 100000
    ax_time.plot(time_axis, input_signal)
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Amplitude (V)')
    ax_time.set_title(f'Time Domain - DA_{excited_ch}')
    ax_time.grid(True, alpha=0.3)
    
    # 頻域圖
    input_magnitude = np.abs(input_fft_pos)
    max_input_idx = np.argmax(input_magnitude)
    max_input_freq = freqs_pos[max_input_idx]
    
    ax_freq.loglog(freqs_pos, input_magnitude)
    ax_freq.scatter(max_input_freq, input_magnitude[max_input_idx], color='red', s=100, zorder=5,
                    label=f'Max: {max_input_freq:.1f} Hz')
    ax_freq.axvline(target_freq, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Target: {target_freq} Hz')
    ax_freq.set_xlabel('Frequency (Hz)')
    ax_freq.set_ylabel('|Input(f)| Magnitude')
    ax_freq.set_title(f'Frequency Domain - DA_{excited_ch}')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend()
    
    plt.tight_layout()
    plt.show()
    
    # === VM通道分析圖 ===
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))
    fig.suptitle(f'VM Analysis - DA_{excited_ch} Excitation (Target: {target_freq} Hz)', fontsize=14)
    
    for vm_ch in range(6):
        output_signal = data['vm'][vm_ch][start_idx:end_idx]
        
        # 預處理和FFT
        output_clean = output_signal #- np.mean(output_signal)
        output_fft = fft(output_clean)
        output_fft_pos = output_fft[positive_mask]
        
        # 計算傳遞函數
        valid_mask = np.abs(input_fft_pos) > 1e-10
        H_complex = np.zeros_like(input_fft_pos, dtype=complex)
        H_complex[valid_mask] = output_fft_pos[valid_mask] / input_fft_pos[valid_mask]
        
        # VM響應頻譜
        vm_magnitude = np.abs(output_fft_pos)
        max_vm_idx = np.argmax(vm_magnitude)
        max_vm_freq = freqs_pos[max_vm_idx]
        
        axes[vm_ch, 0].loglog(freqs_pos, vm_magnitude)
        axes[vm_ch, 0].scatter(max_vm_freq, vm_magnitude[max_vm_idx], color='red', s=50, zorder=5)
        axes[vm_ch, 0].axvline(target_freq, color='orange', linestyle='--', alpha=0.7)
        axes[vm_ch, 0].set_title(f'VM_{vm_ch} Response')
        axes[vm_ch, 0].set_ylabel('|VM(f)|')
        axes[vm_ch, 0].grid(True, alpha=0.3)
        
        # 傳遞函數頻譜
        H_magnitude = np.abs(H_complex)
        freqs_valid = freqs_pos[valid_mask]
        max_h_idx = np.argmax(H_magnitude[valid_mask])
        max_h_freq = freqs_valid[max_h_idx]
        
        axes[vm_ch, 1].loglog(freqs_valid, H_magnitude[valid_mask])
        axes[vm_ch, 1].scatter(max_h_freq, H_magnitude[valid_mask][max_h_idx], color='red', s=50, zorder=5)
        axes[vm_ch, 1].axvline(target_freq, color='orange', linestyle='--', alpha=0.7)
        axes[vm_ch, 1].set_title(f'H_{vm_ch} Transfer Function')
        axes[vm_ch, 1].set_ylabel('|H(f)|')
        axes[vm_ch, 1].grid(True, alpha=0.3)
        
        if vm_ch == 5:  # 最後一行添加x軸標籤
            axes[vm_ch, 0].set_xlabel('Frequency (Hz)')
            axes[vm_ch, 1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()

def calculate_b_matrix_integrated(csv_files, target_freq, tolerance_percent=5,
                                steady_params=None, plot_file=None):
    """
    整合計算B矩陣
    
    Parameters:
    ----------
    tolerance_percent : float, VM驗證容差百分比
    plot_file : str or None, 要繪圖的檔案名稱，None表示不繪圖
    """
    
    if steady_params is None:
        steady_params = {
            'consecutive_periods': 3,
            'check_points': 10,
            'threshold': 1e-3
        }
    
    print(f"B矩陣計算: {target_freq}Hz, VM驗證容差±{tolerance_percent}%")
    
    # === 統一穩態檢測 ===
    all_steady_info = []
    
    for csv_file in csv_files:
        result = detect_steady_state_by_periods(csv_file, target_freq, **steady_params)
        if result:
            all_steady_info.append(result)
        else:
            print(f"錯誤: {csv_file} 未檢測到穩態")
            return None
    
    # 使用最保守的穩態點
    unified_steady_period = max([info['period'] for info in all_steady_info])
    print(f"統一穩態點: 第{unified_steady_period}週期")
    
    # === B矩陣計算 ===
    period_samples = int(100000 / target_freq)
    B_signed = np.zeros((6, 6))
    B_magnitude = np.zeros((6, 6))
    B_phase = np.zeros((6, 6))
    validation_results = np.zeros((6, 6), dtype=bool)
    
    for i, csv_file in enumerate(csv_files):
        data = load_single_test(csv_file)
        da_voltage = dac_to_voltage(data['da'])
        excited_ch = find_excitation_channel(da_voltage)
        
        # 計算可用的完整週期數
        steady_info = all_steady_info[i]
        available_periods = steady_info['max_periods'] - unified_steady_period
        
        start_idx = steady_info['index'] + (unified_steady_period - steady_info['period']) * period_samples
        end_idx = start_idx + available_periods * period_samples
        
        input_signal = da_voltage[excited_ch][start_idx:end_idx]
        
        print(f"{csv_file}: excited_ch{excited_ch}, 使用{available_periods}個完整週期")
        
        # === 繪圖 ===
        if plot_file == csv_file:
            plot_frequency_analysis(csv_file, target_freq, 
                                   {'period': unified_steady_period, 'index': start_idx, 
                                    'max_periods': steady_info['max_periods']}, 
                                   tolerance_percent)
        
        for vm_ch in range(6):
            output_signal = data['vm'][vm_ch][start_idx:end_idx]
            
            signed_gain, magnitude, phase_deg, is_valid = calculate_transfer_function_with_validation(
                input_signal, output_signal, 100000, target_freq, tolerance_percent
            )
            
            B_signed[vm_ch, excited_ch] = signed_gain
            B_magnitude[vm_ch, excited_ch] = magnitude
            B_phase[vm_ch, excited_ch] = phase_deg
            validation_results[vm_ch, excited_ch] = is_valid
    
    # === 顯示結果 ===
    print(f"\n含符號B矩陣 (VM行×DA列):")
    print("=" * 50)
    for i in range(6):
        row_str = f"VM_{i}  "
        for j in range(6):
            row_str += f"{B_signed[i, j]:+7.4f}   "
        print(row_str)
    
    print(f"\n幅值矩陣:")
    print(B_magnitude)
    
    print(f"\n相位矩陣 (度):")
    print(B_phase)
    
    total_valid = np.sum(validation_results)
    print(f"\n驗證統計: {total_valid}/36 通過 ({total_valid/36*100:.1f}%)")
    
    return {
        'B_signed': B_signed,
        'B_magnitude': B_magnitude,
        'B_phase': B_phase,
        'validation_results': validation_results
    }

# === 使用範例 ===
if __name__ == "__main__":
    
    csv_files = ['1_1.csv', '2_1.csv', '3_1.csv', '4_1.csv', '5_1.csv', '6_1.csv']
    
    # 穩態檢測參數
    steady_params = {
        'consecutive_periods': 2,
        'check_points': 5,
        'threshold': 2e-3
    }
    
    # 計算B矩陣並繪製指定檔案的圖形
    result = calculate_b_matrix_integrated(
        csv_files=csv_files,
        target_freq=1,              # 目標頻率
        tolerance_percent=5,        # VM驗證容差
        steady_params=steady_params,
        plot_file='1_1.csv'           # 繪製*.csv的圖形，設為None不繪圖
    )