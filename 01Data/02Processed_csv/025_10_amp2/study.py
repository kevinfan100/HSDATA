import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def dac_to_voltage(dac_value):
    """將16位元DAC數值轉換為電壓"""
    return (dac_value - 32768) * (20.0 / 65536)

def load_single_test(csv_file):
    """載入CSV測試數據"""
    df = pd.read_csv(csv_file)
    sampling_rate = 100000
    n_samples = len(df)
    
    vm = df[['vm_0', 'vm_1', 'vm_2', 'vm_3', 'vm_4', 'vm_5']].values.T
    da = df[['da_0', 'da_1', 'da_2', 'da_3', 'da_4', 'da_5']].values.T
    
    return {'vm': vm, 'da': da, 'sampling_rate': sampling_rate}

def find_excitation_channel(da_data):
    """找出激勵通道"""
    da_rms = np.sqrt(np.mean(da_data**2, axis=1))
    return np.argmax(da_rms)

def calculate_transfer_function(input_signal, output_signal, fs, target_freq):
    """
    計算傳遞函數 H = Output/Input 在特定頻率
    
    Returns:
    -------
    tuple: (含符號增益, 幅值, 相位度數)
    """
    
    # 1. 去除DC分量
    input_clean = input_signal - np.mean(input_signal)
    output_clean = output_signal - np.mean(output_signal)
    
    # 2. 加Hanning窗減少頻譜洩漏
    window = np.hanning(len(input_clean))
    input_windowed = input_clean * window
    output_windowed = output_clean * window
    
    # 3. FFT計算
    input_fft = fft(input_windowed)
    output_fft = fft(output_windowed)
    freqs = fftfreq(len(input_clean), 1/fs)
    
    # 4. 找目標頻率（5%容差）
    freq_tolerance = 0.03
    freq_mask = (freqs > 0) & (np.abs(freqs - target_freq) <= target_freq * freq_tolerance)
    
    if not np.any(freq_mask):
        return 0, 0, 0  # 找不到目標頻率
    
    # 5. 在容差範圍內找能量最大的頻率點
    freq_indices = np.where(freq_mask)[0]
    target_powers = np.abs(input_fft[freq_indices])**2
    best_idx = freq_indices[np.argmax(target_powers)]
    
    # 6. 計算傳遞函數
    H_complex = output_fft[best_idx] / input_fft[best_idx]
    magnitude = np.abs(H_complex)
    phase_deg = np.angle(H_complex) * 180 / np.pi
    signed_gain = np.real(H_complex)  # 實數部分包含正負號
    
    return signed_gain, magnitude, phase_deg

def process_single_file(csv_file, target_freq):
    """
    處理單個檔案，計算B矩陣的一行
    
    Returns:
    -------
    tuple: (激勵通道, 含符號B矩陣行, 幅值行, 相位行)
    """
    
    # 載入數據
    data = load_single_test(csv_file)
    
    # 轉換DAC為電壓
    da_voltage = dac_to_voltage(data['da'])
    
    # 找激勵通道
    excited_ch = find_excitation_channel(da_voltage)
    input_signal = da_voltage[excited_ch]
    
    print(f"檔案 {csv_file}: 激勵通道 DA_{excited_ch}")
    
    # 計算每個VM通道的傳遞函數
    b_signed_row = np.zeros(6)
    b_magnitude_row = np.zeros(6)
    phase_row = np.zeros(6)
    
    for vm_ch in range(6):
        output_signal = data['vm'][vm_ch]
        
        signed_gain, magnitude, phase_deg = calculate_transfer_function(
            input_signal, output_signal, data['sampling_rate'], target_freq
        )
        
        b_signed_row[vm_ch] = signed_gain
        b_magnitude_row[vm_ch] = magnitude
        phase_row[vm_ch] = phase_deg
        
        print(f"  VM_{vm_ch}: {signed_gain:+.6f} (|H|={magnitude:.6f}, ∠={phase_deg:+6.1f}°)")
    
    return excited_ch, b_signed_row, b_magnitude_row, phase_row

def calculate_b_matrix(csv_files, target_freq):
    """
    從多個CSV檔案計算B矩陣
    
    Parameters:
    ----------
    csv_files : list, 6個CSV檔案路徑
    target_freq : float, 激勵頻率 (Hz)
    
    Returns:
    -------
    tuple: (含符號B矩陣, 幅值矩陣, 相位矩陣)
    """
    
    print(f"計算B矩陣，激勵頻率: {target_freq} Hz")
    print("=" * 50)
    
    B_signed = np.zeros((6, 6))
    B_magnitude = np.zeros((6, 6))
    Phase_matrix = np.zeros((6, 6))
    
    for csv_file in csv_files:
        excited_ch, signed_row, mag_row, phase_row = process_single_file(csv_file, target_freq)
        
        # 將結果填入對應矩陣 (一個檔案對應一列，垂直方向)
        B_signed[:, excited_ch] = signed_row
        B_magnitude[:, excited_ch] = mag_row
        Phase_matrix[:, excited_ch] = phase_row
        print()
    
    # 顯示結果
    print("\nMagnitude Matrix |H|:")
    print("=" * 20)
    print("Rows: Response channels (VM_0 to VM_5)")
    print("Cols: Excitation channels (DA_0 to DA_5) - One file per column")
    print(B_magnitude)
    
    print("\nPhase Matrix (degrees):")
    print("=" * 25)
    print("Rows: Response channels (VM_0 to VM_5)")
    print("Cols: Excitation channels (DA_0 to DA_5) - One file per column")
    print(Phase_matrix)

    return B_signed, B_magnitude, Phase_matrix

def analyze_single_excitation_channel(csv_file, target_freq):
    """
    分析單一激勵通道下所有VM的頻譜響應和傳遞函數
    
    Parameters:
    ----------
    csv_file : str, CSV檔案路徑
    target_freq : float, 目標頻率
    """
    
    # 載入數據
    data = load_single_test(csv_file)
    da_voltage = dac_to_voltage(data['da'])
    
    # 找激勵通道
    excited_ch = find_excitation_channel(da_voltage)
    input_signal = da_voltage[excited_ch]
    fs = data['sampling_rate']
    
    print(f"Analyzing file: {csv_file}")
    print(f"Excitation channel: DA_{excited_ch}")
    print(f"Target frequency: {target_freq} Hz")
    print("=" * 60)
    
    # 預處理輸入信號
    input_clean = input_signal - np.mean(input_signal)
    window = np.hanning(len(input_clean))
    input_windowed = input_clean * window
    input_fft = fft(input_windowed)
    freqs = fftfreq(len(input_clean), 1/fs)
    
    # 只取正頻率
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    input_fft_pos = input_fft[positive_freq_mask]
    
    # === 繪製輸入信號的時域和頻域圖 ===
    fig_input, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
    fig_input.suptitle(f'Input Signal Analysis - DA_{excited_ch} (Target: {target_freq} Hz)', fontsize=14)
    
    # 時域圖
    time_axis = np.arange(len(input_signal)) / fs
    ax_time.plot(time_axis, input_signal)
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Amplitude (V)')
    ax_time.set_title(f'Time Domain - DA_{excited_ch}')
    ax_time.grid(True, alpha=0.3)
    
    # 頻域圖
    input_magnitude = np.abs(input_fft_pos)
    input_power = input_magnitude ** 2
    max_input_idx = np.argmax(input_power)
    max_input_freq = freqs_pos[max_input_idx]
    max_input_mag = input_magnitude[max_input_idx]
    
    ax_freq.loglog(freqs_pos, input_magnitude)
    ax_freq.scatter(max_input_freq, max_input_mag, color='red', s=100, zorder=5, 
                    label=f'Max Energy: {max_input_freq:.1f} Hz')
    ax_freq.axvline(target_freq, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Target: {target_freq} Hz')
    ax_freq.set_xlabel('Frequency (Hz)')
    ax_freq.set_ylabel('|Input(f)| Magnitude')
    ax_freq.set_title(f'Frequency Domain - DA_{excited_ch}')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input signal max energy at: {max_input_freq:.2f} Hz (Magnitude: {max_input_mag:.6f})")
    print("-" * 60)
    
    # 建立圖表 - 6個VM通道，每個2個子圖（響應頻譜+傳遞函數頻譜）
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))
    fig.suptitle(f'Frequency Analysis - DA_{excited_ch} Excitation (Target: {target_freq} Hz)', fontsize=14)
    
    max_info = []  # 儲存每個通道的最大能量點資訊
    
    for vm_ch in range(6):
        output_signal = data['vm'][vm_ch]
        
        # 預處理輸出信號
        output_clean = output_signal - np.mean(output_signal)
        output_windowed = output_clean * window
        output_fft = fft(output_windowed)
        output_fft_pos = output_fft[positive_freq_mask]
        
        # 計算傳遞函數
        valid_mask = np.abs(input_fft_pos) > 1e-10
        freqs_valid = freqs_pos[valid_mask]
        H_complex = np.zeros_like(input_fft_pos, dtype=complex)
        H_complex[valid_mask] = output_fft_pos[valid_mask] / input_fft_pos[valid_mask]
        
        # VM響應頻譜 (左側子圖)
        vm_magnitude = np.abs(output_fft_pos)
        vm_power = vm_magnitude ** 2
        max_vm_idx = np.argmax(vm_power)
        max_vm_freq = freqs_pos[max_vm_idx]
        max_vm_mag = vm_magnitude[max_vm_idx]
        
        axes[vm_ch, 0].loglog(freqs_pos, vm_magnitude)
        axes[vm_ch, 0].scatter(max_vm_freq, max_vm_mag, color='red', s=50, zorder=5)
        axes[vm_ch, 0].axvline(target_freq, color='orange', linestyle='--', alpha=0.7)
        axes[vm_ch, 0].set_title(f'VM_{vm_ch} Response Spectrum')
        axes[vm_ch, 0].set_ylabel('|VM(f)| Magnitude')
        axes[vm_ch, 0].grid(True, alpha=0.3)
        
        # 傳遞函數頻譜 (右側子圖)
        H_magnitude = np.abs(H_complex)
        H_power = H_magnitude[valid_mask] ** 2
        max_h_idx = np.argmax(H_power)
        max_h_freq = freqs_valid[max_h_idx]
        max_h_mag = H_magnitude[valid_mask][max_h_idx]
        max_h_phase = np.angle(H_complex[valid_mask][max_h_idx]) * 180 / np.pi
        
        axes[vm_ch, 1].loglog(freqs_valid, H_magnitude[valid_mask])
        axes[vm_ch, 1].scatter(max_h_freq, max_h_mag, color='red', s=50, zorder=5)
        axes[vm_ch, 1].axvline(target_freq, color='orange', linestyle='--', alpha=0.7)
        axes[vm_ch, 1].set_title(f'H_{vm_ch} Transfer Function Spectrum')
        axes[vm_ch, 1].set_ylabel('|H(f)| Magnitude')
        axes[vm_ch, 1].grid(True, alpha=0.3)
        
        # 最後一行添加x軸標籤
        if vm_ch == 5:
            axes[vm_ch, 0].set_xlabel('Frequency (Hz)')
            axes[vm_ch, 1].set_xlabel('Frequency (Hz)')
        
        # 儲存最大能量點資訊
        max_info.append({
            'vm_channel': vm_ch,
            'vm_max_freq': max_vm_freq,
            'vm_max_mag': max_vm_mag,
            'h_max_freq': max_h_freq,
            'h_max_mag': max_h_mag,
            'h_max_phase': max_h_phase
        })
        
        print(f"VM_{vm_ch}: Response max@{max_vm_freq:.1f}Hz, H max@{max_h_freq:.1f}Hz (|H|={max_h_mag:.6f}, ∠={max_h_phase:.1f}°)")
    
    plt.tight_layout()
    plt.show()
    
    return max_info

# === 使用範例 ===
if __name__ == "__main__":
    # 你的6個CSV檔案
    csv_files = [
        '0.csv',  # DA_0激勵
        '1.csv',  # DA_1激勵
        '2.csv',  # DA_2激勵
        '3.csv',  # DA_3激勵
        '4.csv',  # DA_4激勵
        '5.csv',  # DA_5激勵
    ]
    
    # 計算B矩陣
    B_signed, B_magnitude, Phase_matrix = calculate_b_matrix(csv_files, target_freq=10)
    
    # === 單通道頻域分析範例 ===
    print("\n" + "="*60)
    print("單通道頻域分析範例")
    print("="*60)
    
    # 選擇要分析的檔案（可以改變索引來分析不同的激勵通道）
    analysis_file_index = 0  # 0~5對應不同的激勵通道
    analysis_file = csv_files[analysis_file_index]
    
    # 分析單一激勵通道下所有VM的響應頻譜和傳遞函數頻譜
    # max_info = analyze_single_excitation_channel(analysis_file, target_freq=10)
    
    # === 如需單獨進行頻域分析，可直接調用 ===
    analyze_single_excitation_channel('0.csv', target_freq=10)