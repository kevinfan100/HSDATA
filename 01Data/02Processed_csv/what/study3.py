import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def dac_to_voltage(dac):  # 16-bit DAC to voltage
    return (dac - 32768) * (20.0 / 65536)

def load_data(file):
    df = pd.read_csv(file)
    vm = df[[f'vm_{i}' for i in range(6)]].values.T
    da = df[[f'da_{i}' for i in range(6)]].values.T
    return {'vm': vm, 'da': da}

def find_excitation_channel(da):
    return np.argmax(np.sqrt(np.mean(da**2, axis=1)))

def detect_steady_state(file, freq, sampling_rate=100000, start_period=1, 
                        consecutive_periods=3, check_points=10, threshold=1e-3):
    df = pd.read_csv(file)
    valid = np.ones(len(df), dtype=bool)
    valid[::10000] = False
    df = df[valid].reset_index(drop=True)
    vm = df[[f'vm_{i}' for i in range(6)]].values.T
    original_idx = np.where(valid)[0]

    per_samples = int(sampling_rate / freq)
    max_periods = len(df) // per_samples
    check_pos = np.linspace(0, per_samples-1, check_points, dtype=int)
    results = []

    for ch in range(6):
        sig = vm[ch]
        for p in range(start_period, max_periods - consecutive_periods):
            if all(np.max(np.abs(sig[(p+i)*per_samples + check_pos] - sig[(p+i+1)*per_samples + check_pos])) < threshold 
                   for i in range(consecutive_periods)):
                results.append(p)
                break

    if results:
        period = max(results)
        return {'period': period, 'index': original_idx[period * per_samples], 'max_periods': max_periods}
    return None

def compute_tf(input_sig, output_sig, fs, target_freq, tol_pct=5):
    input_sig -= np.mean(input_sig)
    output_sig -= np.mean(output_sig)

    U = fft(input_sig)
    Y = fft(output_sig)
    freqs = fftfreq(len(U), 1/fs)
    pos = freqs > 0
    freqs = freqs[pos]
    U, Y = U[pos], Y[pos]

    vm_power = np.abs(Y)**2
    max_freq = freqs[np.argmax(vm_power)]
    valid = abs(max_freq - target_freq) <= target_freq * tol_pct / 100

    idx = np.argmin(np.abs(freqs - target_freq))
    H = Y[idx] / U[idx]
    return np.real(H), np.abs(H), np.angle(H, deg=True), valid

def plot_frequency_analysis(file, freq, steady, tol_pct=5):
    data = load_data(file)
    da_v = dac_to_voltage(data['da'])
    ch = find_excitation_channel(da_v)

    ps = int(100000 / freq)
    ap = steady['max_periods'] - steady['period']
    s_idx = steady['index']
    e_idx = s_idx + ap * ps
    u = da_v[ch][s_idx:e_idx]

    u_clean = u - np.mean(u)
    freqs = fftfreq(len(u), 1/100000)
    pos = freqs > 0
    freqs, U = freqs[pos], fft(u_clean)[pos]

    # Input FFT plot
    plt.figure(figsize=(12, 4))
    plt.title(f"Input DA_{ch} Spectrum")
    plt.loglog(freqs, np.abs(U))
    plt.axvline(freq, color='orange', linestyle='--', label=f'Target {freq} Hz')
    plt.grid(True)
    plt.legend()
    plt.show()

    # VM & TF plots
    fig, axs = plt.subplots(6, 2, figsize=(14, 18))
    for i in range(6):
        y = data['vm'][i][s_idx:e_idx] - np.mean(data['vm'][i][s_idx:e_idx])
        Y = fft(y)[pos]
        H = Y / U

        axs[i, 0].loglog(freqs, np.abs(Y))
        axs[i, 0].set_title(f'VM_{i} Spectrum')
        axs[i, 1].loglog(freqs, np.abs(H))
        axs[i, 1].set_title(f'H_{i} Transfer Function')
        for ax in axs[i]:
            ax.axvline(freq, color='orange', linestyle='--')
            ax.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_b_matrix(csv_files, target_freq, tol_pct=5, steady_params=None, plot_file=None):
    if steady_params is None:
        steady_params = {'consecutive_periods': 3, 'check_points': 10, 'threshold': 1e-3}

    B_signed, B_mag, B_phase = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))
    validation = np.zeros((6,6), dtype=bool)
    steady_all = []

    for f in csv_files:
        info = detect_steady_state(f, target_freq, **steady_params)
        if not info:
            print(f"[Error] No steady state in {f}")
            return None
        steady_all.append(info)

    uni_period = max([i['period'] for i in steady_all])
    print(f"[Info] Unified Steady Period: {uni_period}")

    ps = int(100000 / target_freq)
    for i, f in enumerate(csv_files):
        data = load_data(f)
        da_v = dac_to_voltage(data['da'])
        ch = find_excitation_channel(da_v)
        info = steady_all[i]

        ap = info['max_periods'] - uni_period
        s_idx = info['index'] + (uni_period - info['period']) * ps
        e_idx = s_idx + ap * ps
        u = da_v[ch][s_idx:e_idx]

        if f == plot_file:
            plot_frequency_analysis(f, target_freq, {
                'period': uni_period, 'index': s_idx, 'max_periods': info['max_periods']
            }, tol_pct)

        for vm in range(6):
            y = data['vm'][vm][s_idx:e_idx]
            s, m, p, ok = compute_tf(u, y, 100000, target_freq, tol_pct)
            B_signed[vm, ch], B_mag[vm, ch], B_phase[vm, ch], validation[vm, ch] = s, m, p, ok

    print("\n[Signed Gain Matrix] VM row × DA col")
    for i in range(6):
        row = f"VM_{i} " + " ".join(f"{B_signed[i,j]:+7.4f}{'✓' if validation[i,j] else '✗'}" for j in range(6))
        print(row)

    passed = np.sum(validation)
    print(f"\n[Summary] {passed}/36 Valid ({passed/36*100:.1f}%)")
    return {'B_signed': B_signed, 'B_magnitude': B_mag, 'B_phase': B_phase, 'validation': validation}

# === Usage Entry Point ===
if __name__ == "__main__":
    csv_files = ['0.csv', '1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
    target_freq = 10               # Hz
    tolerance_pct = 5             # ±%
    plot_file = '0.csv'           # or None
    steady_params = {
        'consecutive_periods': 2,
        'check_points': 5,
        'threshold': 2e-3
    }

    result = calculate_b_matrix(
        csv_files=csv_files,
        target_freq=target_freq,
        tol_pct=tolerance_pct,
        steady_params=steady_params,
        plot_file=plot_file
    )
