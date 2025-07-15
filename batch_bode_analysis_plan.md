# 批量Bode分析完整計劃書

## 1. 專案目標

基於現有的Bode分析架構，開發一個自動化批量處理系統，能夠：
- 自動處理 `0715_ndc_data/` 資料夾中的所有 `.dat` 檔案
- 生成完整的六通道頻率響應波德圖
- 實現高度自動化的數據處理流程

## 2. 需求規格

### 2.1 自動化要求
1. **自動輸入信號檢測**：選擇能量最大的VD通道作為輸入
2. **自動週期性檢測**：對應VM通道的週期性驗證
3. **自動主頻驗證**：確保數據可信度
4. **自動FFT配置**：解析度優於1 Hz
5. **批量處理**：處理所有頻率檔案

### 2.2 輸出要求
1. **互動式波德圖**：使用plotly實現
2. **對數頻率軸**：標準波德圖格式
3. **線性幅度**：0-1範圍，不使用dB
4. **相位度數**：標準角度單位
5. **不保存圖片**：只顯示，不產生檔案

### 2.3 終端輸出要求
1. **精簡信息**：只顯示關鍵處理狀態
2. **進度追蹤**：清楚顯示處理進度
3. **錯誤提醒**：跳過問題檔案並提醒

## 3. 技術架構

### 3.1 核心模組設計

```
batch_bode_analyzer.py
├── 數據讀取模組
│   ├── extract_frequency_from_filename()
│   └── read_data_file()
├── 自動檢測模組
│   ├── auto_detect_input_channel()
│   └── validate_data_quality()
├── 分析處理模組
│   ├── analyze_single_frequency()
│   ├── auto_configure_fft_points()
│   └── apply_periodicity_to_channels()
├── 批量處理模組
│   └── batch_bode_analysis()
└── 視覺化模組
    └── plot_interactive_bode()
```

### 3.2 數據流程

```
檔案列表 → 頻率提取 → 數據讀取 → 輸入檢測 → 品質驗證 → 
週期分析 → FFT處理 → FRF計算 → 結果收集 → 波德圖生成
```

## 4. 詳細實現計劃

### 4.1 自動輸入信號檢測

**策略**：選擇目標頻率處能量最大的VD通道

```python
def auto_detect_input_channel(records, target_freq, sampling_freq):
    """自動檢測能量最大的輸入信號通道"""
    best_channel = None
    max_energy = 0
    
    for vd_channel in range(6):
        signal = np.array([record['vd'][vd_channel] for record in records])
        fft_result = np.fft.fft(signal)
        freq_axis = np.fft.fftfreq(len(signal), 1/sampling_freq)
        
        # 尋找目標頻率附近的峰值
        target_bin = int(target_freq / (sampling_freq / len(signal)))
        if target_bin < len(fft_result) // 2:
            energy = np.abs(fft_result[target_bin])
            if energy > max_energy:
                max_energy = energy
                best_channel = vd_channel
    
    return best_channel
```

### 4.2 數據品質驗證

**週期性檢測標準**（沿用現有邏輯）：
- 變異係數 < 20%
- 至少2個自相關峰值
- 峰值突出度 > std(autocorr) * 0.1

**主頻驗證標準**（沿用現有邏輯）：
- 主頻與目標頻率差異 < 2 Hz
- 主頻能量 > 最大能量的1%

```python
def validate_data_quality(records, input_channel, target_freq):
    """驗證數據品質"""
    # 1. 週期性檢測
    vm_signal = np.array([record['vm'][input_channel] for record in records])
    _, _, _, is_periodic = detect_periodicity(vm_signal, target_freq, 100000)
    
    if not is_periodic:
        return False
    
    # 2. 主頻驗證
    fft_result = np.fft.fft(vm_signal)
    freq_axis = np.fft.fftfreq(len(vm_signal), 1/100000)
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
```

### 4.3 自動FFT點數配置

```python
def auto_configure_fft_points(signal_length, target_freq, sampling_freq):
    """自動配置FFT點數以達到所需解析度"""
    min_resolution = 0.5  # 比1 Hz更好
    min_points = int(sampling_freq / min_resolution)
    
    if signal_length >= min_points:
        return signal_length
    else:
        # 使用零填充達到所需解析度
        return min_points
```

### 4.4 單頻率分析

```python
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
```

### 4.5 批量處理主函數

```python
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
```

### 4.6 互動式波德圖

```python
def plot_interactive_bode(results):
    """互動式波德圖"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
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
    
    fig.show()
```

## 5. 輔助函數

### 5.1 頻率提取

```python
def extract_frequency_from_filename(filename):
    """從檔名提取頻率"""
    # 例如: ndc100.dat -> 100
    import re
    match = re.search(r'ndc(\d+)\.dat', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"無法從檔名 {filename} 提取頻率")
```

### 5.2 週期性參數應用

```python
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
```

## 6. 主程式流程

```python
def main():
    """主程式"""
    print("批量Bode分析系統")
    print("=" * 50)
    
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
```

## 7. 預期輸出範例

```
批量Bode分析系統
==================================================
 發現 6 個數據檔案

 [1/6] 處理 ndc1.dat (1 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 [2/6] 處理 ndc10.dat (10 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 [3/6] 處理 ndc50.dat (50 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 [4/6] 處理 ndc100.dat (100 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 [5/6] 處理 ndc500.dat (500 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 [6/6] 處理 ndc1000.dat (1000 Hz)
   檢測到輸入通道: VD4
   ✓ 分析完成

 批量分析完成
   成功處理: 6/6 檔案

 生成互動式波德圖
   有效通道: VM0, VM1, VM2, VM3, VM4, VM5

 分析完成
```

## 8. 技術要點

### 8.1 沿用現有架構
- 完全基於現有的 `test_bode_analysis.py` 架構
- 保持數據處理邏輯的一致性
- 重用所有核心函數

### 8.2 新增功能
- 自動輸入檢測
- 批量處理能力
- 互動式視覺化
- 精簡終端輸出

### 8.3 錯誤處理
- 優雅的錯誤處理機制
- 保持處理流程的連續性
- 詳細的錯誤信息

### 8.4 性能優化
- 自動FFT點數配置
- 避免不必要的計算
- 高效的數據結構

## 9. 依賴項

```python
# requirements.txt 新增
plotly>=5.0.0
```

## 10. 驗收標準

1. **自動化程度**：無需手動干預即可完成所有檔案處理
2. **數據品質**：所有處理的檔案都通過品質驗證
3. **視覺化效果**：生成清晰的互動式波德圖
4. **輸出效率**：終端輸出精簡且信息完整
5. **錯誤處理**：優雅處理各種異常情況

---

**此計劃書涵蓋了完整的實現細節，確保與現有Bode分析架構的完全兼容性，同時實現所有自動化需求。** 