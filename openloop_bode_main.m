function openloop_bode_main(varargin)
% 開環波德圖分析主函數 - 修復版本
% 
% 使用方法:
%   openloop_bode_main()                     % 使用預設資料夾
%   openloop_bode_main('custom_folder/')     % 使用自訂資料夾

fprintf('=== 開環波德圖分析系統（修復版）===\n');

% 配置參數
SAMPLING_RATE = 100000;
VM_FREQ_TOLERANCE = 0.05;
MIN_DA_THRESHOLD = 1e-10;
DATA_FOLDER = '01Data\02Processed_csv\openloop_Cali_P5';
CONSECUTIVE_PERIODS = 2;
CHECK_POINTS = 5;
STABILITY_THRESHOLD = 2e-3;
START_PERIOD = 1;
CHANNEL_COLORS = ['k','b','g','r','m','c'];  % 黑藍綠紅紫淺藍
DISPLAY_CHANNELS = [1,2,3,4,5,6];          % 控制要顯示的通道，可方便調整

% 解析輸入參數
if nargin > 0
    csv_folder = varargin{1};
else
    csv_folder = DATA_FOLDER;
end

fprintf('使用資料夾: %s\n', csv_folder);

% 檢查資料夾是否存在
if ~exist(csv_folder, 'dir')
    error('數據資料夾不存在: %s', csv_folder);
end

% 獲取所有CSV檔案
csv_files = dir(fullfile(csv_folder, '*.csv'));

if isempty(csv_files)
    error('資料夾中沒有找到CSV檔案: %s', csv_folder);
end

fprintf('找到 %d 個CSV檔案\n', length(csv_files));

% 初始化結果陣列
frequencies = [];
magnitudes_db = zeros(6, 0);
phases = zeros(6, 0);
excitation_channels = []; % 記錄每個頻率點的激勵通道

% 處理每個CSV檔案
for i = 1:length(csv_files)
    csv_file = csv_files(i);
    file_path = fullfile(csv_folder, csv_file.name);
    
    fprintf('\n[%d/%d] 處理檔案: %s (%.1f MB)\n', ...
        i, length(csv_files), csv_file.name, csv_file.bytes/1024/1024);
    
    try
        % 讀取數據
        fprintf('  步驟1: 讀取CSV數據...\n');
        raw_data = readtable(file_path);
        data_length = height(raw_data);
        
        % 初始化數據矩陣
        vm_data = zeros(6, data_length);
        da_data = zeros(6, data_length);
        
        % 提取各通道數據
        for ch = 1:6
            vm_col = sprintf('vm_%d', ch-1);
            da_col = sprintf('da_%d', ch-1);
            
            if ismember(vm_col, raw_data.Properties.VariableNames)
                vm_data(ch, :) = raw_data.(vm_col);
            end
            
            if ismember(da_col, raw_data.Properties.VariableNames)
                da_data(ch, :) = raw_data.(da_col);
            end
        end
        
        % 清理數據（排除每10000個樣本點）
        fprintf('  步驟2: 清理數據...\n');
        exclude_indices = 1:10000:data_length;
        valid_mask = true(1, data_length);
        valid_mask(exclude_indices) = false;
        
        vm_clean = vm_data(:, valid_mask);
        da_clean = da_data(:, valid_mask);
        
        % 轉換DA為電壓
        da_volt = (da_clean - 32768) * (20.0 / 65536);
        
        % 檢測激勵通道和頻率
        fprintf('  步驟3: 檢測激勵通道和頻率...\n');
        [excite_ch, excite_freq] = detect_excitation(da_volt, SAMPLING_RATE);
        fprintf('    激勵通道: DA%d, 頻率: %.1f Hz\n', excite_ch, excite_freq);
        
        % 穩態檢測
        fprintf('  步驟4: 穩態檢測...\n');
        steady_info = detect_steady_state(vm_clean(1, :), excite_freq, SAMPLING_RATE);
        
        if isempty(steady_info)
            fprintf('  ✗ 穩態檢測失敗，跳過此檔案\n');
            continue;
        end
        
        fprintf('    穩態起始點: 第%d個週期\n', steady_info.period);
        
        % FFT分析
        fprintf('  步驟5: FFT分析...\n');
        current_magnitudes_db = zeros(6, 1);
        current_phases = zeros(6, 1);
        
        for ch = 1:6
            % 提取穩態後的完整週期數據
            vm_signal = vm_clean(ch, :);
            da_signal = da_volt(excite_ch, :);
            
            % 計算週期長度
            period_samples = round(SAMPLING_RATE / excite_freq);
            steady_start = steady_info.index;
            available_length = length(vm_signal) - steady_start + 1;
            available_periods = floor(available_length / period_samples);
            
            if available_periods < 1
                fprintf('    警告: CH%d 數據不足一個完整週期\n', ch);
                current_magnitudes_db(ch) = -Inf;
                current_phases(ch) = 0;
                continue;
            end
            
            % 提取數據
            end_index = steady_start + available_periods * period_samples - 1;
            vm_period_data = vm_signal(steady_start:end_index);
            da_period_data = da_signal(steady_start:end_index);
            
            % FFT
            vm_fft = fft(vm_period_data);
            da_fft = fft(da_period_data);
            
            % 找到目標頻率的bin
            N = length(vm_period_data);
            freq_axis = (0:N-1) * SAMPLING_RATE / N;
            freq_resolution = freq_axis(2) - freq_axis(1);
            target_bin = round(excite_freq / freq_resolution) + 1;
            
            % 計算傳遞函數
            vm_complex = vm_fft(target_bin);
            da_complex = da_fft(target_bin);
            
            if abs(da_complex) > MIN_DA_THRESHOLD
                transfer_function = vm_complex / da_complex;
                magnitude_linear = abs(transfer_function);
                current_magnitudes_db(ch) = 20 * log10(magnitude_linear);
                current_phases(ch) = angle(transfer_function) * 180 / pi;
            else
                current_magnitudes_db(ch) = -Inf;
                current_phases(ch) = 0;
            end
        end
        
        % 從檔案名提取標稱頻率
        nominal_freq = extract_nominal_frequency(csv_file.name);
        if ~isempty(nominal_freq)
            display_freq = nominal_freq;  % 使用標稱頻率顯示
        else
            display_freq = excite_freq;   % 回退到檢測頻率
        end
        
        % 添加到結果
        frequencies(end+1) = display_freq;
        magnitudes_db(:, end+1) = current_magnitudes_db;
        phases(:, end+1) = current_phases;
        excitation_channels(end+1) = excite_ch; % 記錄激勵通道
        
        fprintf('  ✓ 分析完成：頻率 %.1f Hz\n', excite_freq);
        
    catch ME
        fprintf('  ✗ 處理失敗: %s\n', ME.message);
        continue;
    end
end

% 排序結果
if ~isempty(frequencies)
    [frequencies, sort_idx] = sort(frequencies);
    magnitudes_db = magnitudes_db(:, sort_idx);
    phases = phases(:, sort_idx);
    excitation_channels = excitation_channels(sort_idx);
    
    % 正規化大小數據 - 每個通道以最低頻為基準
    fprintf('\n正規化大小數據...\n');
    magnitudes_db_normalized = normalize_magnitudes(magnitudes_db, frequencies);
    
    fprintf('\n=== 分析完成 ===\n');
    fprintf('成功處理 %d 個頻率點\n', length(frequencies));
    fprintf('頻率範圍: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));
    
    % 繪製波德圖
    fprintf('\nGenerating Bode plots...\n');
    plot_bode_results(frequencies, magnitudes_db_normalized, phases, CHANNEL_COLORS, magnitudes_db, excitation_channels, DISPLAY_CHANNELS);
    
    % 保存到工作空間
    assignin('base', 'openloop_frequencies', frequencies);
    assignin('base', 'openloop_magnitudes_db_original', magnitudes_db);
    assignin('base', 'openloop_magnitudes_db_normalized', magnitudes_db_normalized);
    assignin('base', 'openloop_phases', phases);
    fprintf('結果已保存到工作空間變數\n');
else
    fprintf('\n沒有成功處理任何檔案\n');
end

end

function nominal_freq = extract_nominal_frequency(filename)
% 從檔案名提取標稱頻率
% 例如: P5_100.csv -> 100, P5_0.1.csv -> 0.1

nominal_freq = [];

% 使用正則表達式提取數字
pattern = '_([0-9]*\.?[0-9]+)\.csv';
match = regexp(filename, pattern, 'tokens');

if ~isempty(match)
    freq_str = match{1}{1};
    nominal_freq = str2double(freq_str);
    
    % 確認提取的頻率是合理的（0.01 Hz 到 10000 Hz）
    if isnan(nominal_freq) || nominal_freq < 0.01 || nominal_freq > 10000
        nominal_freq = [];
    end
end
end

function magnitudes_normalized = normalize_magnitudes(magnitudes_db, frequencies)
% 正規化大小數據 - 每個通道以最低頻為基準
% 輸入: magnitudes_db (6 x N), frequencies (1 x N)
% 輸出: magnitudes_normalized (6 x N)

if isempty(magnitudes_db) || isempty(frequencies)
    magnitudes_normalized = magnitudes_db;
    return;
end

% 找到最低頻率的索引
[~, min_freq_idx] = min(frequencies);

magnitudes_normalized = zeros(size(magnitudes_db));

fprintf('正規化參考點：%.2f Hz\n', frequencies(min_freq_idx));

% 對每個通道進行正規化
for ch = 1:6
    reference_value = magnitudes_db(ch, min_freq_idx);
    
    % 檢查參考值是否有效
    if isfinite(reference_value)
        % 正規化：每個頻率的dB值減去最低頻的dB值
        magnitudes_normalized(ch, :) = magnitudes_db(ch, :) - reference_value;
        
        fprintf('  CH%d: 參考值 = %.2f dB\n', ch, reference_value);
    else
        % 如果參考值無效，保持原值
        magnitudes_normalized(ch, :) = magnitudes_db(ch, :);
        fprintf('  CH%d: 參考值無效，保持原始數據\n', ch);
    end
end

fprintf('正規化完成！最低頻率處所有通道都成為 0 dB 參考點\n');
end

function [excite_ch, excite_freq] = detect_excitation(da_voltage, sampling_rate)
% 檢測激勵通道和頻率
best_channel = 0;
max_energy = 0;
best_freq = 0;

for ch = 1:6
    signal_data = da_voltage(ch, :);
    signal_energy = sqrt(mean(signal_data.^2));
    
    if signal_energy > 0.1
        N = length(signal_data);
        fft_result = fft(signal_data);
        freq_axis = (0:N-1) * sampling_rate / N;
        
        positive_freqs = freq_axis(2:floor(N/2));
        positive_fft = abs(fft_result(2:floor(N/2)));
        
        [max_amplitude, max_idx] = max(positive_fft);
        dominant_freq = positive_freqs(max_idx);
        
        if max_amplitude > max_energy
            max_energy = max_amplitude;
            best_channel = ch;
            best_freq = dominant_freq;
        end
    end
end

if best_channel == 0
    error('未檢測到有效的激勵通道');
end

excite_ch = best_channel;
excite_freq = best_freq;
end

function steady_info = detect_steady_state(vm_signal, target_freq, sampling_rate)
% 簡化的穩態檢測
period_samples = round(sampling_rate / target_freq);
max_periods = floor(length(vm_signal) / period_samples);

if max_periods < 5
    steady_info = [];
    return;
end

% 使用後半段作為穩態
steady_period = max(1, max_periods - 3);
steady_info = struct('period', steady_period, 'index', steady_period * period_samples + 1);
end

function plot_bode_results(frequencies, magnitudes_db, phases, colors, original_magnitudes_db, excitation_channels, display_channels)
% 繪製波德圖（幅度和相位）

% 處理相位數據
phases_processed = phases;

% 對每個頻率點處理相位
for freq_idx = 1:length(frequencies)
    excite_ch = excitation_channels(freq_idx);
    
    if ismember(excite_ch, [1, 3, 6])
        % 如果激勵通道是1、3、6，只對激勵通道減去180度
        phases_processed(excite_ch, freq_idx) = phases_processed(excite_ch, freq_idx) - 180;
    else
        % 如果激勵通道是2、4、5，對非激勵通道減去180度
        for ch = 1:6
            if ch ~= excite_ch
                phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) - 180;
            end
        end
    end
end

% 計算正規化除得值（最低頻率處的原始值）
[~, min_freq_idx] = min(frequencies);
normalization_values_db = original_magnitudes_db(:, min_freq_idx);
normalization_values_linear = 10.^(normalization_values_db/20);

% 創建兩個子圖
figure('Name', 'Open-loop Bode Plot', 'Position', [100, 100, 900, 720]);

% 上方子圖：幅度響應
subplot(2,1,1);
hold on;
for ch = display_channels
    norm_val_linear = normalization_values_linear(ch);
    if isfinite(norm_val_linear)
        legend_text = sprintf('P%d (%.3f)', ch, norm_val_linear);
    else
        legend_text = sprintf('P%d (N/A)', ch);
    end
    
    % === P4通道特殊處理：排除最後一個頻率點 ===
    if ch == 4 && length(frequencies) > 1
        % P4通道排除最後一個點
        freq_plot = frequencies(1:end-1);
        mag_plot = magnitudes_db(ch, 1:end-1);
        fprintf('P4通道排除最後一個頻率點: %.2f Hz\n', frequencies(end));
    else
        % 其他通道正常繪製
        freq_plot = frequencies;
        mag_plot = magnitudes_db(ch, :);
    end
    % === P4通道特殊處理結束 ===
    
    semilogx(freq_plot, mag_plot, ...
            'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
            'MarkerSize', 10, 'DisplayName', legend_text);
end

% 添加二階系統理論響應
if ~isempty(frequencies)
    wn_squared = 1.4848e7;
    two_zeta_wn = 8.1877e3;
    
    omega = 2 * pi * frequencies;
    s = 1j * omega;
    
    H_s = wn_squared ./ (s.^2 + two_zeta_wn * s + wn_squared);
    H_magnitude_db = 20 * log10(abs(H_s));
    
    semilogx(frequencies, H_magnitude_db, '-', ...
            'Color', 'k', 'LineWidth', 3, ...
            'DisplayName', 'Model');
end

xlabel('Frequency (Hz)', 'FontWeight', 'bold', 'FontSize', 40);
ylabel('Magnitude (dB)', 'FontWeight', 'bold', 'FontSize', 40);
legend('Location', 'southwest', 'FontWeight', 'bold','FontSize', 24);

% 設置對數軸
set(gca, 'XScale', 'log');
if ~isempty(frequencies)
    freq_max = max(frequencies);
    xlim([0.1, freq_max]);
    
    log_min = -1;
    log_max = ceil(log10(freq_max));
    log_ticks = 10.^(log_min+1:log_max);
    set(gca, 'XTick', log_ticks);
end

% 設置Y軸範圍
if ~isempty(magnitudes_db)
    y_min = min(magnitudes_db(:));
    if isfinite(y_min)
        ylim([y_min - 5, 2]);
    end
end

set(gca, 'FontWeight', 'bold', 'FontSize', 24, 'LineWidth', 2);
% 加粗X&Y軸和刻度
ax = gca;
ax.XAxis.LineWidth = 3;
ax.YAxis.LineWidth = 3;
ax.XAxis.FontWeight = 'bold';
ax.YAxis.FontWeight = 'bold';
% 加粗外框
box on;
ax.Box = 'on';
ax.BoxStyle = 'full';

% 下方子圖：相位響應
subplot(2,1,2);
hold on;
for ch = display_channels
    norm_val_linear = normalization_values_linear(ch);
    if isfinite(norm_val_linear)
        legend_text = sprintf('P%d (%.3f)', ch, norm_val_linear);
    else
        legend_text = sprintf('P%d (N/A)', ch);
    end
    
    % === P4通道特殊處理：排除最後一個頻率點 ===
    if ch == 4 && length(frequencies) > 1
        % P4通道排除最後一個點
        freq_plot = frequencies(1:end-1);
        phase_plot = phases_processed(ch, 1:end-1);
    else
        % 其他通道正常繪製
        freq_plot = frequencies;
        phase_plot = phases_processed(ch, :);
    end
    % === P4通道特殊處理結束 ===
    
    semilogx(freq_plot, phase_plot, ...
            'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
            'MarkerSize', 10, 'DisplayName', legend_text);
end

% 添加二階系統理論相位響應
if ~isempty(frequencies)
    omega = 2 * pi * frequencies;
    s = 1j * omega;
    
    H_s = wn_squared ./ (s.^2 + two_zeta_wn * s + wn_squared);
    H_phase_deg = angle(H_s) * 180 / pi;
    
    semilogx(frequencies, H_phase_deg, '-', ...
            'Color', 'k', 'LineWidth', 3, ...
            'DisplayName', 'Model');
end

xlabel('Frequency (Hz)', 'FontWeight', 'bold', 'FontSize', 40);
ylabel('Phase (deg)', 'FontWeight', 'bold', 'FontSize', 40);

% 設置對數軸
set(gca, 'XScale', 'log');
if ~isempty(frequencies)
    xlim([0.1, freq_max]);
    set(gca, 'XTick', log_ticks);
end

set(gca, 'FontWeight', 'bold', 'FontSize', 24, 'LineWidth', 2);
% 加粗X&Y軸和刻度
ax = gca;
ax.XAxis.LineWidth = 3;
ax.YAxis.LineWidth = 3;
ax.XAxis.FontWeight = 'bold';
ax.YAxis.FontWeight = 'bold';
% 加粗外框
box on;
ax.Box = 'on';
ax.BoxStyle = 'full';

fprintf('Bode plots completed\n');
fprintf('Frequency range: %.2f - %.2f Hz\n', min(frequencies), max(frequencies));
fprintf('Frequency points: %d\n', length(frequencies));
end