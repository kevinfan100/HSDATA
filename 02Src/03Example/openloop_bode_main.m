function openloop_bode_main(varargin)

% 清空命令窗口
clc;

% 配置參數
SAMPLING_RATE = 100000;              % 採樣率 (Hz)
MIN_DA_THRESHOLD = 1e-10;           % DA信號最小閾值
DATA_FOLDER = 'C:\Users\PME406_01\Desktop\code\HSDATA\01Data\02Processed_csv\openloop_Cali_P5';

% 穩態檢測參數
STABILITY_THRESHOLD = 3e-3;          % 穩定性閾值 (1mV)
CONSECUTIVE_PERIODS = 3;             % 需要連續穩定的週期數
CHECK_POINTS = 100;                    % 每週期的檢查點數
START_PERIOD = 1;                    % 開始檢測的週期

% 顯示設定
CHANNEL_COLORS = ['k','b','g','r','m','c'];  % 黑藍綠紅紫淺藍
DISPLAY_CHANNELS = [1,2,3,4,5,6];          % 控制要顯示的通道，可方便調整

% 穩態波形疊圖設定
PLOT_STEADY_STATE = true;           % 是否顯示穩態疊圖
PLOT_CHANNEL = 5;                   % 0=所有通道，1-6=特定通道
PLOT_FREQUENCY_LIST = [1, 10];           % 只對特定頻率顯示（空=全部）例如：[50, 100, 500]

% FFT分析模式
FFT_MODE = 'averaged';               % 'full': 完整週期FFT, 'averaged': 週期平均後FFT, 'both': 同時計算並比較
COMPARE_FFT_METHODS = false;         % 是否比較兩種FFT方法的結果

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
        
        % 修復數據
        fprintf('  步驟2: ...\n');
        bad_indices = 1:10000:data_length;
        num_bad_points = length(bad_indices);

        % 'linear' - 線性插值（簡單快速）
        % 'spline' - 樣條插值（最平滑，適合週期性信號）
        % 'pchip' - 分段三次Hermite插值（保形，避免過衝）
        % 'makima' - 修正Akima插值（平衡平滑度和穩定性）
        interpolation_method = 'spline';  % 可修改此處來對比不同方法

        fprintf('    使用插值方法: %s\n', interpolation_method);

        % 複製原始數據（不改變大小）
        vm_clean = vm_data;
        da_clean = da_data;

        % 保存原始壞點值用於對比
        vm_bad_original = vm_data(:, bad_indices);
        da_bad_original = da_data(:, bad_indices);

        % 使用高級插值方法
        if num_bad_points > 0
            % 找出好的數據點索引
            good_indices = setdiff(1:data_length, bad_indices);

            % 對每個通道進行插值
            for ch = 1:6
                % 確保有足夠的好點進行插值
                if length(good_indices) >= 4
                    % 使用選定的插值方法
                    try
                        vm_clean(ch, bad_indices) = interp1(good_indices, ...
                            vm_data(ch, good_indices), bad_indices, interpolation_method, 'extrap');
                        da_clean(ch, bad_indices) = interp1(good_indices, ...
                            da_data(ch, good_indices), bad_indices, interpolation_method, 'extrap');
                    catch
                        % 如果高級插值失敗，降級到線性插值
                        fprintf('    警告: 通道%d插值失敗 使用線性插值\n', ch);
                        vm_clean(ch, bad_indices) = interp1(good_indices, ...
                            vm_data(ch, good_indices), bad_indices, 'linear', 'extrap');
                        da_clean(ch, bad_indices) = interp1(good_indices, ...
                            da_data(ch, good_indices), bad_indices, 'linear', 'extrap');
                    end
                else
                    % 數據點太少，使用簡單線性插值
                    for idx = bad_indices
                        if idx > 1 && idx < data_length
                            vm_clean(ch, idx) = (vm_data(ch, idx-1) + vm_data(ch, idx+1)) / 2;
                            da_clean(ch, idx) = (da_data(ch, idx-1) + da_data(ch, idx+1)) / 2;
                        end
                    end
                end
            end

            % 計算插值修復的統計信息
            vm_error_rms = sqrt(mean((vm_clean(:, bad_indices) - vm_bad_original).^2, 2));
            da_error_rms = sqrt(mean((da_clean(:, bad_indices) - da_bad_original).^2, 2));

            % 顯示修復效果統計
            fprintf('    VM修復RMS差異: %.6f V (平均)\n', mean(vm_error_rms));
            fprintf('    DA修復RMS差異: %.6f V (平均)\n', mean(da_error_rms));
        end

        fprintf('    原始數據點: %d, 修復壞點: %d\n', data_length, num_bad_points);
        
        % 轉換DA為電壓
        da_volt = (da_clean - 32768) * (20.0 / 65536);
        
        % 檢測激勵通道和頻率
        fprintf('  步驟3: 檢測激勵通道和頻率...\n');
        [excite_ch, excite_freq] = detect_excitation(da_volt, SAMPLING_RATE);
        fprintf('    激勵通道: DA%d, 頻率: %.1f Hz\n', excite_ch, excite_freq);
        
        % 穩態檢測 - 根據設定選擇檢測方法
        fprintf('  步驟4: 穩態檢測...\n');
        steady_info = detect_steady_state_advanced(vm_clean, excite_freq, SAMPLING_RATE, ...
            STABILITY_THRESHOLD, CONSECUTIVE_PERIODS, CHECK_POINTS, START_PERIOD);

        if isempty(steady_info)
            fprintf('  ✗ 穩態檢測失敗，跳過此檔案\n');
            continue;
        end

        fprintf('    穩態起始點: 第%d個週期\n', steady_info.period);

        % 繪製穩態波形疊圖（如果啟用）
        if PLOT_STEADY_STATE && should_plot_frequency(excite_freq, PLOT_FREQUENCY_LIST)
            plot_steady_state_overlay(vm_clean, da_volt, steady_info, excite_ch, ...
                excite_freq, CONSECUTIVE_PERIODS, PLOT_CHANNEL, SAMPLING_RATE);
        end

        % FFT分析
        fprintf('  步驟5: FFT分析 (模式: %s)...\n', FFT_MODE);
        current_magnitudes_db = zeros(6, 1);
        current_phases = zeros(6, 1);

        % 使用steady_info中的週期資訊
        period_samples = steady_info.period_samples;
        steady_start = steady_info.index;
        available_length = size(vm_clean, 2) - steady_start + 1;
        available_periods = floor(available_length / period_samples);

        if available_periods < 1
            fprintf('    警告: 數據不足一個完整週期\n');
            continue;
        end

        if strcmp(FFT_MODE, 'averaged')
            % === 週期平均模式 ===
            fprintf('    使用週期平均法: %d個週期\n', available_periods);

            % 提取激勵通道的DA信號用於所有通道
            da_signal = da_volt(excite_ch, :);

            % 提取所有週期並平均
            vm_periods = zeros(6, available_periods, period_samples);
            da_periods = zeros(available_periods, period_samples);

            for p = 1:available_periods
                start_idx = steady_start + (p-1) * period_samples;
                end_idx = start_idx + period_samples - 1;

                % 提取每個週期
                for ch = 1:6
                    vm_periods(ch, p, :) = vm_clean(ch, start_idx:end_idx);
                end
                da_periods(p, :) = da_signal(start_idx:end_idx);
            end

            % 計算平均週期
            vm_avg_periods = squeeze(mean(vm_periods, 2));  % 6 x period_samples
            da_avg_period = mean(da_periods, 1);            % 1 x period_samples

            % 計算週期標準差（用於評估穩定性）
            vm_std = squeeze(std(vm_periods, 0, 2));
            fprintf('    VM週期間標準差: %.6f (平均)\n', mean(vm_std(:)));

            % 對每個通道進行FFT
            for ch = 1:6
                % 單週期FFT
                vm_fft = fft(vm_avg_periods(ch, :));
                da_fft = fft(da_avg_period);

                % 對於單週期，基頻總是第2個bin
                target_bin = 2;

                % 計算轉移函數
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

            % 如果需要比較，也計算完整FFT
            if COMPARE_FFT_METHODS
                fprintf('    同時計算完整FFT進行比較...\n');
                magnitudes_full = zeros(6, 1);
                phases_full = zeros(6, 1);

                for ch = 1:6
                    % 提取完整數據
                    end_index = steady_start + available_periods * period_samples - 1;
                    vm_full = vm_clean(ch, steady_start:end_index);
                    da_full = da_volt(excite_ch, steady_start:end_index);

                    % 完整FFT
                    vm_fft_full = fft(vm_full);
                    da_fft_full = fft(da_full);

                    % 找目標頻率
                    N_full = length(vm_full);
                    freq_res = SAMPLING_RATE / N_full;
                    target_bin_full = round(excite_freq / freq_res) + 1;

                    % 計算
                    if abs(da_fft_full(target_bin_full)) > MIN_DA_THRESHOLD
                        H_full = vm_fft_full(target_bin_full) / da_fft_full(target_bin_full);
                        magnitudes_full(ch) = 20 * log10(abs(H_full));
                        phases_full(ch) = angle(H_full) * 180 / pi;
                    end
                end

                % 顯示比較結果
                fprintf('    === FFT方法比較 ===\n');
                for ch = 1:6
                    mag_diff = current_magnitudes_db(ch) - magnitudes_full(ch);
                    phase_diff = current_phases(ch) - phases_full(ch);
                    fprintf('    CH%d: ΔMag=%.3f dB, ΔPhase=%.2f°\n', ch, mag_diff, phase_diff);
                end
            end

        else
            % === 完整FFT模式（原方法） ===
            fprintf('    使用完整FFT法: %d個週期\n', available_periods);

            for ch = 1:6
                % 提取穩態後的完整週期數據
                vm_signal = vm_clean(ch, :);
                da_signal = da_volt(excite_ch, :);

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

                % 計算轉移函數
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
        end

        % 儲存當前頻率點的結果
        frequencies = [frequencies, excite_freq];
        magnitudes_db(:, end+1) = current_magnitudes_db;
        phases(:, end+1) = current_phases;
        excitation_channels = [excitation_channels, excite_ch];

        fprintf('  ✓ 成功處理頻率 %.1f Hz\n', excite_freq);

    catch err
        fprintf('  ✗ 處理失敗: %s\n', err.message);
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

%% 正規化大小數據
function magnitudes_normalized = normalize_magnitudes(magnitudes_db, frequencies)
% 每個通道以最低頻為基準
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

%% 檢測激勵通道和頻率
function [excite_ch, excite_freq] = detect_excitation(da_voltage, sampling_rate)
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

%% 穩態檢測
function steady_info = detect_steady_state_advanced(vm_signal, target_freq, sampling_rate, ...
    stability_threshold, consecutive_periods, check_points, start_period)
% 對多個VM通道進行穩態檢測，選擇最保守的結果
% 所有參數從主程式傳入
% 如果輸入是單一通道，轉為矩陣格式
if isvector(vm_signal)
    vm_clean = vm_signal(:)';  % 確保是行向量
    vm_clean = repmat(vm_clean, 6, 1);  % 複製成6通道
else
    vm_clean = vm_signal;
end

clean_length = size(vm_clean, 2);

% 計算週期相關參數
period_samples = round(sampling_rate / target_freq);
max_periods = floor(clean_length / period_samples);
check_positions = round(linspace(1, period_samples, check_points));

fprintf('    週期樣本數: %d，最大週期數: %d\n', period_samples, max_periods);

if max_periods < 5
    % 數據不足，使用最後一個週期
    fprintf('    ⚠ 警告: 週期數不足（只有%d個），使用最後1個週期作為穩態\n', max_periods);
    fprintf('    ⚠ 此數據可能包含暫態響應，結果可靠性較低！\n');

    % 使用最後一個完整週期
    steady_period = max_periods - 1;  % 最後一個週期的起始
    if steady_period < 1
        steady_period = 1;  % 至少要有一個週期
    end

    steady_info = struct('period', steady_period, ...
                        'index', steady_period * period_samples + 1, ...
                        'max_periods', max_periods, ...
                        'period_samples', period_samples);
    return;
end

steady_periods = [];

% 對每個VM通道進行穩態檢測
for vm_ch = 1:6
    signal = vm_clean(vm_ch, :);

    % 測試從start_period開始的每個週期
    for test_period = start_period:(max_periods - consecutive_periods)
        all_stable = true;

        % 檢查連續週期的穩定性
        for i = 1:consecutive_periods
            current_period = test_period + i - 1;
            next_period = current_period + 1;

            current_start = current_period * period_samples + 1;
            next_start = next_period * period_samples + 1;

            max_diff = 0;

            % 在指定檢查點比較相鄰週期的差異
            for pos_idx = 1:length(check_positions)
                pos = check_positions(pos_idx);
                current_idx = current_start + pos - 1;
                next_idx = next_start + pos - 1;

                if current_idx <= length(signal) && next_idx <= length(signal)
                    current_val = signal(current_idx);
                    next_val = signal(next_idx);
                    diff = abs(current_val - next_val);
                    max_diff = max(max_diff, diff);
                end
            end

            % 如果差異超過閾值，標記為不穩定
            if max_diff >= stability_threshold
                all_stable = false;
                break;
            end
        end

        % 如果找到穩定週期，記錄並停止搜索
        if all_stable
            steady_periods(end+1) = test_period;
            break;
        end
    end
end

% 選擇最保守的穩態點（最大值）
if ~isempty(steady_periods)
    recommended_period = max(steady_periods);
    clean_index = recommended_period * period_samples + 1;

    steady_info = struct(...
        'period', recommended_period, ...
        'index', clean_index, ...
        'max_periods', max_periods, ...
        'period_samples', period_samples);

    fprintf('    穩態檢測成功: 第%d週期，索引%d\n', recommended_period, clean_index);
else
    % 如果未找到穩定週期，使用最後的週期
    fprintf('    警告: 未找到符合穩定條件的週期（閾值%.4fV）\n', stability_threshold);
    fprintf('    使用最後%d個週期作為穩態（可能不夠穩定）\n', consecutive_periods);

    % 使用最後幾個週期
    steady_period = max(1, max_periods - consecutive_periods);
    steady_info = struct('period', steady_period, ...
                        'index', steady_period * period_samples + 1, ...
                        'max_periods', max_periods, ...
                        'period_samples', period_samples);
end
end

%% 繪製波德圖（幅度和相位）
function plot_bode_results(frequencies, magnitudes_db, phases, colors, original_magnitudes_db, excitation_channels, display_channels)

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
    
    % 正常繪製所有通道
    freq_plot = frequencies;
    mag_plot = magnitudes_db(ch, :);
    
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
    
    % 正常繪製所有通道
    freq_plot = frequencies;
    phase_plot = phases_processed(ch, :);
    
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

%% 穩態波形疊圖視覺化
function plot_steady_state_overlay(vm_clean, da_volt, steady_info, excite_ch, ...
    target_freq, num_periods, plot_channel, sampling_rate)
% 繪製穩態後的波形疊圖
% 輸入:
%   vm_clean: VM數據 (6 x N)
%   da_volt: DA電壓數據 (6 x N)
%   steady_info: 穩態資訊結構
%   excite_ch: 激勵通道
%   target_freq: 信號頻率
%   num_periods: 顯示週期數（來自CONSECUTIVE_PERIODS）
%   plot_channel: 0=所有通道, 1-6=特定通道
%   sampling_rate: 採樣率

% 提取穩態資訊
steady_start = steady_info.index;
period_samples = steady_info.period_samples;
max_periods = steady_info.max_periods;

% 計算可用週期數
available_from_steady = floor((size(vm_clean, 2) - steady_start + 1) / period_samples);
periods_to_plot = min(num_periods, available_from_steady);

if periods_to_plot < 1
    fprintf('    警告: 穩態後數據不足，無法繪製疊圖\n');
    return;
end

% 時間軸（正規化到一個週期）
time_axis = (0:period_samples-1) / sampling_rate * 1000;  % 轉為毫秒
time_normalized = (0:period_samples-1) / period_samples * 2 * pi;  % 正規化到0-2π

% 決定要繪製的通道
if plot_channel == 0
    channels_to_plot = 1:6;
else
    channels_to_plot = plot_channel;
end

% 創建圖形
figure('Name', sprintf('穩態波形疊圖 - %.1f Hz', target_freq), ...
       'Position', [50, 50, 1200, 800]);

% 顏色設定
colors = lines(max(periods_to_plot, 3));  % 確保至少有3個顏色

num_subplots = length(channels_to_plot);
plot_rows = ceil(sqrt(num_subplots));
plot_cols = ceil(num_subplots / plot_rows);

for idx = 1:num_subplots
    ch = channels_to_plot(idx);
    subplot(plot_rows, plot_cols, idx);
    hold on;

    % 繪製每個週期的VM數據
    legend_entries = {};
    all_vm_data = [];  % 儲存所有週期數據

    for p = 1:periods_to_plot
        period_start = steady_start + (p-1) * period_samples;
        period_end = period_start + period_samples - 1;

        if period_end <= size(vm_clean, 2)
            vm_data = vm_clean(ch, period_start:period_end);
            all_vm_data(p, :) = vm_data;  % 儲存數據

            % 繪製VM波形（不使用Alpha通道，改用顏色漸變）
            color_adjusted = colors(p,:) * (0.3 + 0.7 * (p/periods_to_plot));  % 顏色漸變
            h_vm = plot(time_axis, vm_data, '-', ...
                       'Color', color_adjusted, ...
                       'LineWidth', 1.5);
            legend_entries{end+1} = sprintf('週期 %d', steady_info.period + p - 1);
        end
    end

    % 計算實際偏差（如果有多個週期）
    max_deviation = 0;
    if size(all_vm_data, 1) > 1
        baseline = all_vm_data(1, :);  % 第一個週期作為基準

        for p = 2:size(all_vm_data, 1)
            deviations = abs(all_vm_data(p, :) - baseline);
            max_deviation = max(max_deviation, max(deviations));
        end

        % 添加參考線（基於第一個週期）
        stability_threshold = evalin('caller', 'STABILITY_THRESHOLD');

        % 畫出偏差容許範圍（淺灰色區域）
        upper_limit = baseline + stability_threshold;
        lower_limit = baseline - stability_threshold;

        % 使用 fill 創建陰影區域
        fill_x = [time_axis, fliplr(time_axis)];
        fill_y = [upper_limit, fliplr(lower_limit)];
        h_fill = fill(fill_x, fill_y, [0.8, 0.8, 0.8], ...
                     'FaceAlpha', 0.2, 'EdgeColor', 'none');
        uistack(h_fill, 'bottom');  % 放到最底層

        % 添加閾值線（虛線）
        plot(time_axis, baseline, 'k--', 'LineWidth', 1);
        legend_entries{end+1} = '基準線';
    end

    % 繪製DA波形（激勵通道）
    if ch == excite_ch || plot_channel == 0
        % 取第一個週期的DA作為參考
        da_start = steady_start;
        da_end = da_start + period_samples - 1;

        if da_end <= size(da_volt, 2)
            da_data = da_volt(excite_ch, da_start:da_end);

            % 創建右側Y軸
            yyaxis right;
            h_da = plot(time_axis, da_data, 'k-', ...
                       'LineWidth', 2, 'DisplayName', sprintf('DA%d', excite_ch));
            ylabel('DA電壓 (V)', 'FontWeight', 'bold');
            set(gca, 'YColor', 'k');

            % 切回左側Y軸
            yyaxis left;
        end
    end

    % 設定標籤和格式
    xlabel('時間 (ms)', 'FontWeight', 'bold');
    ylabel('VM值', 'FontWeight', 'bold');

    % 設定標題與偏差標註
    if ch == excite_ch
        title_str = sprintf('通道 %d (excited)', ch);
    else
        title_str = sprintf('通道 %d', ch);
    end

    % 設定標題（暫時移除偏差標註功能）
    title(title_str, 'FontWeight', 'bold');

    grid on;
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    set(gca, 'FontWeight', 'bold');

    % 標註穩態檢測點
    if p == 1
        xline(0, 'r--', '穩態起始', 'LabelVerticalAlignment', 'top');
    end
end

% 總標題（根據穩態狀態調整）
% 判斷穩態檢測狀態
consecutive_periods = evalin('caller', 'CONSECUTIVE_PERIODS');
if steady_info.max_periods < 5
    % 數據不足
    status_str = ' [數據不足，使用備用]';
    title_color = [1, 0.5, 0];  % 橙色
elseif steady_info.period >= steady_info.max_periods - consecutive_periods
    % 未達標準，使用備用（使用最後幾個週期）
    status_str = ' [未達穩態標準，使用備用]';
    title_color = 'r';  % 紅色
else
    % 穩態檢測成功（找到真正的穩態）
    status_str = ' [穩態檢測成功]';
    title_color = 'k';  % 黑色
end

sgtitle(sprintf('穩態波形疊圖 @ %.1f Hz (第%d週期開始，共%d個週期)%s', ...
        target_freq, steady_info.period, periods_to_plot, status_str), ...
        'FontWeight', 'bold', 'FontSize', 14, 'Color', title_color);

% 添加說明文字
annotation('textbox', [0.02, 0.02, 0.96, 0.03], ...
          'String', sprintf('穩態檢測: 閾值=%.4f V, 連續週期=%d, 檢查點=%d', ...
                           evalin('caller', 'STABILITY_THRESHOLD'), ...
                           num_periods, ...
                           evalin('caller', 'CHECK_POINTS')), ...
          'HorizontalAlignment', 'center', ...
          'EdgeColor', 'none', 'FontSize', 10);

fprintf('    波形疊圖已顯示: %d個週期從第%d週期開始\n', ...
        periods_to_plot, steady_info.period);
end

%% 判斷是否需要繪圖
function should_plot = should_plot_frequency(freq, freq_list)
% 根據頻率列表判斷是否需要繪圖
if isempty(freq_list)
    should_plot = true;  % 空列表表示繪製所有頻率
else
    % 容差範圍（考慮浮點數比較）
    tolerance = 0.1;  % Hz
    should_plot = any(abs(freq - freq_list) < tolerance);
end
end