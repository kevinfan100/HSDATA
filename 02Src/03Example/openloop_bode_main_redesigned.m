function openloop_bode_main(varargin)
% OPENLOOP_BODE_MAIN 
%% ===== 全域參數設定 =====
% 硬體參數
SAMPLING_RATE = 100000;        % 採樣率 (Hz)
DAC_ZERO = 32768;              % 16-bit DAC 中點值
DAC_RANGE = 20.0;              % DAC 電壓範圍 (±10V = 20V)
DAC_BITS = 65536;              % 2^16

% 數據處理參數
EXCLUDE_INTERVAL = 10000;      % 清理異常值的間隔
RMS_THRESHOLD = 0.1;           % 激勵檢測的RMS閾值 (V)
MIN_DA_THRESHOLD = 1e-6;       % DA信號最小閾值

% 穩態檢測參數
STABILITY_THRESHOLD = 0.001;    % 穩定性閾值 (1mV)
CONSECUTIVE_PERIODS = 3;        % 需要連續穩定的週期數
CHECK_POINTS = 10;             % 每週期的檢查點數
START_PERIOD = 2;              % 開始檢測的週期

% 視覺化控制
PLOT_STEADY = false;           % 是否顯示穩態驗證圖
PLOT_FFT_SPECTRUM = false;     % 是否顯示FFT頻譜圖
DEBUG_MODE = true;            % 除錯模式（顯示詳細相位處理資訊）

% 顯示設定
DISPLAY_CHANNELS = 1:6;        % 要顯示的通道
CHANNEL_COLORS = [
    0.0, 0.4, 0.7;  % 通道1: 藍
    0.8, 0.2, 0.2;  % 通道2: 紅
    0.2, 0.6, 0.2;  % 通道3: 綠
    0.9, 0.5, 0.0;  % 通道4: 橘
    0.5, 0.0, 0.5;  % 通道5: 紫
    0.0, 0.6, 0.6;  % 通道6: 青
];

%% ===== 主程式開始 =====
% 決定資料目錄
if nargin > 0
    data_dir = varargin{1};
else
    data_dir = 'C:\Users\PME406_01\Desktop\code\HSDATA\01Data\02Processed_csv\openloop_Cali_P5';
end

fprintf('資料目錄: %s\n', data_dir);

% 取得所有CSV檔案
csv_files = dir(fullfile(data_dir, '*.csv'));
if isempty(csv_files)
    error('找不到CSV檔案');
end

fprintf('找到 %d 個CSV檔案\n\n', length(csv_files));

% 初始化結果陣列
frequencies = [];
magnitudes_db = zeros(6, 0);
phases = zeros(6, 0);
excitation_channels = [];
original_magnitudes_db = zeros(6, 0);

% 處理每個CSV檔案
for file_idx = 1:length(csv_files)
    filename = csv_files(file_idx).name;
    filepath = fullfile(data_dir, filename);

    fprintf('[%d/%d] 處理檔案: %s\n', file_idx, length(csv_files), filename);

    try
        %% 步驟1: 讀取與清理數據
        fprintf('  步驟1: 讀取與清理數據...\n');
        [vm_clean, da_clean] = read_and_clean_data(filepath, EXCLUDE_INTERVAL);

        %% 步驟2: DAC轉電壓
        fprintf('  步驟2: DAC轉電壓...\n');
        da_volt = dac_to_voltage(da_clean, DAC_ZERO, DAC_RANGE, DAC_BITS);

        %% 步驟3: 激勵檢測
        fprintf('  步驟3: 激勵檢測...\n');
        [excite_ch, excite_freq] = detect_excitation(da_volt, filename, RMS_THRESHOLD);
        fprintf('    激勵通道: DA%d, 頻率: %.1f Hz\n', excite_ch, excite_freq);

        %% 步驟4: 穩態檢測
        fprintf('  步驟4: 穩態檢測...\n');
        steady_info = detect_steady_state_advanced(vm_clean, excite_freq, ...
            SAMPLING_RATE, STABILITY_THRESHOLD, CONSECUTIVE_PERIODS, ...
            CHECK_POINTS, START_PERIOD);

        if isempty(steady_info)
            fprintf('  ✗ 穩態檢測失敗，跳過此檔案\n');
            continue;
        end

        fprintf('    穩態起始: 第%d週期 (索引%d)\n', ...
            steady_info.period, steady_info.index);

        % 可選：顯示穩態驗證圖
        if PLOT_STEADY
            plot_steady_validation_combined(vm_clean, da_volt, steady_info, ...
                excite_ch, excite_freq, SAMPLING_RATE);
        end

        %% 步驟5: 提取穩態數據
        vm_steady = vm_clean(:, steady_info.index:end);
        da_steady = da_volt(:, steady_info.index:end);

        %% 步驟6: FFT分析（使用原始程式碼方式）
        fprintf('  步驟5: FFT分析...\n');
        current_magnitudes_db = zeros(6, 1);
        current_phases = zeros(6, 1);

        for ch = 1:6
            % 提取穩態後的完整週期數據
            vm_signal = vm_steady(ch, :);
            da_signal = da_steady(excite_ch, :);

            % 計算可用的完整週期
            period_samples = round(SAMPLING_RATE / excite_freq);
            available_length = length(vm_signal);
            available_periods = floor(available_length / period_samples);

            if available_periods < 1
                fprintf('    警告: CH%d 數據不足一個完整週期\n', ch);
                current_magnitudes_db(ch) = -Inf;
                current_phases(ch) = 0;
                continue;
            end

            % 提取整數週期的數據
            end_index = available_periods * period_samples;
            vm_period_data = vm_signal(1:end_index);
            da_period_data = da_signal(1:end_index);

            % 直接FFT（不做週期平均）
            vm_fft = fft(vm_period_data);
            da_fft = fft(da_period_data);

            % 找到目標頻率的bin
            N = length(vm_period_data);
            freq_resolution = SAMPLING_RATE / N;
            target_bin = round(excite_freq / freq_resolution) + 1;

            % 計算轉移函數
            vm_complex = vm_fft(target_bin);
            da_complex = da_fft(target_bin);

            if abs(da_complex) > MIN_DA_THRESHOLD
                transfer_function = vm_complex / da_complex;
                magnitude_linear = abs(transfer_function);
                current_magnitudes_db(ch) = 20 * log10(magnitude_linear);
                current_phases(ch) = angle(transfer_function) * 180 / pi;

                % Debug輸出
                if DEBUG_MODE && (excite_freq >= 500)
                    fprintf('    CH%d @ %.1fHz:\n', ch, excite_freq);
                    fprintf('      VM: mag=%.4f, phase=%.2f°\n', ...
                        abs(vm_complex), angle(vm_complex)*180/pi);
                    fprintf('      DA: mag=%.4f, phase=%.2f°\n', ...
                        abs(da_complex), angle(da_complex)*180/pi);
                    fprintf('      轉移函數相位: %.2f°\n', current_phases(ch));
                end
            else
                current_magnitudes_db(ch) = -Inf;
                current_phases(ch) = 0;
            end
        end

        fprintf('    完成FFT分析\n');

        %% 步驟8: 儲存結果
        frequencies = [frequencies, excite_freq];
        magnitudes_db(:, end+1) = current_magnitudes_db;
        phases(:, end+1) = current_phases;
        excitation_channels = [excitation_channels, excite_ch];
        original_magnitudes_db(:, end+1) = current_magnitudes_db;

        fprintf('  ✓ 成功處理頻率 %.1f Hz\n', excite_freq);

    catch err
        fprintf('  ✗ 處理失敗: %s\n', err.message);
        continue;
    end
end

%% ===== 後處理與繪圖 =====
if isempty(frequencies)
    fprintf('\n沒有成功處理的檔案\n');
    return;
end

% 排序結果（按頻率）
[frequencies, sort_idx] = sort(frequencies);
magnitudes_db = magnitudes_db(:, sort_idx);
phases = phases(:, sort_idx);
excitation_channels = excitation_channels(sort_idx);
original_magnitudes_db = original_magnitudes_db(:, sort_idx);

% 處理相位數據（根據激勵通道群組調整）
fprintf('\n處理相位數據...\n');
phases_processed = phases;
for freq_idx = 1:length(frequencies)
    excite_ch = excitation_channels(freq_idx);

    if ismember(excite_ch, [1, 3, 6])
        % 如果激勵通道是1、3、6，只對激勵通道減去180度
        phases_processed(excite_ch, freq_idx) = phases_processed(excite_ch, freq_idx) - 180;
        if DEBUG_MODE
            fprintf('  %.1fHz: 激勵通道%d屬於[1,3,6]群組，CH%d相位-180°\n', ...
                frequencies(freq_idx), excite_ch, excite_ch);
        end
    else
        % 如果激勵通道是2、4、5，對非激勵通道減去180度
        for ch = 1:6
            if ch ~= excite_ch
                phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) - 180;
            end
        end
        if DEBUG_MODE
            fprintf('  %.1fHz: 激勵通道%d屬於[2,4,5]群組，非激勵通道相位-180°\n', ...
                frequencies(freq_idx), excite_ch);
        end
    end

    % 確保相位在 -180 到 180 度範圍內
    for ch = 1:6
        while phases_processed(ch, freq_idx) > 180
            phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) - 360;
        end
        while phases_processed(ch, freq_idx) < -180
            phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) + 360;
        end
    end
end

% 使用處理後的相位數據
phases = phases_processed;

% 正規化幅值（以最低頻率為基準）
fprintf('\n正規化幅值數據...\n');
magnitudes_db_normalized = normalize_magnitudes(magnitudes_db, frequencies);

fprintf('\n=== 分析完成 ===\n');
fprintf('成功處理 %d 個頻率點\n', length(frequencies));
fprintf('頻率範圍: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));

% 繪製Bode圖
plot_bode_results(frequencies, magnitudes_db_normalized, phases, ...
    CHANNEL_COLORS, original_magnitudes_db, excitation_channels, ...
    DISPLAY_CHANNELS);

fprintf('\n程式執行完成\n');
end

%% ========== 子函數定義 ==========

%% 1. 數據讀取與清理
function [vm_clean, da_clean] = read_and_clean_data(filepath, exclude_interval)
    % 讀取CSV檔案
    raw_data = readtable(filepath);
    data_length = height(raw_data);

    % 初始化數據矩陣
    vm_data = zeros(6, data_length);
    da_data = zeros(6, data_length);

    % 提取VM和DA數據（6個通道）
    for ch = 1:6
        vm_col = sprintf('vm_%d', ch-1);
        da_col = sprintf('da_%d', ch-1);

        % 檢查欄位是否存在
        if ~ismember(vm_col, raw_data.Properties.VariableNames)
            error('缺少欄位: %s', vm_col);
        end
        if ~ismember(da_col, raw_data.Properties.VariableNames)
            error('缺少欄位: %s', da_col);
        end

        vm_data(ch, :) = raw_data.(vm_col);
        da_data(ch, :) = raw_data.(da_col);
    end

    % 清理數據：排除每10000點的第一個點
    exclude_indices = 1:exclude_interval:data_length;
    valid_mask = true(1, data_length);
    valid_mask(exclude_indices) = false;

    vm_clean = vm_data(:, valid_mask);
    da_clean = da_data(:, valid_mask);

    fprintf('    原始數據: %d 點, 清理後: %d 點\n', ...
        data_length, sum(valid_mask));
end

%% 2. DAC轉電壓
function da_volt = dac_to_voltage(da_clean, dac_zero, dac_range, dac_bits)
    % DAC數值轉換為電壓
    % 16-bit DAC: 0-65535 對應 -10V 到 +10V
    da_volt = (da_clean - dac_zero) * (dac_range / dac_bits);
end

%% 3. 激勵檢測
function [excite_ch, excite_freq] = detect_excitation(da_volt, filename, rms_threshold)
    % 從檔名擷取頻率
    % 支援兩種格式: P5_100Hz.csv 或 P5_100.csv
    pattern = '_(\d+(?:\.\d+)?)(?:Hz)?\.csv';
    tokens = regexp(filename, pattern, 'tokens');

    if isempty(tokens)
        error('無法從檔名擷取頻率: %s', filename);
    end

    excite_freq = str2double(tokens{1}{1});

    % 計算每個通道的RMS值
    rms_values = sqrt(mean(da_volt.^2, 2));

    % 找出RMS最大的通道作為激勵通道
    [max_rms, excite_ch] = max(rms_values);

    % 強制使用通道5（如果檔名是P5開頭）
    if contains(filename, 'P5')
        excite_ch = 5;
        fprintf('    強制使用通道5作為激勵通道\n');
    end

    % 檢查激勵信號強度
    if max_rms < rms_threshold
        warning('激勵信號過弱: %.4f V < %.4f V', max_rms, rms_threshold);
    end

    fprintf('    DA%d RMS: %.4f V\n', excite_ch, max_rms);
end

%% 4. 進階穩態檢測
function steady_info = detect_steady_state_advanced(vm_signal, target_freq, ...
    sampling_rate, stability_threshold, consecutive_periods, check_points, start_period)

    % 計算週期參數
    period_samples = round(sampling_rate / target_freq);
    [num_channels, data_length] = size(vm_signal);
    max_periods = floor(data_length / period_samples);

    % 生成檢查點位置（均勻分布在週期內）
    check_positions = round(linspace(1, period_samples, check_points));

    fprintf('    週期樣本數: %d, 總週期數: %d\n', period_samples, max_periods);

    % 檢查是否有足夠的數據
    if max_periods < (start_period + consecutive_periods)
        fprintf('    警告: 數據不足，使用備用方法\n');
        steady_period = max(1, max_periods - 2);
        steady_info = struct(...
            'period', steady_period, ...
            'index', steady_period * period_samples + 1, ...
            'period_samples', period_samples, ...
            'max_periods', max_periods);
        return;
    end

    % 記錄每個通道的穩態週期
    channel_steady_periods = zeros(1, num_channels);

    % 對每個VM通道獨立檢測
    for ch = 1:num_channels
        signal = vm_signal(ch, :);
        found_steady = false;

        % 從前往後搜索穩態點
        for test_period = start_period:(max_periods - consecutive_periods)
            all_stable = true;

            % 取參考週期
            ref_start = test_period * period_samples + 1;
            ref_end = ref_start + period_samples - 1;

            if ref_end > data_length
                break;
            end

            % 檢查後續consecutive_periods個週期
            for p = 1:consecutive_periods
                next_start = (test_period + p) * period_samples + 1;
                next_end = next_start + period_samples - 1;

                if next_end > data_length
                    all_stable = false;
                    break;
                end

                % 在檢查點比較週期差異
                max_diff = 0;
                for check_idx = 1:length(check_positions)
                    pos = check_positions(check_idx);

                    ref_idx = ref_start + pos - 1;
                    next_idx = next_start + pos - 1;

                    if ref_idx <= data_length && next_idx <= data_length
                        diff = abs(signal(ref_idx) - signal(next_idx));
                        max_diff = max(max_diff, diff);
                    end
                end

                % 檢查是否超過閾值
                if max_diff > stability_threshold
                    all_stable = false;
                    break;
                end
            end

            % 找到穩定週期
            if all_stable
                channel_steady_periods(ch) = test_period;
                found_steady = true;
                break;
            end
        end

        % 如果沒找到，設為最後幾個週期
        if ~found_steady
            channel_steady_periods(ch) = max(start_period, max_periods - consecutive_periods);
        end
    end

    % 選擇最保守的結果（最晚進入穩態）
    final_steady_period = max(channel_steady_periods);
    steady_index = final_steady_period * period_samples + 1;

    % 返回結果
    steady_info = struct(...
        'period', final_steady_period, ...
        'index', steady_index, ...
        'period_samples', period_samples, ...
        'max_periods', max_periods, ...
        'channel_periods', channel_steady_periods, ...
        'threshold_used', stability_threshold);

    % 顯示結果
    fprintf('    各通道穩態週期: [');
    fprintf('%d ', channel_steady_periods);
    fprintf(']\n');
    fprintf('    最終選擇: 第%d週期\n', final_steady_period);
end

%% 5. FFT分析（含VM主頻驗證）
function [vm_mag, vm_phase, da_mag, da_phase, quality_check] = ...
    analyze_with_validation(vm_steady, da_steady, excite_freq, excite_ch, sampling_rate)

    % 計算週期參數
    period_samples = round(sampling_rate / excite_freq);
    n_periods = floor(size(vm_steady, 2) / period_samples);

    fprintf('    使用 %d 個完整週期進行FFT分析\n', n_periods);

    % 初始化輸出
    vm_mag = zeros(6, 1);
    vm_phase = zeros(6, 1);
    quality_check = struct();
    quality_check.is_valid = true(6, 1);
    quality_check.vm_dominant_freq = zeros(6, 1);
    quality_check.freq_error_percent = zeros(6, 1);

    % VM通道分析
    for ch = 1:6
        % 週期平均（降噪）
        data_segment = vm_steady(ch, 1:n_periods*period_samples);
        vm_periods = reshape(data_segment, period_samples, n_periods);
        vm_avg = mean(vm_periods, 2)';

        % 完整FFT
        N = length(vm_avg);
        fft_result = fft(vm_avg);
        magnitude_spectrum = abs(fft_result) * 2 / N;
        magnitude_spectrum(1) = magnitude_spectrum(1) / 2;  % DC修正

        % 頻率軸
        freq_resolution = sampling_rate / N;

        % 找VM主頻（排除DC）
        [max_mag, max_idx] = max(magnitude_spectrum(2:floor(N/2)));
        vm_dominant_freq = max_idx * freq_resolution;

        % 檢查主頻是否與激勵頻率一致
        freq_error = abs(vm_dominant_freq - excite_freq);
        freq_error_percent = freq_error / excite_freq * 100;

        quality_check.vm_dominant_freq(ch) = vm_dominant_freq;
        quality_check.freq_error_percent(ch) = freq_error_percent;

        if freq_error_percent > 5
            quality_check.is_valid(ch) = false;
            fprintf('    ⚠ VM%d: 主頻=%.2fHz, 預期=%.2fHz (誤差%.1f%%)\n', ...
                ch, vm_dominant_freq, excite_freq, freq_error_percent);
        end

        % 提取激勵頻率響應
        target_bin = round(excite_freq / freq_resolution) + 1;
        vm_mag(ch) = magnitude_spectrum(target_bin);
        vm_phase(ch) = angle(fft_result(target_bin));
    end

    % DA激勵通道分析（直接計算，不檢查主頻）
    da_segment = da_steady(excite_ch, 1:n_periods*period_samples);
    da_periods = reshape(da_segment, period_samples, n_periods);
    da_avg = mean(da_periods, 2)';

    % FFT
    N = length(da_avg);
    da_fft = fft(da_avg);
    target_bin = round(excite_freq / (sampling_rate/N)) + 1;

    da_mag = abs(da_fft(target_bin)) * 2 / N;
    da_phase = angle(da_fft(target_bin));

    fprintf('    DA%d激勵: %.4f V @ %.1f Hz\n', excite_ch, da_mag, excite_freq);

    % 驗證總結
    if all(quality_check.is_valid)
        fprintf('    ✅ 所有VM通道主頻驗證通過\n');
    else
        failed_channels = find(~quality_check.is_valid);
        fprintf('    ⚠ 通道 %s 需要檢查\n', num2str(failed_channels));
    end
end

%% 6. 穩態視覺化（可選）
function plot_steady_validation_combined(vm_signal, da_volt, steady_info, ...
    excite_ch, freq, fs)

    period_samples = steady_info.period_samples;
    steady_start = steady_info.index;

    n_cycles = 3;  % 顯示3個週期
    colors = [0 0 1; 1 0 0; 0 0.7 0];  % 藍、紅、綠

    figure('Name', sprintf('穩態驗證 - %.1fHz', freq), ...
        'Position', [100, 100, 1400, 900]);

    % 提取DA的一個週期（作為參考）
    da_cycle = da_volt(excite_ch, steady_start:steady_start+period_samples-1);
    time_norm = (0:period_samples-1) / period_samples;

    for ch = 1:6
        subplot(2, 3, ch);
        hold on;

        % 先畫DA激勵（黑色虛線，作為參考）
        plot(time_norm, da_cycle, 'k--', ...
            'LineWidth', 1.5, ...
            'DisplayName', 'DA激勵');

        % 疊加VM的3個週期（彩色實線）
        for cycle = 0:2
            start_idx = steady_start + cycle * period_samples;
            end_idx = start_idx + period_samples - 1;

            if end_idx <= size(vm_signal, 2)
                vm_cycle = vm_signal(ch, start_idx:end_idx);
                plot(time_norm, vm_cycle, ...
                    'Color', colors(cycle+1, :), ...
                    'LineWidth', 1.5, ...
                    'DisplayName', sprintf('VM週期%d', cycle+1));
            end
        end

        xlabel('正規化週期');
        ylabel('電壓 (V)');
        title(sprintf('VM%d vs DA%d', ch, excite_ch));
        grid on;

        if ch == 1
            legend('Location', 'best', 'FontSize', 8);
        end
    end

    sgtitle(sprintf('穩態驗證 @ %.1f Hz (從第%d週期開始)', ...
        freq, steady_info.period));
end

%% 7. FFT頻譜視覺化（可選）
function plot_fft_spectrum(vm_steady, excite_freq, fs, show_full_spectrum)
    if ~show_full_spectrum
        return;
    end

    % 計算FFT（使用週期平均）
    period_samples = round(fs / excite_freq);
    n_periods = floor(size(vm_steady, 2) / period_samples);

    figure('Name', sprintf('FFT頻譜分析 @ %.1f Hz激勵', excite_freq), ...
        'Position', [100, 100, 1400, 900]);

    for ch = 1:6
        % 週期平均
        vm_periods = reshape(vm_steady(ch, 1:n_periods*period_samples), ...
            period_samples, n_periods);
        vm_avg = mean(vm_periods, 2)';

        % FFT
        N = length(vm_avg);
        fft_result = fft(vm_avg);
        magnitude = abs(fft_result) * 2 / N;
        magnitude(1) = magnitude(1) / 2;  % DC成分不需要乘2

        freq_axis = (0:N-1) * fs / N;

        % 只顯示到Nyquist頻率
        half_N = floor(N/2);
        freq_plot = freq_axis(1:half_N);
        mag_plot = magnitude(1:half_N);

        subplot(2, 3, ch);

        % DC成分（灰色bar）
        if mag_plot(1) > 0
            bar(0, mag_plot(1), 0.5, 'FaceColor', [0.7 0.7 0.7], ...
                'DisplayName', sprintf('DC: %.4f V', mag_plot(1)));
            hold on;
        end

        % AC成分
        semilogy(freq_plot(2:end), mag_plot(2:end), 'b-', 'LineWidth', 1);
        hold on;

        % 標記激勵頻率（紅圈）
        excite_bin = round(excite_freq / (fs/N)) + 1;
        if excite_bin <= half_N
            semilogy(excite_freq, mag_plot(excite_bin), 'ro', ...
                'MarkerSize', 10, 'LineWidth', 2, ...
                'DisplayName', sprintf('%.1f Hz: %.4f V', ...
                excite_freq, mag_plot(excite_bin)));
        end

        % 標記諧波（2次、3次）
        for harmonic = 2:3
            harm_freq = harmonic * excite_freq;
            harm_bin = round(harm_freq / (fs/N)) + 1;
            if harm_bin <= half_N && harm_freq < fs/2
                semilogy(harm_freq, mag_plot(harm_bin), 'gx', ...
                    'MarkerSize', 8, 'LineWidth', 2);
                text(harm_freq, mag_plot(harm_bin), ...
                    sprintf(' %dx', harmonic), 'FontSize', 8);
            end
        end

        xlabel('頻率 (Hz)');
        ylabel('幅值 (V)');
        title(sprintf('VM%d 頻譜', ch));
        grid on;
        xlim([0, min(1000, fs/2)]);
        ylim([1e-6, max(mag_plot)*2]);

        legend('Location', 'northeast');
    end

    sgtitle(sprintf('FFT頻譜分析 - 激勵頻率: %.1f Hz', excite_freq));
end

%% 8. 幅值正規化
function magnitudes_normalized = normalize_magnitudes(magnitudes_db, frequencies)
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

        if isfinite(reference_value)
            magnitudes_normalized(ch, :) = magnitudes_db(ch, :) - reference_value;
            fprintf('  通道%d: 參考值 = %.2f dB\n', ch, reference_value);
        else
            magnitudes_normalized(ch, :) = magnitudes_db(ch, :);
            fprintf('  通道%d: 無有效參考值\n', ch);
        end
    end
end

%% 9. 繪製Bode圖
function plot_bode_results(frequencies, magnitudes_db, phases, colors, ...
    original_magnitudes_db, excitation_channels, display_channels)

    % 處理相位數據（沿用原程式碼方案）
    phases_processed = phases;

    % 診斷輸出
    fprintf('\n相位處理診斷:\n');
    fprintf('激勵通道序列: ');
    fprintf('%d ', unique(excitation_channels));
    fprintf('\n');

    % 檢查各通道的數據
    fprintf('各通道數據檢查:\n');
    for ch = 1:6
        valid_points = sum(isfinite(magnitudes_db(ch, :)));
        if valid_points == 0
            fprintf('  通道%d: 無有效數據！\n', ch);
        else
            fprintf('  通道%d: %d個有效點\n', ch, valid_points);
        end
    end

    % 顯示處理前的相位範圍
    fprintf('\n處理前相位範圍:\n');
    for ch = 1:6
        fprintf('  通道%d: %.1f 到 %.1f 度\n', ch, min(phases(ch,:)), max(phases(ch,:)));
    end

    for freq_idx = 1:length(frequencies)
        excite_ch = excitation_channels(freq_idx);

        if ismember(excite_ch, [1, 3, 6])
            % 激勵通道是1,3,6：只對激勵通道減180度
            phases_processed(excite_ch, freq_idx) = phases_processed(excite_ch, freq_idx) - 180;
        else
            % 激勵通道是2,4,5：對非激勵通道減180度
            for ch = 1:6
                if ch ~= excite_ch
                    phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) - 180;
                end
            end
        end
    end

    % 確保相位在 -180 到 180 範圍內
    fprintf('\n相位範圍調整:\n');
    for ch = 1:6
        for freq_idx = 1:length(frequencies)
            original = phases_processed(ch, freq_idx);
            adjusted = original;

            while adjusted > 180
                adjusted = adjusted - 360;
            end
            while adjusted < -180
                adjusted = adjusted + 360;
            end

            if original ~= adjusted && frequencies(freq_idx) >= 500
                fprintf('  CH%d @ %.1fHz: %.1f° -> %.1f°\n', ...
                    ch, frequencies(freq_idx), original, adjusted);
            end

            phases_processed(ch, freq_idx) = adjusted;
        end
    end

    % 顯示處理後的相位範圍
    fprintf('\n處理後相位範圍:\n');
    for ch = 1:6
        fprintf('  通道%d: %.1f 到 %.1f 度\n', ch, min(phases_processed(ch,:)), max(phases_processed(ch,:)));
    end

    % 計算正規化參考值（最低頻率處的原始線性值）
    [~, min_freq_idx] = min(frequencies);
    normalization_values_db = original_magnitudes_db(:, min_freq_idx);
    normalization_values_linear = 10.^(normalization_values_db/20);

    % 創建圖形
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

        semilogx(frequencies, magnitudes_db(ch, :), ...
            'Color', colors(ch, :), 'LineWidth', 2, 'Marker', 'o', ...
            'MarkerSize', 10, 'DisplayName', legend_text);
    end

    % 添加二階系統理論響應（可選）
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
    legend('Location', 'southwest', 'FontWeight', 'bold', 'FontSize', 24);

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
    grid on;

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

        semilogx(frequencies, phases_processed(ch, :), ...
            'Color', colors(ch, :), 'LineWidth', 2, 'Marker', 'o', ...
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
    grid on;

    fprintf('\nBode圖繪製完成\n');
    fprintf('頻率範圍: %.2f - %.2f Hz\n', min(frequencies), max(frequencies));
    fprintf('頻率點數: %d\n', length(frequencies));
end