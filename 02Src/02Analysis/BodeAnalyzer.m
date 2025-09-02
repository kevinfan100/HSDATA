% BodeAnalyzer.m - 多頻率波德圖分析系統
% 
% 主要功能：
% 1. 批量處理多個頻率的CSV數據檔案
% 2. 自動檢測激勵通道和頻率
% 3. 使用穩態後所有整數週期進行FFT分析
% 4. 生成6通道的波德圖（線性大小 + 對數頻率）
%
% 使用方法：
% 1. 將所有頻率的CSV檔案放在 '01Data/02Processed_csv/' 資料夾
% 2. 執行 main_bode_analysis() 函數
%
% 基於HSDataTemplate.m架構，擴展波德圖分析功能

%% ===== 配置參數區 =====
% 系統參數
SAMPLING_RATE = 100000;                    % 採樣頻率 (Hz)
VM_FREQ_TOLERANCE = 0.05;                  % VM頻率偏差容差 (5%)
MIN_VD_THRESHOLD = 1e-10;                  % VD最小幅值閾值
DATA_FOLDER = '01Data/02Processed_csv/';   % 數據資料夾路徑

% 穩態檢測參數（沿用HSDataTemplate設定）
CONSECUTIVE_PERIODS = 2;                   % 連續穩定週期數
CHECK_POINTS = 5;                          % 每週期檢查點數
STABILITY_THRESHOLD = 1e-3;                % 穩定性閾值
START_PERIOD = 1;                          % 開始檢測的週期數

% 波德圖設定
MAGNITUDE_SCALE = 'linear';                % 大小：線性尺度
FREQUENCY_SCALE = 'log';                   % 頻率：對數尺度
CHANNEL_COLORS = ['r','b','g','m','c','k']; % 6通道顏色

%% ===== 主要執行函數 =====

function main_bode_analysis(varargin)
    % 主執行函數 - 批量波德圖分析
    %
    % 使用方法:
    %   main_bode_analysis()                    % 使用預設資料夾
    %   main_bode_analysis('custom_folder/')    % 使用自訂資料夾
    
    fprintf('=== 波德圖分析系統 ===\n');
    
    % 解析輸入參數
    if nargin > 0
        csv_folder = varargin{1};
    else
        csv_folder = DATA_FOLDER;  % 使用配置中的預設路徑
    end
    
    fprintf('使用資料夾: %s\n', csv_folder);
    
    % 檢查資料夾是否存在
    if ~exist(csv_folder, 'dir')
        error('數據資料夾不存在: %s', csv_folder);
    end
    
    % 批量分析所有CSV檔案
    fprintf('\n開始批量分析...\n');
    [frequencies, magnitudes, phases] = batch_analyze_csv_folder(csv_folder);
    
    % 檢查是否有有效結果
    if isempty(frequencies)
        fprintf('\n錯誤: 沒有有效的分析結果\n');
        fprintf('請檢查:\n');
        fprintf('1. 資料夾中是否有CSV檔案\n');
        fprintf('2. CSV檔案格式是否正確\n');
        fprintf('3. 數據中是否有有效的激勵信號\n');
        return;
    end
    
    fprintf('\n=== 分析完成 ===\n');
    fprintf('成功處理 %d 個頻率點\n', length(frequencies));
    fprintf('頻率範圍: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));
    
    % 繪製波德圖
    fprintf('\n生成波德圖...\n');
    plot_bode_diagram_vertical(frequencies, magnitudes, phases);
    
    % 顯示完成信息
    fprintf('\n=== 波德圖分析完成！===\n');
    fprintf('結果:\n');
    fprintf('- 頻率點數: %d\n', length(frequencies));
    fprintf('- 通道數: 6\n');
    fprintf('- 圖形已顯示\n');
    
    % 可選：將結果保存到工作空間變數
    assignin('base', 'bode_frequencies', frequencies);
    assignin('base', 'bode_magnitudes', magnitudes);
    assignin('base', 'bode_phases', phases);
    fprintf('- 結果已保存到工作空間變數: bode_frequencies, bode_magnitudes, bode_phases\n');
end

%% ===== 第一部分：資料預處理函數（基於HSDataTemplate.m）=====

function [vm_data, vd_data, da_data] = load_csv_data(csv_filepath)
    % 讀取CSV檔案並分離VM、VD、DA數據（沿用HSDataTemplate.m）
    
    if ~exist(csv_filepath, 'file')
        error('檔案不存在: %s', csv_filepath);
    end
    
    raw_data = readtable(csv_filepath);
    data_length = height(raw_data);
    
    fprintf('成功讀取CSV檔案，共%d筆記錄\n', data_length);
    
    % 初始化數據矩陣
    vm_data = zeros(6, data_length);
    vd_data = zeros(6, data_length);
    da_data = zeros(6, data_length);
    
    % 提取各通道數據
    for i = 1:6
        vm_col = sprintf('vm_%d', i-1);
        vd_col = sprintf('vd_%d', i-1);
        da_col = sprintf('da_%d', i-1);
        
        if ismember(vm_col, raw_data.Properties.VariableNames)
            vm_data(i, :) = raw_data.(vm_col);
        else
            warning('找不到欄位: %s', vm_col);
        end
        
        if ismember(vd_col, raw_data.Properties.VariableNames)
            vd_data(i, :) = raw_data.(vd_col);
        else
            warning('找不到欄位: %s', vd_col);
        end
        
        if ismember(da_col, raw_data.Properties.VariableNames)
            da_data(i, :) = raw_data.(da_col);
        else
            warning('找不到欄位: %s', da_col);
        end
    end
end

function voltage = dac_to_voltage(dac_value)
    % 將16位DAC值轉換為±10V電壓（沿用HSDataTemplate.m）
    voltage = (dac_value - 32768) * (20.0 / 65536);
end

function [vm_clean, vd_clean, da_clean] = clean_all_data(vm_raw, vd_raw, da_raw)
    % 統一清理所有數據，排除異常點（沿用HSDataTemplate.m）
    
    data_length = size(vm_raw, 2);
    
    % 排除每10000個樣本點（避免採集異常）
    exclude_indices = 1:10000:data_length;
    valid_mask = true(1, data_length);
    valid_mask(exclude_indices) = false;
    
    % 統一應用遮罩到所有數據
    vm_clean = vm_raw(:, valid_mask);
    vd_clean = vd_raw(:, valid_mask);
    da_clean = da_raw(:, valid_mask);
    
    fprintf('排除%d個異常點，乾淨數據長度: %d\n', length(exclude_indices), sum(valid_mask));
end

function steady_info = detect_steady_state_clean(vm_clean, target_freq, varargin)
    % 在乾淨數據上進行穩態檢測（沿用HSDataTemplate.m）
    
    % 解析輸入參數
    p = inputParser;
    addParameter(p, 'sampling_rate', SAMPLING_RATE);
    addParameter(p, 'start_period', START_PERIOD);
    addParameter(p, 'consecutive_periods', CONSECUTIVE_PERIODS);
    addParameter(p, 'check_points', CHECK_POINTS);
    addParameter(p, 'threshold', STABILITY_THRESHOLD);
    parse(p, varargin{:});
    
    params = p.Results;
    clean_length = size(vm_clean, 2);
    
    % 計算週期相關參數
    period_samples = round(params.sampling_rate / target_freq);
    max_periods = floor(clean_length / period_samples);
    check_positions = round(linspace(1, period_samples, params.check_points));
    
    fprintf('乾淨數據週期樣本數: %d，最大週期數: %d\n', period_samples, max_periods);
    
    steady_periods = [];
    
    % 對每個VM通道進行穩態檢測
    for vm_ch = 1:6
        signal = vm_clean(vm_ch, :);
        
        % 測試從start_period開始的每個週期
        for test_period = params.start_period:(max_periods - params.consecutive_periods)
            all_stable = true;
            
            % 檢查連續週期的穩定性
            for i = 1:params.consecutive_periods
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
                if max_diff >= params.threshold
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
        clean_index = recommended_period * period_samples + 1;  % 乾淨數據中的索引
        
        steady_info = struct(...
            'period', recommended_period, ...
            'index', clean_index, ...           % 乾淨數據中的索引
            'max_periods', max_periods, ...
            'period_samples', period_samples);
        
        fprintf('穩態檢測成功: 第%d週期，乾淨數據索引%d\n', recommended_period, clean_index);
    else
        steady_info = [];
        fprintf('穩態檢測失敗: 未找到穩定週期\n');
    end
end

function [excite_ch, excite_freq] = find_excitation_channel(da_voltage, sampling_rate)
    % 同時檢測激勵通道和頻率
    %
    % 輸入:
    %   da_voltage - DA電壓數據 (6 x N)
    %   sampling_rate - 採樣頻率 (Hz)
    % 輸出:
    %   excite_ch - 激勵通道編號 (1-6)
    %   excite_freq - 激勵頻率 (Hz)
    
    fprintf('檢測激勵通道和頻率...\n');
    
    best_channel = 0;
    max_energy = 0;
    best_freq = 0;
    
    % 檢查每個DA通道
    for ch = 1:6
        signal_data = da_voltage(ch, :);
        
        % 計算信號的總能量（RMS）
        signal_energy = sqrt(mean(signal_data.^2));
        
        % 只對能量較大的信號進行FFT分析
        if signal_energy > 0.1  % 閾值可調整
            % FFT分析檢測主頻率
            N = length(signal_data);
            fft_result = fft(signal_data);
            freq_axis = (0:N-1) * sampling_rate / N;
            
            % 只考慮正頻率部分，且排除DC成分
            positive_freqs = freq_axis(2:floor(N/2));
            positive_fft = abs(fft_result(2:floor(N/2)));
            
            % 找到最大幅值對應的頻率
            [max_amplitude, max_idx] = max(positive_fft);
            dominant_freq = positive_freqs(max_idx);
            
            % 計算該頻率處的能量
            freq_energy = max_amplitude;
            
            fprintf('  CH%d: 總能量=%.3f, 主頻率=%.1fHz, 頻率能量=%.1f\n', ...
                    ch, signal_energy, dominant_freq, freq_energy);
            
            % 選擇頻率能量最大的通道
            if freq_energy > max_energy
                max_energy = freq_energy;
                best_channel = ch;
                best_freq = dominant_freq;
            end
        else
            fprintf('  CH%d: 總能量=%.3f (太小，跳過)\n', ch, signal_energy);
        end
    end
    
    if best_channel == 0
        error('未檢測到有效的激勵通道');
    end
    
    excite_ch = best_channel;
    excite_freq = best_freq;
    
    fprintf('檢測結果: 激勵通道 DA%d, 激勵頻率 %.1f Hz\n', excite_ch, excite_freq);
end

%% ===== 第二部分：FFT週期性分析函數 =====

function period_data = extract_all_integer_periods(signal, steady_info, excite_freq, sampling_rate)
    % 提取穩態後所有可用的完整週期
    %
    % 輸入:
    %   signal - 輸入信號 (1 x N)
    %   steady_info - 穩態檢測結果結構
    %   excite_freq - 激勵頻率 (Hz)
    %   sampling_rate - 採樣頻率 (Hz)
    % 輸出:
    %   period_data - 完整週期數據 (1 x M)
    
    % 計算每週期的採樣點數
    period_samples = round(sampling_rate / excite_freq);
    
    % 計算穩態後可用的數據長度
    steady_start = steady_info.index;
    available_length = length(signal) - steady_start + 1;
    
    % 計算可提取的完整週期數
    available_periods = floor(available_length / period_samples);
    
    if available_periods < 1
        error('穩態後數據不足一個完整週期');
    end
    
    % 提取所有完整週期的數據
    end_index = steady_start + available_periods * period_samples - 1;
    period_data = signal(steady_start:end_index);
    
    fprintf('提取了 %d 個完整週期，數據長度: %d 點\n', available_periods, length(period_data));
end

function [vm_fft_results, vd_fft_results, freq_axis] = fft_analysis_with_all_periods(vm_data, vd_data, excite_ch, excite_freq, steady_info)
    % 對所有VM通道和激勵VD通道進行FFT分析
    %
    % 輸入:
    %   vm_data - VM數據 (6 x N)
    %   vd_data - VD數據 (6 x N)
    %   excite_ch - 激勵通道編號
    %   excite_freq - 激勵頻率 (Hz)
    %   steady_info - 穩態檢測結果
    % 輸出:
    %   vm_fft_results - VM的FFT結果 (6 x M)
    %   vd_fft_results - VD的FFT結果 (1 x M)
    %   freq_axis - 頻率軸 (1 x M)
    
    fprintf('執行FFT分析...\n');
    
    % 對所有VM通道提取週期數據並進行FFT
    vm_fft_results = zeros(6, 0);
    
    for ch = 1:6
        vm_signal = vm_data(ch, :);
        vm_period_data = extract_all_integer_periods(vm_signal, steady_info, excite_freq, SAMPLING_RATE);
        
        % FFT分析
        vm_fft = fft(vm_period_data);
        vm_fft_results(ch, :) = vm_fft;
    end
    
    % 對激勵VD通道進行相同處理
    vd_signal = vd_data(excite_ch, :);
    vd_period_data = extract_all_integer_periods(vd_signal, steady_info, excite_freq, SAMPLING_RATE);
    vd_fft_results = fft(vd_period_data);
    
    % 建立頻率軸
    N = length(vd_period_data);
    freq_axis = (0:N-1) * SAMPLING_RATE / N;
    
    fprintf('FFT分析完成，頻率解析度: %.3f Hz\n', SAMPLING_RATE / N);
end

function [actual_freq_bin, deviation_percent] = detect_vm_frequency_deviation(vm_fft, excite_freq, freq_axis)
    % 檢測VM信號主頻率與激勵頻率的偏差
    %
    % 輸入:
    %   vm_fft - VM通道的FFT結果 (1 x N)
    %   excite_freq - 激勵頻率 (Hz)
    %   freq_axis - 頻率軸 (1 x N)
    % 輸出:
    %   actual_freq_bin - 實際使用的頻率bin
    %   deviation_percent - 頻率偏差百分比
    
    % 計算理論頻率bin
    freq_resolution = freq_axis(2) - freq_axis(1);
    theoretical_bin = round(excite_freq / freq_resolution) + 1;  % +1因為MATLAB索引從1開始
    
    % 定義搜尋範圍（±5%）
    tolerance = VM_FREQ_TOLERANCE;
    freq_range = excite_freq * tolerance;
    search_bins = round(freq_range / freq_resolution);
    
    % 確保搜尋範圍在有效範圍內
    start_bin = max(1, theoretical_bin - search_bins);
    end_bin = min(length(freq_axis), theoretical_bin + search_bins);
    
    % 在搜尋範圍內找到最大幅值
    search_range = start_bin:end_bin;
    [~, max_idx] = max(abs(vm_fft(search_range)));
    actual_freq_bin = search_range(max_idx);
    
    % 計算偏差
    actual_freq = freq_axis(actual_freq_bin);
    deviation_percent = abs(actual_freq - excite_freq) / excite_freq * 100;
    
    % 檢查偏差是否超過容差
    if deviation_percent > tolerance * 100
        fprintf('警告: 頻率偏差 %.1f%% 超過容差 %.1f%%\n', deviation_percent, tolerance * 100);
    end
end

function [magnitude_ratio, phase_diff] = calculate_vm_vd_ratio(vm_fft, vd_fft, freq_bin)
    % 計算傳遞函數 H(jω) = VM(jω) / VD(jω)
    %
    % 輸入:
    %   vm_fft - VM通道的FFT結果 (1 x N)
    %   vd_fft - VD通道的FFT結果 (1 x N)
    %   freq_bin - 目標頻率的bin索引
    % 輸出:
    %   magnitude_ratio - 大小比值（線性）
    %   phase_diff - 相位差（度數）
    
    % 提取複數值
    vm_complex = vm_fft(freq_bin);
    vd_complex = vd_fft(freq_bin);
    
    % 檢查VD信號是否足夠大
    vd_magnitude = abs(vd_complex);
    if vd_magnitude < MIN_VD_THRESHOLD
        fprintf('警告: VD信號幅值太小 (%.2e)，設為零\n', vd_magnitude);
        magnitude_ratio = 0;
        phase_diff = 0;
        return;
    end
    
    % 計算傳遞函數
    transfer_function = vm_complex / vd_complex;
    
    % 提取大小和相位
    magnitude_ratio = abs(transfer_function);
    phase_diff = angle(transfer_function) * 180 / pi;  % 轉換為度數
end

%% ===== 第三部分：波德圖繪製函數 =====

function [frequencies, magnitudes, phases] = batch_analyze_csv_folder(csv_folder_path)
    % 批量處理資料夾內所有CSV檔案
    %
    % 輸入:
    %   csv_folder_path - CSV檔案資料夾路徑
    % 輸出:
    %   frequencies - 所有頻率點 (1 x N)
    %   magnitudes - 6通道的大小數據 (6 x N)
    %   phases - 6通道的相位數據 (6 x N)
    
    fprintf('批量分析資料夾: %s\n', csv_folder_path);
    
    % 獲取所有CSV檔案
    csv_files = dir(fullfile(csv_folder_path, '*.csv'));
    
    if isempty(csv_files)
        error('資料夾中沒有找到CSV檔案: %s', csv_folder_path);
    end
    
    fprintf('找到 %d 個CSV檔案\n', length(csv_files));
    
    % 初始化結果陣列
    frequencies = [];
    magnitudes = zeros(6, 0);
    phases = zeros(6, 0);
    
    % 處理每個CSV檔案
    for i = 1:length(csv_files)
        csv_file = csv_files(i);
        file_path = fullfile(csv_folder_path, csv_file.name);
        
        fprintf('\n[%d/%d] 處理檔案: %s\n', i, length(csv_files), csv_file.name);
        
        try
            % 第一部分：預處理
            fprintf('  步驟1: 讀取和清理數據...\n');
            [vm_raw, vd_raw, da_raw] = load_csv_data(file_path);
            [vm, vd, da] = clean_all_data(vm_raw, vd_raw, da_raw);
            da_volt = dac_to_voltage(da);
            
            % 檢測激勵通道和頻率
            fprintf('  步驟2: 檢測激勵通道和頻率...\n');
            [excite_ch, excite_freq] = find_excitation_channel(da_volt, SAMPLING_RATE);
            
            % 穩態檢測
            fprintf('  步驟3: 穩態檢測...\n');
            steady_info = detect_steady_state_clean(vm, excite_freq);
            
            if isempty(steady_info)
                fprintf('  ✗ 穩態檢測失敗，跳過此檔案\n');
                continue;
            end
            
            % 第二部分：FFT分析
            fprintf('  步驟4: FFT分析...\n');
            [vm_fft, vd_fft, freq_axis] = fft_analysis_with_all_periods(vm, vd, excite_ch, excite_freq, steady_info);
            
            % 計算所有6個通道的VM/VD比值
            fprintf('  步驟5: 計算傳遞函數...\n');
            current_magnitudes = zeros(6, 1);
            current_phases = zeros(6, 1);
            
            for ch = 1:6
                % 檢測VM頻域偏差
                [freq_bin, ~] = detect_vm_frequency_deviation(vm_fft(ch, :), excite_freq, freq_axis);
                
                % 計算VM/VD比值
                [mag, phase] = calculate_vm_vd_ratio(vm_fft(ch, :), vd_fft, freq_bin);
                
                current_magnitudes(ch) = mag;
                current_phases(ch) = phase;
            end
            
            % 添加到結果陣列
            frequencies(end+1) = excite_freq;
            magnitudes(:, end+1) = current_magnitudes;
            phases(:, end+1) = current_phases;
            
            fprintf('  ✓ 分析完成：頻率 %.1f Hz\n', excite_freq);
            
        catch ME
            fprintf('  ✗ 處理失敗: %s\n', ME.message);
            continue;
        end
    end
    
    % 按頻率排序結果
    if ~isempty(frequencies)
        [frequencies, sort_idx] = sort(frequencies);
        magnitudes = magnitudes(:, sort_idx);
        phases = phases(:, sort_idx);
        
        fprintf('\n批量分析完成！\n');
        fprintf('成功處理 %d 個頻率點: ', length(frequencies));
        fprintf('%.1f ', frequencies);
        fprintf('Hz\n');
    else
        fprintf('\n批量分析完成，但沒有有效結果\n');
    end
end

function plot_bode_diagram_vertical(frequencies, magnitudes, phases)
    % 繪製6通道垂直排列的波德圖
    %
    % 輸入:
    %   frequencies - 頻率點 (1 x N)
    %   magnitudes - 6通道大小數據 (6 x N)
    %   phases - 6通道相位數據 (6 x N)
    
    if isempty(frequencies)
        error('沒有數據可繪製');
    end
    
    fprintf('繪製波德圖...\n');
    
    % 創建圖形窗口
    figure('Name', '6通道波德圖', 'Position', [100, 100, 1200, 800]);
    
    % 定義6通道顏色
    colors = CHANNEL_COLORS;
    
    % 上圖：大小響應（線性尺度）
    subplot(2, 1, 1);
    hold on;
    
    for ch = 1:6
        % 使用對數頻率軸和線性大小
        semilogx(frequencies, magnitudes(ch, :), ...
                'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
                'MarkerSize', 6, 'DisplayName', sprintf('CH%d', ch));
    end
    
    % 設定上圖屬性
    xlabel('頻率 (Hz)');
    ylabel('大小 (線性)');
    title('波德圖 - 大小響應');
    legend('Location', 'best');
    grid on;
    grid minor;
    hold off;
    
    % 設定合理的Y軸範圍
    y_min = min(magnitudes(:));
    y_max = max(magnitudes(:));
    if y_max > y_min
        ylim([max(0, y_min * 0.9), y_max * 1.1]);
    end
    
    % 下圖：相位響應
    subplot(2, 1, 2);
    hold on;
    
    for ch = 1:6
        % 使用對數頻率軸
        semilogx(frequencies, phases(ch, :), ...
                'Color', colors(ch), 'LineWidth', 2, 'Marker', 's', ...
                'MarkerSize', 6, 'DisplayName', sprintf('CH%d', ch));
    end
    
    % 設定下圖屬性
    xlabel('頻率 (Hz)');
    ylabel('相位 (度)');
    title('波德圖 - 相位響應');
    legend('Location', 'best');
    grid on;
    grid minor;
    hold off;
    
    % 設定相位軸範圍
    ylim([-180, 180]);
    
    % 添加相位參考線
    yline(0, '--k', 'Alpha', 0.3);
    yline(90, '--k', 'Alpha', 0.3);
    yline(-90, '--k', 'Alpha', 0.3);
    
    % 整體佈局調整
    sgtitle('6通道波德圖分析結果', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 顯示統計信息
    fprintf('波德圖統計信息:\n');
    fprintf('頻率範圍: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));
    fprintf('大小範圍: %.3f - %.3f\n', min(magnitudes(:)), max(magnitudes(:)));
    fprintf('相位範圍: %.1f - %.1f 度\n', min(phases(:)), max(phases(:)));
    
    fprintf('波德圖繪製完成！\n');
end

%% ===== 使用範例 =====

% 範例1: 基本使用（使用預設資料夾）
% main_bode_analysis();

% 範例2: 指定自訂資料夾
% main_bode_analysis('01Data/02Processed_csv/my_experiment/');

% 範例3: 手動單檔案分析
% [vm_raw, vd_raw, da_raw] = load_csv_data('01Data/02Processed_csv/test1.csv');
% [vm, vd, da] = clean_all_data(vm_raw, vd_raw, da_raw);
% da_volt = dac_to_voltage(da);
% [excite_ch, excite_freq] = find_excitation_channel(da_volt, SAMPLING_RATE);
% steady_info = detect_steady_state_clean(vm, excite_freq);
% [vm_fft, vd_fft, freq_axis] = fft_analysis_with_all_periods(vm, vd, excite_ch, excite_freq, steady_info);

% 範例4: 僅繪製已有的數據
% plot_bode_diagram_vertical(bode_frequencies, bode_magnitudes, bode_phases);

%% ===== 系統功能總結 =====
% 
% 1. 自動批量處理: 掃描資料夾中所有CSV檔案
% 2. 智慧激勵檢測: 自動找出激勵通道和頻率
% 3. 穩態分析: 確保使用穩定的信號段
% 4. 週期性FFT: 使用整數週期避免頻譜洩漏
% 5. 多通道分析: 同時分析6個VM通道
% 6. 波德圖可視化: 線性大小+對數頻率，垂直佈局
% 
% 輸入要求:
% - CSV檔案包含vm_0~vm_5, vd_0~vd_5, da_0~da_5欄位
% - 激勵信號需要有明顯的週期性
% - 數據長度足夠包含穩態後的多個週期
% 
% 輸出結果:
% - 6通道波德圖（大小+相位）
% - 工作空間變數: bode_frequencies, bode_magnitudes, bode_phases
%
%% ===== 故障排除 =====
%
% 常見問題:
% 1. "數據資料夾不存在" → 檢查路徑設定
% 2. "未檢測到有效的激勵通道" → 檢查DA信號是否有週期性
% 3. "穩態檢測失敗" → 調整STABILITY_THRESHOLD或檢查信號品質
% 4. "頻率偏差超過容差" → 檢查激勵頻率穩定性
% 5. "VD信號幅值太小" → 檢查激勵通道的信號強度
%
% 參數調整:
% - VM_FREQ_TOLERANCE: 調整頻率偏差容差
% - STABILITY_THRESHOLD: 調整穩態檢測敏感度
% - MIN_VD_THRESHOLD: 調整最小激勵信號閾值