% HSDataTemplate.m - VM/VD/DA數據處理基礎模板，模組化進行相關分析
% 
% 主要功能：
% 1. 讀取CSV格式的VM、VD、DA數據
% 2. DAC轉電壓處理
% 3. 穩態檢測
% 4. 時域響應可視化
%
% 使用方法：
% 1. 設定參數區的檔案路徑和參數
% 2. 執行main()函數
% 3. 或依需求調用個別功能函數

%% ===== 參數設定區 =====
% 數據路徑設定
DATA_FOLDER = '01Data/02Processed_csv/0805_B_data/';  % 數據資料夾路徑
CSV_FILE = '0_1.csv';                                % 要處理的CSV檔案

% 系統參數
SAMPLING_RATE = 100000;                              % 採樣頻率 (Hz)
TARGET_FREQ = 1;                                     % 目標頻率 (Hz)

% 穩態檢測參數
CONSECUTIVE_PERIODS = 2;                             % 連續穩定週期數
CHECK_POINTS = 5;                                    % 每週期檢查點數
STABILITY_THRESHOLD = 1e-3;                          % 穩定性閾值
START_PERIOD = 1;                                    % 開始檢測的週期數

% 可視化參數
DISPLAY_PERIODS = 10;                                % 顯示週期數
CHANNELS_TO_PLOT = 1:6;                              % 要顯示的通道 (1-6)

%% ===== 主要功能函數(進行數據處裡) =====

function main()
    % 主執行函數 - 展示完整的數據處理流程（使用乾淨數據）
    
    % 設定參數
    data_folder = '01Data/02Processed_csv/0805_B_data/';
    csv_file = '0_1.csv';
    target_freq = 1;  % Hz
    
    fprintf('=== HSData處理模板（乾淨數據版本）===\n');
    fprintf('處理檔案: %s%s\n', data_folder, csv_file);
    fprintf('目標頻率: %d Hz\n', target_freq);
    
    % 1. 讀取原始CSV數據
    fprintf('\n步驟1: 讀取原始CSV數據...\n');
    [vm_raw, vd_raw, da_raw] = load_csv_data([data_folder csv_file]);
    
    % 2. 清理數據（排除異常點）
    fprintf('步驟2: 清理數據...\n');
    [vm_clean, vd_clean, da_clean] = clean_all_data(vm_raw, vd_raw, da_raw);
    
    % 3. DAC轉電壓
    fprintf('步驟3: 轉換DAC為電壓...\n');
    da_voltage_clean = dac_to_voltage(da_clean);
    
    % 4. 穩態檢測（在乾淨數據上）
    fprintf('步驟4: 穩態檢測...\n');
    steady_info = detect_steady_state_clean(vm_clean, target_freq);
    
    if isempty(steady_info)
        fprintf('錯誤: 未檢測到穩態\n');
        return;
    end
    
    fprintf('檢測到穩態: 第%d週期，乾淨數據索引%d\n', steady_info.period, steady_info.index);
    
    % 5. 可視化乾淨數據
    fprintf('步驟5: 數據可視化...\n');
    plot_clean_signals(vm_clean, 'VM', steady_info, target_freq, 10, 1:6);
    plot_clean_signals(vd_clean, 'VD', steady_info, target_freq, 10, 1:6);
    plot_clean_signals(da_voltage_clean, 'DA', steady_info, target_freq, 10, 1:6);
    
    fprintf('處理完成！返回的數據都是已清理的乾淨數據\n');
end

function [vm_data, vd_data, da_data] = load_csv_data(csv_filepath)
    % 讀取CSV檔案並分離VM、VD、DA數據
    %
    % 輸入:
    %   csv_filepath - CSV檔案完整路徑
    % 輸出:
    %   vm_data - VM數據 (6 x N)
    %   vd_data - VD數據 (6 x N) 
    %   da_data - DA數據 (6 x N)
    %   time_axis - 時間軸 (1 x N)
    
    % 檢查檔案是否存在
    if ~exist(csv_filepath, 'file')
        error('檔案不存在: %s', csv_filepath);
    end
    
    % 讀取CSV檔案
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
    
    % 注意：時間軸在後續清理數據時重新建立
end

function voltage = dac_to_voltage(dac_value)
    % 將16位DAC值轉換為±10V電壓
    %
    % 輸入:
    %   dac_value - DAC數值 (0-65535)
    % 輸出:
    %   voltage - 電壓值 (±10V)
    
    voltage = (dac_value - 32768) * (20.0 / 65536);
end

function [vm_clean, vd_clean, da_clean] = clean_all_data(vm_raw, vd_raw, da_raw)
    % 統一清理所有數據，排除異常點
    %
    % 輸入:
    %   vm_raw, vd_raw, da_raw - 原始數據 (6 x N)
    % 輸出:
    %   vm_clean, vd_clean, da_clean - 清理後數據 (6 x M, M<N)
    
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
    % 在乾淨數據上進行穩態檢測
    %
    % 輸入:
    %   vm_clean - 已清理的VM數據 (6 x N)
    %   target_freq - 目標頻率 (Hz)
    %   可選參數: 同原版
    
    % 解析輸入參數
    p = inputParser;
    addParameter(p, 'sampling_rate', 100000);
    addParameter(p, 'start_period', 1);
    addParameter(p, 'consecutive_periods', 2);
    addParameter(p, 'check_points', 5);
    addParameter(p, 'threshold', 1e-3);
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

%%===== 檢測激勵通道與頻率(如有需要) =====
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

%% ===== 繪製乾淨數據的時域響應圖 =====
function plot_clean_signals(clean_data, data_type, steady_info, target_freq, display_periods, channels)
    % 繪製乾淨數據的時域響應圖
    %
    % 輸入:
    %   clean_data - 乾淨數據矩陣 (6 x N)
    %   data_type - 數據類型字串 ('VM', 'VD', 'DA')
    %   steady_info - 穩態檢測結果
    %   target_freq - 目標頻率 (Hz)
    %   display_periods - 顯示週期數
    %   channels - 要顯示的通道 (陣列)
    
    if isempty(steady_info)
        fprintf('無法繪圖: 穩態檢測失敗\n');
        return;
    end
    
    % 計算顯示範圍（在乾淨數據中）
    period_samples = steady_info.period_samples;
    start_idx = steady_info.index;
    end_idx = start_idx + display_periods * period_samples - 1;
    
    % 檢查數據範圍
    if end_idx > size(clean_data, 2)
        end_idx = size(clean_data, 2);
        actual_periods = floor((end_idx - start_idx + 1) / period_samples);
        fprintf('警告: 乾淨數據不足，實際顯示%d個週期\n', actual_periods);
    end
    
    % 提取顯示數據
    display_data = clean_data(channels, start_idx:end_idx);
    time_axis = (0:(end_idx-start_idx)) / 100000;  % 轉換為秒
    
    % 創建圖形
    figure('Name', sprintf('%s時域響應（乾淨數據）- %dHz', data_type, target_freq), ...
           'Position', [100, 100, 1200, 800]);
    
    % 顏色配置
    colors = ['b', 'r', 'g', 'm', 'c', 'k'];
    
    % 繪製各通道數據
    hold on;
    for i = 1:length(channels)
        ch_idx = channels(i);
        plot(time_axis, display_data(i, :), 'Color', colors(mod(i-1, 6)+1), ...
             'LineWidth', 1.5, 'DisplayName', sprintf('Ch%d', ch_idx));
    end
    
    % 標記週期分界線
    for p = 1:display_periods
        period_time = p * (1/target_freq);
        if period_time <= time_axis(end)
            xline(period_time, '--k', 'Alpha', 0.3);
        end
    end
    
    % 設定圖形屬性
    xlabel('時間 (秒)');
    
    if strcmp(data_type, 'VM') || strcmp(data_type, 'VD')
        ylabel('電壓 (V)');
    else
        ylabel('電壓 (V)');  % DA已轉為電壓
    end
    
    title(sprintf('%s時域響應（乾淨數據）- %dHz (%d個週期)', data_type, target_freq, display_periods));
    legend('Location', 'best');
    grid on;
    hold off;
    
    fprintf('已顯示%s乾淨數據: %d個通道，%d個週期\n', data_type, length(channels), display_periods);
end

%% ===== 繪製VM vs VD相位圖 =====
function plot_vm_vd_phase_diagram(vm_clean, vd_clean, steady_info, target_freq, channels)
    % 繪製VM vs VD相位圖（一個週期的疊圖）
    %
    % 輸入:
    %   vm_clean - 乾淨VM數據 (6 x N)
    %   vd_clean - 乾淨VD數據 (6 x N)
    %   steady_info - 穩態檢測結果
    %   target_freq - 目標頻率 (Hz)
    %   channels - 要顯示的通道 (預設: 1:6)
    
    if nargin < 5
        channels = 1:6;  % 預設顯示所有通道
    end
    
    if isempty(steady_info)
        fprintf('無法繪圖: 穩態檢測失敗\n');
        return;
    end
    
    % 提取一個完整週期的數據
    period_samples = steady_info.period_samples;
    start_idx = steady_info.index;
    end_idx = start_idx + period_samples - 1;
    
    % 檢查數據範圍
    if end_idx > size(vm_clean, 2)
        fprintf('警告: 數據不足，無法提取完整週期\n');
        return;
    end
    
    % 創建圖形
    figure('Name', sprintf('VM vs VD 相位圖 - %dHz (1個週期)', target_freq), ...
           'Position', [100, 100, 1000, 800]);
    
    % 顏色配置
    colors = ['b', 'r', 'g', 'm', 'c', 'k'];
    
    % 繪製各通道的VM vs VD圖
    hold on;
    for i = 1:length(channels)
        ch = channels(i);
        
        % 提取該通道一個週期的VM和VD數據
        vm_period = vm_clean(ch, start_idx:end_idx);
        vd_period = vd_clean(ch, start_idx:end_idx);
        
        % 繪製相位圖（連線）
        plot(vd_period, vm_period, 'Color', colors(mod(i-1, 6)+1), ...
             'LineWidth', 2, 'DisplayName', sprintf('Ch%d', ch));
        
        % 標記起始點（圓點）
        scatter(vd_period(1), vm_period(1), 80, colors(mod(i-1, 6)+1), ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
        
        % 標記結束點（方形）
        scatter(vd_period(end), vm_period(end), 80, colors(mod(i-1, 6)+1), ...
                's', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    end
    
    % 設定圖形屬性
    xlabel('VD 電壓 (V)');
    ylabel('VM 電壓 (V)');
    title(sprintf('VM vs VD 相位圖 - %dHz (穩態後1個週期)', target_freq));
    legend('Location', 'best');
    grid on;
    axis equal;  % 保持軸比例相等
    hold off;
    
    % 添加圖例說明
    text(0.02, 0.98, '● 起始點  ■ 結束點', 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
         'EdgeColor', 'k', 'FontSize', 10);
    
    fprintf('已顯示VM vs VD相位圖: %d個通道，1個週期\n', length(channels));
end

%% ===== 開環FFT 分析 =====
%執行 FFT 並從其結果中計算出轉移函數的完整過程 同時將數據點必須按照頻率從低到高的順序連接起來
function [frequencies, magnitudes_db, phases] = batch_analyze_csv_folder_openloop(csv_folder_path)
    % 批量處理資料夾內所有CSV檔案（開環版本）
    %
    % 輸入:
    %   csv_folder_path - CSV檔案資料夾路徑
    % 輸出:
    %   frequencies - 所有頻率點 (1 x N)
    %   magnitudes_db - 6通道的大小數據 dB (6 x N)
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
    magnitudes_db = zeros(6, 0);
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
            
            % 第二部分：開環FFT分析
            fprintf('  步驟4: 開環FFT分析...\n');
            [vm_fft, da_fft, freq_axis] = fft_analysis_with_all_periods_openloop(vm, da_volt, excite_ch, excite_freq, steady_info);
            
            % 計算所有6個通道的VM/DA比值
            fprintf('  步驟5: 計算開環傳遞函數...\n');
            current_magnitudes_db = zeros(6, 1);
            current_phases = zeros(6, 1);
            
            for ch = 1:6
                % 檢測VM頻域偏差
                [freq_bin, ~] = detect_vm_frequency_deviation(vm_fft(ch, :), excite_freq, freq_axis);
                
                % 計算VM/DA比值
                [mag_db, phase] = calculate_vm_da_ratio(vm_fft(ch, :), da_fft, freq_bin);
                
                current_magnitudes_db(ch) = mag_db;
                current_phases(ch) = phase;
            end
            
            % 添加到結果陣列
            frequencies(end+1) = excite_freq;
            magnitudes_db(:, end+1) = current_magnitudes_db;
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
        magnitudes_db = magnitudes_db(:, sort_idx);
        phases = phases(:, sort_idx);
        
        fprintf('\n批量分析完成！\n');
        fprintf('成功處理 %d 個頻率點: ', length(frequencies));
        fprintf('%.1f ', frequencies);
        fprintf('Hz\n');
    else
        fprintf('\n批量分析完成，但沒有有效結果\n');
    end
end
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

function [vm_fft_results, da_fft_results, freq_axis] = fft_analysis_with_all_periods_openloop(vm_data, da_data, excite_ch, excite_freq, steady_info)
    % 對所有VM通道和激勵DA通道進行FFT分析（開環版本）
    %
    % 輸入:
    %   vm_data - VM數據 (6 x N)
    %   da_data - DA數據 (6 x N)
    %   excite_ch - 激勵通道編號
    %   excite_freq - 激勵頻率 (Hz)
    %   steady_info - 穩態檢測結果
    % 輸出:
    %   vm_fft_results - VM的FFT結果 (6 x M)
    %   da_fft_results - DA的FFT結果 (1 x M)
    %   freq_axis - 頻率軸 (1 x M)
    
    fprintf('執行開環FFT分析...\n');
    
    % 對所有VM通道提取週期數據並進行FFT
    vm_fft_results = zeros(6, 0);
    
    for ch = 1:6
        vm_signal = vm_data(ch, :);
        vm_period_data = extract_all_integer_periods(vm_signal, steady_info, excite_freq, SAMPLING_RATE);
        
        % FFT分析
        vm_fft = fft(vm_period_data);
        vm_fft_results(ch, :) = vm_fft;
    end
    
    % 對激勵DA通道進行相同處理
    da_signal = da_data(excite_ch, :);
    da_period_data = extract_all_integer_periods(da_signal, steady_info, excite_freq, SAMPLING_RATE);
    da_fft_results = fft(da_period_data);
    
    % 建立頻率軸
    N = length(da_period_data);
    freq_axis = (0:N-1) * SAMPLING_RATE / N;
    
    fprintf('開環FFT分析完成，頻率解析度: %.3f Hz\n', SAMPLING_RATE / N);
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

function [magnitude_db, phase_diff] = calculate_vm_da_ratio(vm_fft, da_fft, freq_bin)
    % 計算開環傳遞函數 H(jω) = VM(jω) / DA(jω)
    %
    % 輸入:
    %   vm_fft - VM通道的FFT結果 (1 x N)
    %   da_fft - DA通道的FFT結果 (1 x N)
    %   freq_bin - 目標頻率的bin索引
    % 輸出:
    %   magnitude_db - 大小比值（dB）
    %   phase_diff - 相位差（度數）
    
    % 提取複數值
    vm_complex = vm_fft(freq_bin);
    da_complex = da_fft(freq_bin);
    
    % 檢查DA信號是否足夠大
    da_magnitude = abs(da_complex);
    if da_magnitude < MIN_DA_THRESHOLD
        fprintf('警告: DA信號幅值太小 (%.2e)，設為零\n', da_magnitude);
        magnitude_db = -Inf;  % dB尺度下的零
        phase_diff = 0;
        return;
    end
    
    % 計算傳遞函數
    transfer_function = vm_complex / da_complex;
    
    % 提取大小和相位
    magnitude_linear = abs(transfer_function);
    magnitude_db = 20 * log10(magnitude_linear);  % 轉換為dB
    phase_diff = angle(transfer_function) * 180 / pi;  % 轉換為度數
end

%% ===== 利用FFT結果繪製open loop Bode plot 的function
function plot_openloop_bode_diagram_vertical(frequencies, magnitudes_db, phases)
    % 繪製6通道垂直排列的開環波德圖
    %
    % 輸入:
    %   frequencies - 頻率點 (1 x N)
    %   magnitudes_db - 6通道大小數據 dB (6 x N)
    %   phases - 6通道相位數據 (6 x N)
    
    if isempty(frequencies)
        error('沒有數據可繪製');
    end
    
    fprintf('繪製開環波德圖...\n');
    
    % 創建圖形窗口
    figure('Name', '6通道開環波德圖 (VM/DA)', 'Position', [100, 100, 1200, 800]);
    
    % 定義6通道顏色
    colors = CHANNEL_COLORS;
    
    % 上圖：大小響應（dB尺度）
    subplot(2, 1, 1);
    hold on;
    
    for ch = 1:6
        % 使用對數頻率軸和dB大小
        semilogx(frequencies, magnitudes_db(ch, :), ...
                'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
                'MarkerSize', 6, 'DisplayName', sprintf('CH%d', ch));
    end
    
    % 設定上圖屬性
    xlabel('頻率 (Hz)');
    ylabel('大小 (dB)');
    title('開環波德圖 - 大小響應 (VM/DA)');
    legend('Location', 'best');
    grid on;
    grid minor;
    hold off;
    
    % 設定合理的Y軸範圍
    finite_mags = magnitudes_db(isfinite(magnitudes_db));
    if ~isempty(finite_mags)
        y_min = min(finite_mags);
        y_max = max(finite_mags);
        if y_max > y_min
            ylim([y_min - 5, y_max + 5]);  % dB範圍留5dB餘量
        end
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
    title('開環波德圖 - 相位響應 (VM/DA)');
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
    sgtitle('6通道開環波德圖分析結果 (VM/DA)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 顯示統計信息
    fprintf('開環波德圖統計信息:\n');
    fprintf('頻率範圍: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));
    finite_mags = magnitudes_db(isfinite(magnitudes_db));
    if ~isempty(finite_mags)
        fprintf('大小範圍: %.1f - %.1f dB\n', min(finite_mags), max(finite_mags));
    end
    fprintf('相位範圍: %.1f - %.1f 度\n', min(phases(:)), max(phases(:)));
    
    fprintf('開環波德圖繪製完成！\n');
end

%% ===== 週期處理 =====
    %使用經過前面(main)中 load and clean 的資料作為輸入
    %這裡僅放如何使用的程式
    period_samples = round(SAMPLING_RATE / TARGET_FREQ);
    total_periods = floor(size(vm_clean, 2) / period_samples);
    
    fprintf('Samples per period: %d | Total periods: %d\n', period_samples, total_periods);
    
    % 檢查週期範圍
    if END_PERIOD > total_periods
        fprintf('警告: 結束週期超出範圍，調整為第%d週期\n', total_periods);
        END_PERIOD = total_periods;
    end
    
    if START_PERIOD < 1
        fprintf('警告: 開始週期調整為第1週期\n');
        START_PERIOD = 1;
    end
    
    if START_PERIOD > END_PERIOD
        fprintf('錯誤: 開始週期大於結束週期\n');
        return;
    end
    
    DISPLAY_PERIODS = END_PERIOD - START_PERIOD + 1;
    fprintf('Display: Period %d-%d (Total %d periods)\n', START_PERIOD, END_PERIOD, DISPLAY_PERIODS);
%% ===== 繪製指定週期範圍的信號圖表 =====
function plot_signal_range(data, signal_type, target_freq, sampling_rate, start_period, end_period, channels)
    % 繪製指定週期範圍的信號圖表
    
    period_samples = round(sampling_rate / target_freq);
    start_idx = (start_period - 1) * period_samples + 1;
    end_idx = end_period * period_samples;
    
    if end_idx > size(data, 2)
        end_idx = size(data, 2);
    end
    
    display_data = data(channels, start_idx:end_idx);
    time_axis = (start_idx-1:end_idx-1) / sampling_rate;
    
    figure('Name', sprintf('%s - 週期%d-%d', signal_type, start_period, end_period));
    
    colors = ['b', 'r', 'g', 'm', 'c', 'k'];
    for i = 1:length(channels)
        plot(time_axis, display_data(i, :), 'Color', colors(mod(i-1, 6)+1), 'LineWidth', 2.5);
        hold on;
    end
    
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s Signal - %dHz (Period %d-%d)', signal_type, target_freq, start_period, end_period), ...
          'FontSize', 14, 'FontWeight', 'bold');
    
    % 加粗座標軸數字
    set(gca, 'FontWeight', 'bold', 'FontSize', 14);
    
    % 加上簡單的顏色標示，圖例移到標題左方
    legend_labels = {};
    for i = 1:length(channels)
        legend_labels{i} = sprintf('Ch%d', channels(i));
    end
    h_legend = legend(legend_labels, 'Location', 'northwest', 'FontSize', 11, 'FontWeight', 'bold');
    h_legend.LineWidth = 2.5;  % 圖例線條加粗
    
    grid on;
    hold off;
end

%% ===== 繪製指定週期範圍的VM與DA疊圖 =====
function plot_overlay_range(vm_data, da_data, target_freq, sampling_rate, start_period, end_period, channels)
    % 繪製指定週期範圍的VM與DA疊圖
    
    period_samples = round(sampling_rate / target_freq);
    start_idx = (start_period - 1) * period_samples + 1;
    end_idx = end_period * period_samples;
    
    if end_idx > size(vm_data, 2)
        end_idx = size(vm_data, 2);
    end
    
    vm_display = vm_data(channels, start_idx:end_idx);
    da_display = da_data(channels, start_idx:end_idx);
    time_axis = (start_idx-1:end_idx-1) / sampling_rate;
    
    figure('Name', sprintf('VM & DA Overlay - Period %d-%d', start_period, end_period));
    
    for i = 1:length(channels)
        subplot(length(channels), 1, i);
        plot(time_axis, vm_display(i, :), 'b-', 'LineWidth', 2.5, 'DisplayName', 'VM');
        hold on;
        plot(time_axis, da_display(i, :), 'r-', 'LineWidth', 2.5, 'DisplayName', 'DA');
        
        ylabel('Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Channel %d', channels(i)), 'FontSize', 13, 'FontWeight', 'bold');
        
        % 加粗座標軸數字
        set(gca, 'FontWeight', 'bold', 'FontSize', 14);
        
        h_legend = legend('VM', 'DA', 'Location', 'northwest', 'FontSize', 11, 'FontWeight', 'bold');
        h_legend.LineWidth = 2.5;  % 圖例線條加粗
        grid on;
        hold off;
        
        if i == length(channels)
            xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end

%% ===== 繪製指定週期的VM vs VD相位圖 =====
function plot_phase_range(vm_data, vd_data, target_freq, sampling_rate, start_period, channels)
    % 繪製指定週期的VM vs VD相位圖
    
    period_samples = round(sampling_rate / target_freq);
    start_idx = (start_period - 1) * period_samples + 1;
    end_idx = start_period * period_samples;
    
    if end_idx > size(vm_data, 2)
        fprintf('警告: 指定週期超出數據範圍\n');
        return;
    end
    
    figure('Name', sprintf('VM vs VD Phase Plot - Period %d', start_period));
    
    colors = ['b', 'r', 'g', 'm', 'c', 'k'];
    legend_labels = {};
    
    for i = 1:length(channels)
        ch = channels(i);
        
        vm_period = vm_data(ch, start_idx:end_idx);
        vd_period = vd_data(ch, start_idx:end_idx);
        
        plot(vd_period, vm_period, 'Color', colors(mod(i-1, 6)+1), 'LineWidth', 2.5);
        legend_labels{i} = sprintf('Ch%d', ch);
        hold on;
    end
    
    xlabel('VD Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('VM Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('VM vs VD Phase Plot - Period %d', start_period), 'FontSize', 14, 'FontWeight', 'bold');
    
    % 加粗座標軸數字
    set(gca, 'FontWeight', 'bold', 'FontSize', 14);
    
    h_legend = legend(legend_labels, 'Location', 'northwest', 'FontSize', 11, 'FontWeight', 'bold');
    h_legend.LineWidth = 2.5;  % 圖例線條加粗
    grid on;
    axis equal;
    hold off;
end
%% ===== 使用範例（乾淨數據版本）=====

% 範例1: 基本使用（完整流程）
% main();

% 範例2: 手動處理流程
% [vm_raw, vd_raw, da_raw] = load_csv_data('01Data/02Processed_csv/0805_B_data/1_1.csv');
% [vm, vd, da] = clean_all_data(vm_raw, vd_raw, da_raw);
% da_volt = dac_to_voltage(da);
% steady = detect_steady_state_clean(vm, 1);
% plot_clean_signals(vm, 'VM', steady, 1, 10, [1,3,5]);

% 範例3: 提取穩態數據進行自定義分析
% periods = 5;
% start_idx = steady.index;
% end_idx = start_idx + periods * steady.period_samples - 1;
% vm_stable = vm(:, start_idx:end_idx);  % 提取VM穩態數據
% vd_stable = vd(:, start_idx:end_idx);  % 提取VD穩態數據

% 範例4: 單通道分析
% channel = 3;
% vm_ch3 = vm(channel, start_idx:end_idx);
% time_axis = (0:length(vm_ch3)-1) / 100000;
% figure; plot(time_axis, vm_ch3); title('VM Channel 3 Analysis');

% 範例5: VM vs VD相位圖
% plot_vm_vd_phase_diagram(vm, vd, steady, 1);  % 所有通道的相位圖
% plot_vm_vd_phase_diagram(vm, vd, steady, 1, [1,3,5]);  % 只顯示通道1,3,5

% 範例6: 一鍵式整合處理
% [vm, vd, da_volt, steady] = load_and_process_hsdata('01Data/02Processed_csv/0811_fd/10.csv', 10);
% [vm, vd, da_volt, steady] = load_and_process_hsdata('your_file.csv', 5, 'threshold', 1e-3);

% 範例7: 週期平均分析
% period_data = extract_period_average(vm, steady, 10, 3);  % 通道3的10週期平均

%% ===== 實用工具函數(各函數皆可分開使用) =====
function show_data_info(vm_data, vd_data, da_data)
    % 顯示數據基本信息
    
    fprintf('\n=== 數據信息 ===\n');
    fprintf('數據長度: %d 樣本點\n', size(vm_data, 2));
    fprintf('採樣時間: %.3f 秒\n', size(vm_data, 2) / 100000);
    
    fprintf('\nVM數據範圍:\n');
    for i = 1:6
        fprintf('  Ch%d: %.6f ~ %.6f V\n', i, min(vm_data(i, :)), max(vm_data(i, :)));
    end
    
    fprintf('\nVD數據範圍:\n');
    for i = 1:6
        fprintf('  Ch%d: %.6f ~ %.6f V\n', i, min(vd_data(i, :)), max(vd_data(i, :)));
    end
    
    fprintf('\nDA數據範圍 (DAC值):\n');
    for i = 1:6
        fprintf('  Ch%d: %d ~ %d\n', i, round(min(da_data(i, :))), round(max(da_data(i, :))));
    end
end

function [vm, vd, da_volt, steady_info] = load_and_process_hsdata(csv_filepath, target_freq, varargin)
    % 整合HSData讀取、清理、轉換和穩態檢測的完整流程
    %
    % 輸入:
    %   csv_filepath - CSV檔案路徑
    %   target_freq - 目標頻率 (Hz)
    %   可選參數: 穩態檢測參數（與detect_steady_state_clean相同）
    % 輸出:
    %   vm, vd - 乾淨的VM/VD數據 (6 x N)
    %   da_volt - 轉換為電壓的DA數據 (6 x N)
    %   steady_info - 穩態檢測結果
    
    fprintf('=== 整合處理HSData ===\n');
    fprintf('檔案: %s\n', csv_filepath);
    fprintf('頻率: %d Hz\n', target_freq);
    
    % 步驟1: 讀取原始數據
    fprintf('步驟1: 讀取CSV數據...\n');
    [vm_raw, vd_raw, da_raw] = load_csv_data(csv_filepath);
    
    % 步驟2: 清理數據
    fprintf('步驟2: 清理數據...\n');
    [vm, vd, da] = clean_all_data(vm_raw, vd_raw, da_raw);
    
    % 步驟3: DAC轉電壓
    fprintf('步驟3: DAC轉電壓...\n');
    da_volt = dac_to_voltage(da);
    
    % 步驟4: 穩態檢測
    fprintf('步驟4: 穩態檢測...\n');
    steady_info = detect_steady_state_clean(vm, target_freq, varargin{:});
    
    if isempty(steady_info)
        warning('穩態檢測失敗');
    else
        fprintf('穩態檢測成功: 第%d週期\n', steady_info.period);
    end
    
    fprintf('處理完成！\n');
end

function period_avg = extract_period_average(clean_data, steady_info, num_periods, channel)
    % 提取特定通道的週期平均波形
    period_samples = steady_info.period_samples;
    start_idx = steady_info.index;
    
    period_matrix = zeros(num_periods, period_samples);
    for p = 1:num_periods
        period_start = start_idx + (p-1) * period_samples;
        period_end = period_start + period_samples - 1;
        if period_end <= size(clean_data, 2)
            period_matrix(p, :) = clean_data(channel, period_start:period_end);
        end
    end
    
    period_avg = mean(period_matrix, 1);
end

function stable_data = extract_stable_data(clean_data, steady_info, num_periods, channels)
    % 提取指定通道的穩態數據
    start_idx = steady_info.index;
    period_samples = steady_info.period_samples;
    end_idx = start_idx + num_periods * period_samples - 1;
    
    stable_data = clean_data(channels, start_idx:end_idx);
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