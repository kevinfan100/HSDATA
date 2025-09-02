% HSDataTemplate.m - VM/VD/DA數據處理基礎模板
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

%% ===== 主要功能函數 =====

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

%% ===== 工具函數 =====

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

%% ===== 實用工具函數 =====

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