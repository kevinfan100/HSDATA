% plot_vm_da_overlay.m - VM與DA疊圖分析（基於HSDataTemplate.m）
%
% 主要功能：
% 1. 讀取0_10.csv中的VM和DA數據
% 2. DA0轉換為電流
% 3. 穩態檢測
% 4. 繪製6個通道的VM與DA0一個週期疊圖
%
% 使用方法：
% 1. 執行plot_vm_da_main()函數

%% ===== 參數設定區 =====
% 數據路徑設定
DATA_FOLDER = '01Data/02Processed_csv/0805_B_data/';  % 數據資料夾路徑
CSV_FILE = '0_10.csv';                               % 要處理的CSV檔案

% 系統參數
SAMPLING_RATE = 100000;                              % 採樣頻率 (Hz)
TARGET_FREQ = 10;                                    % 目標頻率 (Hz)

% 穩態檢測參數
CONSECUTIVE_PERIODS = 2;                             % 連續穩定週期數
CHECK_POINTS = 5;                                    % 每週期檢查點數
STABILITY_THRESHOLD = 1e-3;                          % 穩定性閾值
START_PERIOD = 1;                                    % 開始檢測的週期數

% DA0轉電流參數
DA_TO_CURRENT_FACTOR = 0.3;                         % DA到電流轉換係數 (A/V)

%% ===== 主要執行函數 =====


function [vm_data, da_data] = load_csv_data(csv_filepath)
    % 讀取CSV檔案並分離VM、DA數據
    
    % 確保檔案路徑正確 - 直接使用傳入的路徑
    
    if ~exist(csv_filepath, 'file')
        error('檔案不存在: %s', csv_filepath);
    end
    
    % 讀取CSV檔案 - 指定編碼以處理BOM
    try
        raw_data = readtable(csv_filepath, 'Encoding', 'UTF-8');
    catch
        raw_data = readtable(csv_filepath);
    end
    data_length = height(raw_data);
    
    fprintf('成功讀取CSV檔案，共%d筆記錄\n', data_length);
    
    % 初始化數據矩陣
    vm_data = zeros(6, data_length);
    da_data = zeros(6, data_length);
    
    % 提取各通道數據
    for i = 1:6
        vm_col = sprintf('vm_%d', i-1);
        da_col = sprintf('da_%d', i-1);
        
        if ismember(vm_col, raw_data.Properties.VariableNames)
            vm_data(i, :) = raw_data.(vm_col);
        else
            warning('找不到欄位: %s', vm_col);
        end
        
        if ismember(da_col, raw_data.Properties.VariableNames)
            da_data(i, :) = raw_data.(da_col);
        else
            warning('找不到欄位: %s', da_col);
        end
    end
end

function voltage = dac_to_voltage(dac_value)
    % 將16位DAC值轉換為±10V電壓
    voltage = (dac_value - 32768) * (20.0 / 65536);
end

function current = voltage_to_current(voltage)
    % 將電壓轉換為電流
    DA_TO_CURRENT_FACTOR = 0.3;  % DA到電流轉換係數 (A/V)
    current = voltage * DA_TO_CURRENT_FACTOR;
end

function [vm_clean, da_clean] = clean_vm_da_data(vm_raw, da_raw)
    % 統一清理VM和DA數據，排除異常點
    
    data_length = size(vm_raw, 2);
    
    % 排除每10000個樣本點（避免採集異常）
    exclude_indices = 1:10000:data_length;
    valid_mask = true(1, data_length);
    valid_mask(exclude_indices) = false;
    
    % 統一應用遮罩到所有數據
    vm_clean = vm_raw(:, valid_mask);
    da_clean = da_raw(:, valid_mask);
    
    fprintf('排除%d個異常點，乾淨數據長度: %d\n', length(exclude_indices), sum(valid_mask));
end

function steady_info = detect_steady_state_clean(vm_clean, target_freq)
    % 在乾淨數據上進行穩態檢測
    
    clean_length = size(vm_clean, 2);
    
    % 計算週期相關參數
    SAMPLING_RATE = 100000;  % 採樣頻率 (Hz)
    CHECK_POINTS = 5;       % 每週期檢查點數
    CONSECUTIVE_PERIODS = 2; % 連續穩定週期數
    STABILITY_THRESHOLD = 1e-3; % 穩定性閾值
    START_PERIOD = 1;       % 開始檢測的週期數
    
    period_samples = round(SAMPLING_RATE / target_freq);
    max_periods = floor(clean_length / period_samples);
    check_positions = round(linspace(1, period_samples, CHECK_POINTS));
    
    fprintf('週期樣本數: %d，最大週期數: %d\n', period_samples, max_periods);
    
    steady_periods = [];
    
    % 對每個VM通道進行穩態檢測
    for vm_ch = 1:6
        signal = vm_clean(vm_ch, :);
        
        % 測試從START_PERIOD開始的每個週期
        for test_period = START_PERIOD:(max_periods - CONSECUTIVE_PERIODS)
            all_stable = true;
            
            % 檢查連續週期的穩定性
            for i = 1:CONSECUTIVE_PERIODS
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
                if max_diff >= STABILITY_THRESHOLD
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
    
    % 選擇最保守的穩態點
    if ~isempty(steady_periods)
        recommended_period = max(steady_periods);
        clean_index = recommended_period * period_samples + 1;
        
        steady_info = struct(...
            'period', recommended_period, ...
            'index', clean_index, ...
            'max_periods', max_periods, ...
            'period_samples', period_samples);
        
        fprintf('穩態檢測成功: 第%d週期，索引%d\n', recommended_period, clean_index);
    else
        steady_info = [];
        fprintf('穩態檢測失敗: 未找到穩定週期\n');
    end
end

function plot_vm_da0_overlay(vm_clean, da0_current, steady_info, target_freq)
    % 繪製六個通道VM與DA0電流的疊圖（一個週期）
    
    if isempty(steady_info)
        fprintf('無法繪圖: 穩態檢測失敗\n');
        return;
    end
    
    % 計算一個週期的數據範圍
    period_samples = steady_info.period_samples;
    start_idx = steady_info.index;
    end_idx = start_idx + period_samples - 1;
    
    % 檢查數據範圍
    if end_idx > size(vm_clean, 2)
        fprintf('警告: 數據不足，無法提取完整週期\n');
        return;
    end
    
    % 提取一個週期的數據
    vm_period = vm_clean(:, start_idx:end_idx);
    da0_current_period = da0_current(start_idx:end_idx);
    
    % 建立時間軸
    SAMPLING_RATE = 100000;  % 採樣頻率 (Hz)
    time_axis = (0:(period_samples-1)) / SAMPLING_RATE;
    
    % 創建圖形 - 2x3 子圖布局
    figure('Name', sprintf('VM與DA0疊圖分析 - %dHz (1個週期)', target_freq), ...
           'Position', [100, 100, 1400, 1000]);
    
    % 繪製六個通道的子圖
    for ch = 1:6
        subplot(2, 3, ch);
        
        % 繪製VM和DA0電流
        yyaxis left
        plot(time_axis * 1000, vm_period(ch, :), 'b-', 'LineWidth', 2, 'DisplayName', sprintf('VM_%d', ch-1));
        ylabel('VM 電壓 (V)', 'Color', 'b');
        ylim_vm = ylim;
        
        yyaxis right
        plot(time_axis * 1000, da0_current_period, 'r-', 'LineWidth', 1.5, 'DisplayName', 'DA0 電流');
        ylabel('DA0 電流 (A)', 'Color', 'r');
        ylim_da = ylim;
        
        % 設定標題和格式
        title(sprintf('Channel %d - VM與DA0疊圖', ch));
        xlabel('時間 (ms)');
        grid on;
        
        % 添加圖例
        legend('Location', 'best');
        
        % 設定軸顏色
        ax = gca;
        ax.YAxis(1).Color = 'b';
        ax.YAxis(2).Color = 'r';
    end
    
    % 添加總標題
    sgtitle(sprintf('VM與DA0電流疊圖分析 - %dHz頻率 (穩態後1個週期)', target_freq), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % 顯示數據信息
    fprintf('已顯示VM與DA0疊圖: 6個通道，1個週期 (%.1f ms)\n', ...
            (1/target_freq)*1000);
    fprintf('數據範圍: VM (%.6f ~ %.6f V), DA0電流 (%.6f ~ %.6f A)\n', ...
            min(vm_period(:)), max(vm_period(:)), ...
            min(da0_current_period), max(da0_current_period));
end

%% ===== 主要執行區 =====

% 執行分析
clear; clc;

% 參數定義
DATA_FOLDER = '';  % 當前目錄
CSV_FILE = '0_10.csv';
TARGET_FREQ = 10;

fprintf('=== VM與DA疊圖分析 ===\n');
fprintf('處理檔案: %s\n', [DATA_FOLDER CSV_FILE]);
fprintf('目標頻率: %d Hz\n', TARGET_FREQ);

% 1. 讀取原始CSV數據
fprintf('\n步驟1: 讀取原始CSV數據...\n');
[vm_raw, da_raw] = load_csv_data([DATA_FOLDER CSV_FILE]);

% 2. 清理數據
fprintf('步驟2: 清理數據...\n');
[vm_clean, da_clean] = clean_vm_da_data(vm_raw, da_raw);

% 3. DA0轉電壓再轉電流
fprintf('步驟3: 轉換DA0為電流...\n');
da0_voltage = dac_to_voltage(da_clean(1, :));  % 只處理DA0
da0_current = voltage_to_current(da0_voltage);

% 4. 穩態檢測
fprintf('步驟4: 穩態檢測...\n');
steady_info = detect_steady_state_clean(vm_clean, TARGET_FREQ);

if isempty(steady_info)
    fprintf('錯誤: 未檢測到穩態\n');
    return;
end

fprintf('檢測到穩態: 第%d週期，索引%d\n', steady_info.period, steady_info.index);

% 5. 繪製VM與DA0疊圖（六個通道分開）
fprintf('步驟5: 繪製VM與DA0疊圖...\n');
plot_vm_da0_overlay(vm_clean, da0_current, steady_info, TARGET_FREQ);

fprintf('分析完成！\n');