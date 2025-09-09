% jump.m - 簡化HSData分析與視覺化工具
% 
% 使用方法：修改下方參數設定區，然後直接執行 jump

%% ===== 參數設定區 (在此修改您的設定) =====
CSV_FILE = '500_jump.csv';          % CSV檔案名稱
TARGET_FREQ = 500;                  % 目標頻率 (Hz)
SAMPLING_RATE = 100000;             % 採樣頻率 (Hz)

% 顯示設定
CHANNELS_TO_SHOW = [5];         % 要顯示的通道 (例: [1,3,5] 或 1:6)
START_PERIOD = 1;               % 開始週期 (從第幾個週期開始)
END_PERIOD = 25;                % 結束週期 (到第幾個週期結束)

% 要產生的圖表類型 (設為 1 開啟，0 關閉)
SHOW_VM = 0;                        % VM電壓圖表
SHOW_VD = 0;                        % VD電壓圖表  
SHOW_DA = 0;                        % DA電壓圖表
SHOW_VM_DA_OVERLAY = 1;             % VM與DA疊圖
SHOW_VM_VS_VD_PHASE = 1;            % VM vs VD

%% ===== 執行主程序 =====
run_analysis();

%% ===== 主要執行函數 =====

function run_analysis()
    % 簡化的主執行函數
    
    % 讀取腳本級別的參數
    CSV_FILE = evalin('base', 'CSV_FILE');
    TARGET_FREQ = evalin('base', 'TARGET_FREQ');
    SAMPLING_RATE = evalin('base', 'SAMPLING_RATE');
    CHANNELS_TO_SHOW = evalin('base', 'CHANNELS_TO_SHOW');
    START_PERIOD = evalin('base', 'START_PERIOD');
    END_PERIOD = evalin('base', 'END_PERIOD');
    SHOW_VM = evalin('base', 'SHOW_VM');
    SHOW_VD = evalin('base', 'SHOW_VD');
    SHOW_DA = evalin('base', 'SHOW_DA');
    SHOW_VM_DA_OVERLAY = evalin('base', 'SHOW_VM_DA_OVERLAY');
    SHOW_VM_VS_VD_PHASE = evalin('base', 'SHOW_VM_VS_VD_PHASE');
    
    clc;
    fprintf('=== JUMP - HSData Analysis Tool ===\n');
    fprintf('File: %s | Freq: %d Hz | Period: %d-%d\n', CSV_FILE, TARGET_FREQ, START_PERIOD, END_PERIOD);
    
    % 1. 讀取和處理數據
    [vm_clean, vd_clean, da_clean] = load_and_clean_data(CSV_FILE);
    da_voltage = dac_to_voltage(da_clean);
    
    % 2. 計算週期參數
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
    
    % 3. 產生圖表
    if SHOW_VM
        plot_signal_range(vm_clean, 'VM', TARGET_FREQ, SAMPLING_RATE, START_PERIOD, END_PERIOD, CHANNELS_TO_SHOW);
    end
    if SHOW_VD
        plot_signal_range(vd_clean, 'VD', TARGET_FREQ, SAMPLING_RATE, START_PERIOD, END_PERIOD, CHANNELS_TO_SHOW);
    end
    if SHOW_DA
        plot_signal_range(da_voltage, 'DA', TARGET_FREQ, SAMPLING_RATE, START_PERIOD, END_PERIOD, CHANNELS_TO_SHOW);
    end
    if SHOW_VM_DA_OVERLAY
        plot_overlay_range(vm_clean, da_voltage, TARGET_FREQ, SAMPLING_RATE, START_PERIOD, END_PERIOD, CHANNELS_TO_SHOW);
    end
    if SHOW_VM_VS_VD_PHASE
        plot_phase_range(vm_clean, vd_clean, TARGET_FREQ, SAMPLING_RATE, START_PERIOD, CHANNELS_TO_SHOW);
    end
    
    fprintf('Analysis completed!\n');
end

function [vm_clean, vd_clean, da_clean] = load_and_clean_data(csv_filepath)
    % 簡化的數據讀取和清理函數
    
    % 讀取CSV檔案
    if ~exist(csv_filepath, 'file')
        error('檔案不存在: %s', csv_filepath);
    end
    
    raw_data = readtable(csv_filepath);
    data_length = height(raw_data);
    
    fprintf('讀取CSV檔案: %d筆記錄\n', data_length);
    
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
        end
        if ismember(vd_col, raw_data.Properties.VariableNames)
            vd_data(i, :) = raw_data.(vd_col);
        end
        if ismember(da_col, raw_data.Properties.VariableNames)
            da_data(i, :) = raw_data.(da_col);
        end
    end
    
    % 簡化清理：排除每10000個樣本點
    exclude_indices = 1:10000:data_length;
    valid_mask = true(1, data_length);
    valid_mask(exclude_indices) = false;
    
    vm_clean = vm_data(:, valid_mask);
    vd_clean = vd_data(:, valid_mask);
    da_clean = da_data(:, valid_mask);
    
    fprintf('清理後數據長度: %d\n', sum(valid_mask));
end

function voltage = dac_to_voltage(dac_value)
    % 將16位DAC值轉換為±10V電壓
    voltage = (dac_value - 32768) * (20.0 / 65536);
end

%% ===== 基於週期範圍的繪圖函數 =====

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

%% ===== 使用範例 =====

% 修改頂部的參數設定，然後執行：
% jump