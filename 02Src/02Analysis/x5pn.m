
main()

function main()
    clear; clc; close all;

    [vm_raw, vd_raw, da_raw] = load_csv_data('01Data/02Processed_csv/0811_fd/10_x5p.csv');
    [vm, vd, da] = clean_all_data(vm_raw, vd_raw, da_raw);
    da_volt = dac_to_voltage(da);
    steady = detect_steady_state_clean(vm, 10);
    

    plot_VmVd(vm, vd, steady, 10);

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
    
    fprintf('排除%d個異常點，長度: %d\n', length(exclude_indices), sum(valid_mask));
end

function steady_info = detect_steady_state_clean(vm_clean, target_freq, varargin)
    % 輸入:
    %   vm_clean - 已清理的VM數據 (6 x N)
    %   target_freq - 目標頻率 (Hz)
    %   可選參數
    
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
        
        fprintf('穩態檢測成功: 第%d週期，數據索引%d\n', recommended_period, clean_index);
    else
        steady_info = [];
        fprintf('穩態檢測失敗: 未找到穩定週期\n');
    end
end


function plot_VmVd(vm_clean, vd_clean, steady_info, target_freq, channels)
    % 繪製VM vs VD（一個週期的疊圖）
    %
    % 輸入:
    %   vm_clean - VM數據 (6 x N)
    %   vd_clean - VD數據 (6 x N)
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
    figure('Name', sprintf('VM vs VD - %dHz (1個週期)', target_freq), ...
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
    end
    
    % 設定圖形屬性
    xlabel('VD  (V)');
    ylabel('VM  (V)');
    title(sprintf('VM vs VD  - %dHz ', target_freq));
    legend('Location', 'best');
    grid on;
    axis equal;  % 保持軸比例相等
    hold off;

    
    fprintf('已顯示VM vs VD: %d個通道，1個週期\n', length(channels));
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
        fprintf('警告: 數據不足，實際顯示%d個週期\n', actual_periods);
    end
    
    % 提取顯示數據
    display_data = clean_data(channels, start_idx:end_idx);
    time_axis = (0:(end_idx-start_idx)) / 100000;  % 轉換為秒
    
    % 創建圖形
    figure('Name', sprintf('%s時域響應 - %dHz', data_type, target_freq), ...
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
    xlabel('Time (秒)');
    ylabel('Vol (V)');

    
    title(sprintf('%s時域響應 - %dHz (%d個週期)', data_type, target_freq, display_periods));
    legend('Location', 'best');
    grid on;
    hold off;
    
    fprintf('已顯示%s數據: %d個通道，%d個週期\n', data_type, length(channels), display_periods);
end