% test_steady_detection.m - 測試新的穩態檢測功能
%
% 此腳本用於比較原始和改進版穩態檢測的差異

fprintf('=== 穩態檢測功能測試 ===\n');
fprintf('這個測試會比較簡化版和進階版穩態檢測的結果\n\n');

% 設定測試參數
test_folder = '01Data\02Processed_csv\openloop_Cali_P5';
test_file = 'P5_1.csv';  % 請根據實際檔案調整

% 檢查檔案是否存在
full_path = fullfile(test_folder, test_file);
if ~exist(full_path, 'file')
    fprintf('警告: 測試檔案不存在: %s\n', full_path);
    fprintf('請修改 test_file 變數以指向實際存在的CSV檔案\n');

    % 嘗試列出資料夾中的檔案
    csv_files = dir(fullfile(test_folder, '*.csv'));
    if ~isempty(csv_files)
        fprintf('\n可用的CSV檔案:\n');
        for i = 1:min(5, length(csv_files))
            fprintf('  - %s\n', csv_files(i).name);
        end
        fprintf('\n請選擇其中一個檔案進行測試\n');
    end
    return;
end

fprintf('測試檔案: %s\n', full_path);

% 讀取測試數據
fprintf('\n1. 讀取數據...\n');
raw_data = readtable(full_path);
data_length = height(raw_data);
fprintf('   數據長度: %d 點\n', data_length);

% 提取VM和DA數據
vm_data = zeros(6, data_length);
da_data = zeros(6, data_length);

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

% 清理數據
fprintf('\n2. 清理數據...\n');
exclude_indices = 1:10000:data_length;
valid_mask = true(1, data_length);
valid_mask(exclude_indices) = false;
vm_clean = vm_data(:, valid_mask);
da_clean = da_data(:, valid_mask);
fprintf('   排除 %d 個異常點\n', length(exclude_indices));
fprintf('   清理後數據長度: %d 點\n', size(vm_clean, 2));

% DA轉電壓並檢測激勵頻率
da_volt = (da_clean - 32768) * (20.0 / 65536);

% 簡單的頻率檢測
fprintf('\n3. 檢測激勵頻率...\n');
best_ch = 0;
max_energy = 0;
best_freq = 0;

for ch = 1:6
    signal = da_volt(ch, :);
    energy = sqrt(mean(signal.^2));

    if energy > 0.1
        N = length(signal);
        fft_result = fft(signal);
        freq_axis = (0:N-1) * 100000 / N;

        positive_freqs = freq_axis(2:floor(N/2));
        positive_fft = abs(fft_result(2:floor(N/2)));

        [max_amp, max_idx] = max(positive_fft);
        freq = positive_freqs(max_idx);

        if max_amp > max_energy
            max_energy = max_amp;
            best_ch = ch;
            best_freq = freq;
        end
    end
end

fprintf('   激勵通道: DA%d\n', best_ch);
fprintf('   激勵頻率: %.2f Hz\n', best_freq);

% 測試簡化版穩態檢測
fprintf('\n4. 測試簡化版穩態檢測...\n');
tic;
steady_simple = detect_steady_state_simple(vm_clean(1,:), best_freq, 100000);
time_simple = toc;

if ~isempty(steady_simple)
    fprintf('   ✓ 檢測成功\n');
    fprintf('   穩態週期: %d\n', steady_simple.period);
    fprintf('   穩態索引: %d\n', steady_simple.index);
    fprintf('   執行時間: %.3f 秒\n', time_simple);
else
    fprintf('   ✗ 檢測失敗\n');
end

% 測試進階版穩態檢測
fprintf('\n5. 測試進階版穩態檢測...\n');
tic;
steady_advanced = detect_steady_state_advanced(vm_clean, best_freq, 100000, 1e-3);
time_advanced = toc;

if ~isempty(steady_advanced)
    fprintf('   ✓ 檢測成功\n');
    fprintf('   穩態週期: %d\n', steady_advanced.period);
    fprintf('   穩態索引: %d\n', steady_advanced.index);
    fprintf('   執行時間: %.3f 秒\n', time_advanced);
else
    fprintf('   ✗ 檢測失敗\n');
end

% 比較結果
fprintf('\n6. 結果比較:\n');
fprintf('   =====================================\n');
fprintf('                簡化版    進階版\n');
fprintf('   =====================================\n');
if ~isempty(steady_simple) && ~isempty(steady_advanced)
    fprintf('   穩態週期:     %4d      %4d\n', steady_simple.period, steady_advanced.period);
    fprintf('   穩態索引:   %6d    %6d\n', steady_simple.index, steady_advanced.index);
    fprintf('   執行時間:    %.3fs    %.3fs\n', time_simple, time_advanced);

    period_diff = abs(steady_advanced.period - steady_simple.period);
    fprintf('   =====================================\n');
    fprintf('   週期差異: %d 個週期\n', period_diff);

    if period_diff > 0
        fprintf('\n   ⚠ 注意: 進階版檢測到更精確的穩態起始點\n');
        if steady_advanced.period > steady_simple.period
            fprintf('   進階版更保守（較晚進入穩態）\n');
        else
            fprintf('   進階版更早檢測到穩態\n');
        end
    else
        fprintf('\n   ✓ 兩種方法得到相同結果\n');
    end
end

% 視覺化比較（可選）
if ~isempty(steady_simple) && ~isempty(steady_advanced)
    fprintf('\n7. 是否要顯示視覺化比較? (y/n): ');
    user_input = input('', 's');

    if strcmpi(user_input, 'y')
        figure('Name', '穩態檢測比較', 'Position', [100, 100, 1200, 600]);

        % 選擇要顯示的通道
        display_ch = 1;
        signal = vm_clean(display_ch, :);
        time_axis = (0:length(signal)-1) / 100000;

        % 繪製信號
        plot(time_axis, signal, 'b-', 'LineWidth', 0.5);
        hold on;

        % 標記簡化版穩態點
        simple_time = (steady_simple.index - 1) / 100000;
        xline(simple_time, 'g--', 'LineWidth', 2, 'Label', '簡化版穩態');

        % 標記進階版穩態點
        advanced_time = (steady_advanced.index - 1) / 100000;
        xline(advanced_time, 'r--', 'LineWidth', 2, 'Label', '進階版穩態');

        xlabel('時間 (秒)');
        ylabel('電壓 (V)');
        title(sprintf('穩態檢測比較 - VM通道%d', display_ch));
        legend('VM信號', '簡化版', '進階版', 'Location', 'best');
        grid on;

        % 放大穩態區域
        zoom_start = min(simple_time, advanced_time) - 0.1;
        zoom_end = max(simple_time, advanced_time) + 0.5;
        xlim([max(0, zoom_start), min(time_axis(end), zoom_end)]);

        fprintf('   圖形已顯示\n');
    end
end

fprintf('\n=== 測試完成 ===\n');