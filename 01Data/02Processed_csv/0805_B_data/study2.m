% STUDY2.M - Transfer function analysis and B-matrix calculation
% Converted from study2.py with improved readability and MATLAB optimization

function main()
    % Main execution function
    csv_files = {'0_10.csv', '1_10.csv', '2_10.csv', '3_10.csv', '4_10.csv', '5_10.csv'};
    
    % Steady state detection parameters
    steady_params = struct('consecutive_periods', 2, ...
                          'check_points', 5, ...
                          'threshold', 2e-3);
    
    % Calculate B-matrix and plot specified file
    result = calculate_b_matrix_integrated(csv_files, 10, 5, steady_params, '0_10.csv');
end

function voltage = dac_to_voltage(dac_value)
    % Convert 16-bit DAC code to voltage
    voltage = (dac_value - 32768) * (20.0 / 65536);
end

function data = load_single_test(csv_file)
    % Load single test data from CSV file
    raw_data = readtable(csv_file);
    
    % Extract VM and DA data (6 channels each)
    vm_data = zeros(6, height(raw_data));
    da_data = zeros(6, height(raw_data));
    
    for i = 1:6
        vm_col = sprintf('vm_%d', i-1);
        da_col = sprintf('da_%d', i-1);
        vm_data(i, :) = raw_data.(vm_col);
        da_data(i, :) = raw_data.(da_col);
    end
    
    data = struct('vm', vm_data, 'da', da_data);
end

function excited_ch = find_excitation_channel(da_data)
    % Find the channel with maximum RMS value
    rms_values = sqrt(mean(da_data.^2, 2));
    [~, excited_ch] = max(rms_values);
end

function steady_info = detect_steady_state_by_periods(csv_file, target_freq, varargin)
    % Detect steady state by checking consecutive periods
    
    % Parse input arguments
    p = inputParser;
    addParameter(p, 'sampling_rate', 100000);
    addParameter(p, 'start_period', 1);
    addParameter(p, 'consecutive_periods', 3);
    addParameter(p, 'check_points', 10);
    addParameter(p, 'threshold', 1e-3);
    parse(p, varargin{:});
    
    params = p.Results;
    
    % Load and preprocess data
    raw_data = readtable(csv_file);
    
    % Exclude every 10000th sample
    exclude_indices = 1:10000:height(raw_data);
    valid_mask = true(height(raw_data), 1);
    valid_mask(exclude_indices) = false;
    
    valid_data = raw_data(valid_mask, :);
    original_indices = find(valid_mask);
    
    % Extract VM data
    vm_data = zeros(6, height(valid_data));
    for i = 1:6
        vm_col = sprintf('vm_%d', i-1);
        vm_data(i, :) = valid_data.(vm_col);
    end
    
    period_samples = round(params.sampling_rate / target_freq);
    max_periods = floor(height(valid_data) / period_samples);
    check_positions = round(linspace(1, period_samples, params.check_points));
    
    steady_periods = [];
    
    for vm_ch = 1:6
        signal = vm_data(vm_ch, :);
        
        for test_period = params.start_period:(max_periods - params.consecutive_periods)
            all_stable = true;
            
            for i = 1:params.consecutive_periods
                current_period = test_period + i - 1;
                next_period = current_period + 1;
                
                current_start = current_period * period_samples + 1;
                next_start = next_period * period_samples + 1;
                
                max_diff = 0;
                for pos = check_positions
                    if current_start + pos - 1 <= length(signal) && next_start + pos - 1 <= length(signal)
                        current_val = signal(current_start + pos - 1);
                        next_val = signal(next_start + pos - 1);
                        diff = abs(current_val - next_val);
                        max_diff = max(max_diff, diff);
                    end
                end
                
                if max_diff >= params.threshold
                    all_stable = false;
                    break;
                end
            end
            
            if all_stable
                steady_periods(end+1) = test_period;
                break;
            end
        end
    end
    
    if ~isempty(steady_periods)
        recommended_period = max(steady_periods);
        valid_index = recommended_period * period_samples + 1;
        original_index = original_indices(valid_index);
        
        steady_info = struct('period', recommended_period, ...
                           'index', original_index, ...
                           'max_periods', max_periods);
    else
        steady_info = [];
    end
end

function [signed_gain, magnitude, phase_deg, is_valid] = calculate_transfer_function_with_validation(...
    input_signal, output_signal, fs, target_freq, tolerance_percent)
    % Calculate transfer function with frequency validation
    
    % Remove DC component (commented out to match Python version)
    input_clean = input_signal; % - mean(input_signal);
    output_clean = output_signal; % - mean(output_signal);
    
    % FFT calculation
    input_fft = fft(input_clean);
    output_fft = fft(output_clean);
    freqs = (0:length(input_clean)-1) * fs / length(input_clean);
    
    % Positive frequency range
    positive_mask = freqs > 0;
    positive_freqs = freqs(positive_mask);
    positive_input_fft = input_fft(positive_mask);
    positive_output_fft = output_fft(positive_mask);
    
    % Define tolerance range
    tolerance = target_freq * tolerance_percent / 100;
    freq_range_min = target_freq - tolerance;
    freq_range_max = target_freq + tolerance;
    
    % Find maximum energy point within tolerance
    freq_mask = (positive_freqs >= freq_range_min) & (positive_freqs <= freq_range_max);
    
    if ~any(freq_mask)
        fprintf('    ✗ No frequency points in tolerance range [%.2f, %.2f] Hz\n', ...
                freq_range_min, freq_range_max);
        signed_gain = 0; magnitude = 0; phase_deg = 0; is_valid = false;
        return;
    end
    
    % Find maximum energy point within range
    vm_power = abs(positive_output_fft).^2;
    masked_indices = find(freq_mask);
    masked_powers = vm_power(masked_indices);
    
    [~, local_max_idx] = max(masked_powers);
    global_max_idx = masked_indices(local_max_idx);
    
    % Get information at maximum energy point
    actual_freq = positive_freqs(global_max_idx);
    max_power = masked_powers(local_max_idx);
    
    % Validate this is the global maximum
    global_max_power = max(vm_power);
    power_ratio = max_power / global_max_power;
    
    % Calculate transfer function at maximum energy point
    input_complex = positive_input_fft(global_max_idx);
    output_complex = positive_output_fft(global_max_idx);
    H_complex = output_complex / input_complex;
    
    signed_gain = real(H_complex);
    magnitude = abs(H_complex);
    phase_deg = angle(H_complex) * 180 / pi;
    
    % Validation
    freq_error = abs(actual_freq - target_freq);
    freq_error_percent = freq_error / target_freq * 100;
    
    is_valid = (freq_error <= tolerance) && (power_ratio > 0.5);
    
    fprintf('    Target %.0fHz±%d%% → Max energy @%.2fHz\n', ...
            target_freq, tolerance_percent, actual_freq);
    fprintf('    Error %.1f%%, Energy ratio %.3f\n', ...
            freq_error_percent, power_ratio);
end

function plot_frequency_analysis(csv_file, target_freq, steady_info, tolerance_percent)
    % Plot frequency domain analysis
    
    data = load_single_test(csv_file);
    da_voltage = dac_to_voltage(data.da);
    excited_ch = find_excitation_channel(da_voltage);
    
    % Calculate data range after steady state
    period_samples = round(100000 / target_freq);
    steady_period = steady_info.period;
    max_periods = steady_info.max_periods;
    available_periods = max_periods - steady_period;
    
    start_idx = steady_info.index;
    end_idx = start_idx + available_periods * period_samples - 1;
    
    input_signal = da_voltage(excited_ch, start_idx:end_idx);
    
    fprintf('Plotting: %d complete periods after period %d\n', ...
            available_periods, steady_period);
    
    % Preprocessing and FFT
    input_clean = input_signal; % - mean(input_signal);
    input_fft = fft(input_clean);
    freqs = (0:length(input_clean)-1) * 100000 / length(input_clean);
    
    positive_mask = freqs > 0;
    freqs_pos = freqs(positive_mask);
    input_fft_pos = input_fft(positive_mask);
    
    % Input signal analysis plot
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 1, 1);
    time_axis = (0:length(input_signal)-1) / 100000;
    plot(time_axis, input_signal);
    xlabel('Time (s)');
    ylabel('Amplitude (V)');
    title(sprintf('Time Domain - DA_%d', excited_ch-1));
    grid on;
    
    subplot(2, 1, 2);
    input_magnitude = abs(input_fft_pos);
    [max_input_val, max_input_idx] = max(input_magnitude);
    max_input_freq = freqs_pos(max_input_idx);
    
    loglog(freqs_pos, input_magnitude);
    hold on;
    scatter(max_input_freq, max_input_val, 100, 'red', 'filled');
    xline(target_freq, '--', 'Color', [1 0.5 0], 'Alpha', 0.7);
    xlabel('Frequency (Hz)');
    ylabel('|Input(f)| Magnitude');
    title(sprintf('Frequency Domain - DA_%d', excited_ch-1));
    legend(sprintf('Max: %.1f Hz', max_input_freq), sprintf('Target: %.0f Hz', target_freq));
    grid on;
    hold off;
    
    sgtitle(sprintf('Input Signal Analysis - DA_%d (Target: %.0f Hz)', ...
                   excited_ch-1, target_freq));
    
    % VM channel analysis plot
    figure('Position', [150, 50, 1600, 2000]);
    
    for vm_ch = 1:6
        output_signal = data.vm(vm_ch, start_idx:end_idx);
        
        % Preprocessing and FFT
        output_clean = output_signal; % - mean(output_signal);
        output_fft = fft(output_clean);
        output_fft_pos = output_fft(positive_mask);
        
        % Calculate transfer function
        valid_mask = abs(input_fft_pos) > 1e-10;
        H_complex = zeros(size(input_fft_pos));
        H_complex(valid_mask) = output_fft_pos(valid_mask) ./ input_fft_pos(valid_mask);
        
        % VM response spectrum
        vm_magnitude = abs(output_fft_pos);
        [max_vm_val, max_vm_idx] = max(vm_magnitude);
        max_vm_freq = freqs_pos(max_vm_idx);
        
        subplot(6, 2, (vm_ch-1)*2 + 1);
        loglog(freqs_pos, vm_magnitude);
        hold on;
        scatter(max_vm_freq, max_vm_val, 50, 'red', 'filled');
        xline(target_freq, '--', 'Color', [1 0.5 0], 'Alpha', 0.7);
        title(sprintf('VM_%d Response', vm_ch-1));
        ylabel('|VM(f)|');
        grid on;
        hold off;
        
        % Transfer function spectrum
        H_magnitude = abs(H_complex);
        freqs_valid = freqs_pos(valid_mask);
        H_valid = H_magnitude(valid_mask);
        [max_h_val, max_h_idx] = max(H_valid);
        max_h_freq = freqs_valid(max_h_idx);
        
        subplot(6, 2, (vm_ch-1)*2 + 2);
        loglog(freqs_valid, H_valid);
        hold on;
        scatter(max_h_freq, max_h_val, 50, 'red', 'filled');
        xline(target_freq, '--', 'Color', [1 0.5 0], 'Alpha', 0.7);
        title(sprintf('H_%d Transfer Function', vm_ch-1));
        ylabel('|H(f)|');
        grid on;
        hold off;
        
        if vm_ch == 6
            subplot(6, 2, 11);
            xlabel('Frequency (Hz)');
            subplot(6, 2, 12);
            xlabel('Frequency (Hz)');
        end
    end
    
    sgtitle(sprintf('VM Analysis - DA_%d Excitation (Target: %.0f Hz)', ...
                   excited_ch-1, target_freq));
end

function result = calculate_b_matrix_integrated(csv_files, target_freq, tolerance_percent, ...
                                              steady_params, plot_file)
    % Integrated B-matrix calculation
    
    if nargin < 4
        steady_params = struct('consecutive_periods', 3, ...
                              'check_points', 10, ...
                              'threshold', 1e-3);
    end
    if nargin < 5
        plot_file = [];
    end
    
    fprintf('B-matrix calculation: %.0fHz, VM validation tolerance ±%d%%\n', ...
            target_freq, tolerance_percent);
    
    % Unified steady state detection
    all_steady_info = cell(length(csv_files), 1);
    
    for i = 1:length(csv_files)
        steady_result = detect_steady_state_by_periods(csv_files{i}, target_freq, ...
                                                      'consecutive_periods', steady_params.consecutive_periods, ...
                                                      'check_points', steady_params.check_points, ...
                                                      'threshold', steady_params.threshold);
        if ~isempty(steady_result)
            all_steady_info{i} = steady_result;
        else
            fprintf('Error: %s no steady state detected\n', csv_files{i});
            result = [];
            return;
        end
    end
    
    % Use most conservative steady state point
    periods = cellfun(@(x) x.period, all_steady_info);
    unified_steady_period = max(periods);
    fprintf('Unified steady state point: Period %d\n', unified_steady_period);
    
    % B-matrix calculation
    period_samples = round(100000 / target_freq);
    B_signed = zeros(6, 6);
    B_magnitude = zeros(6, 6);
    B_phase = zeros(6, 6);
    validation_results = false(6, 6);
    
    for i = 1:length(csv_files)
        data = load_single_test(csv_files{i});
        da_voltage = dac_to_voltage(data.da);
        excited_ch = find_excitation_channel(da_voltage);
        
        % Calculate available complete periods
        steady_info = all_steady_info{i};
        available_periods = steady_info.max_periods - unified_steady_period;
        
        start_idx = steady_info.index + (unified_steady_period - steady_info.period) * period_samples;
        end_idx = start_idx + available_periods * period_samples - 1;
        
        input_signal = da_voltage(excited_ch, start_idx:end_idx);
        
        fprintf('%s: excited_ch%d, using %d complete periods\n', ...
                csv_files{i}, excited_ch-1, available_periods);
        
        % Plotting
        if ~isempty(plot_file) && strcmp(plot_file, csv_files{i})
            plot_steady_info = struct('period', unified_steady_period, ...
                                    'index', start_idx, ...
                                    'max_periods', steady_info.max_periods);
            plot_frequency_analysis(csv_files{i}, target_freq, plot_steady_info, tolerance_percent);
        end
        
        for vm_ch = 1:6
            output_signal = data.vm(vm_ch, start_idx:end_idx);
            
            [signed_gain, magnitude, phase_deg, is_valid] = ...
                calculate_transfer_function_with_validation(input_signal, output_signal, ...
                                                          100000, target_freq, tolerance_percent);
            
            B_signed(vm_ch, excited_ch) = signed_gain;
            B_magnitude(vm_ch, excited_ch) = magnitude;
            B_phase(vm_ch, excited_ch) = phase_deg;
            validation_results(vm_ch, excited_ch) = is_valid;
        end
    end
    
    % Display results
    fprintf('\nSigned B-matrix (VM rows × DA columns):\n');
    fprintf('%s\n', repmat('=', 1, 50));
    for i = 1:6
        fprintf('VM_%d  ', i-1);
        for j = 1:6
            fprintf('%+7.4f   ', B_signed(i, j));
        end
        fprintf('\n');
    end
    
    fprintf('\nMagnitude matrix:\n');
    disp(B_magnitude);
    
    fprintf('\nPhase matrix (degrees):\n');
    disp(B_phase);
    
    total_valid = sum(validation_results(:));
    fprintf('\nValidation statistics: %d/36 passed (%.1f%%)\n', ...
            total_valid, total_valid/36*100);
    
    result = struct('B_signed', B_signed, ...
                   'B_magnitude', B_magnitude, ...
                   'B_phase', B_phase, ...
                   'validation_results', validation_results);
end