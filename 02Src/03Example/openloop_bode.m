%==============================================================================
% OPENLOOP BODE ANALYSIS - Main Script
%==============================================================================
% Description:
%   Process CSV frequency response data and generate Bode plot analysis
%
% Workflow:
%   Input:  processed_csv/P1/*.csv (from hsdata_reader.py)
%   Process: FFT analysis, steady-state detection, transfer function calculation
%   Output: P1.m (frequencies, magnitudes_linear, phases, phases_processed)
%
% Usage:
%   openloop_bode_main()           - Use default folder (processed_csv/P1)
%   openloop_bode_main('path')     - Specify custom CSV folder path
%
% Author: Automated Control Lab
% Last Modified: 2025-10-05
%==============================================================================

function openloop_bode_main(varargin)

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  All adjustable parameters are centralized here
%  ========================================================================

% --- File I/O Configuration ---
SCRIPT_DIR = fileparts(mfilename('fullpath'));      % Auto-detect script location
DEFAULT_DATA_FOLDER = fullfile(SCRIPT_DIR, 'processed_csv', 'P1');
OUTPUT_BASE_FOLDER = SCRIPT_DIR;

% --- Signal Processing Parameters ---
SAMPLING_RATE = 100000;              % Sampling rate (Hz)
MIN_DA_THRESHOLD = 1e-10;           % Minimum DA signal threshold for valid FFT
INTERPOLATION_METHOD = 'spline';    % Interpolation method: 'linear'|'spline'|'pchip'|'makima'

% --- Steady-State Detection Parameters ---
STABILITY_THRESHOLD = 2e-3;          % Stability threshold (V) - max allowed difference between periods
CONSECUTIVE_PERIODS = 3;             % Number of consecutive stable periods required
CHECK_POINTS = 25;                   % Number of check points per period for comparison
START_PERIOD = 1;                    % Starting period index for steady-state detection

% --- FFT Analysis Configuration ---
FFT_MODE = 'full';                   % FFT mode: 'full' (multi-period) | 'averaged' (period-averaged)
COMPARE_FFT_METHODS = false;         % Enable comparison between FFT methods (debug mode)

% --- Visualization Settings ---
CHANNEL_COLORS = ['k','b','g','r','m','c'];  % Colors for 6 channels
DISPLAY_CHANNELS = [1,2,3,4,5,6];            % Channels to display in Bode plot
PLOT_STEADY_STATE = true;                    % Enable steady-state waveform overlay plot
PLOT_CHANNEL = [1];                          % Channels for overlay: 0=all, 1-6=specific, [3,5]=multiple
PLOT_FREQUENCY_LIST = [0.1];                 % Specific frequencies to plot (empty=all frequencies)

% --- Model Parameters (for theoretical Bode curve) ---
MODEL_WN_SQUARED = 1.4848e7;         % Natural frequency squared (rad^2/s^2)
MODEL_TWO_ZETA_WN = 8.1877e3;        % Two times damping ratio times natural frequency (rad/s)


%% ========================================================================
%  SECTION 2: INITIALIZATION
%  ========================================================================

clc;  % Clear command window

% Parse input arguments
if nargin > 0
    csv_folder = varargin{1};
else
    csv_folder = DEFAULT_DATA_FOLDER;
end

fprintf('=== OPENLOOP BODE ANALYSIS ===\n');
fprintf('Data folder: %s\n', csv_folder);

% Validate data folder existence
if ~exist(csv_folder, 'dir')
    error('Data folder does not exist: %s', csv_folder);
end

% Get list of CSV files
csv_files = dir(fullfile(csv_folder, '*.csv'));

if isempty(csv_files)
    error('No CSV files found in folder: %s', csv_folder);
end

fprintf('Found %d CSV files\n\n', length(csv_files));

% Initialize result arrays
frequencies = [];
magnitudes_db = zeros(6, 0);
phases = zeros(6, 0);
excitation_channels = [];  % Record excitation channel for each frequency point


%% ========================================================================
%  SECTION 3: MAIN PROCESSING LOOP
%  Process each CSV file (one frequency point per file)
%  ========================================================================

for i = 1:length(csv_files)
    csv_file = csv_files(i);
    file_path = fullfile(csv_folder, csv_file.name);

    fprintf('[%d/%d] Processing: %s (%.1f MB)\n', ...
        i, length(csv_files), csv_file.name, csv_file.bytes/1024/1024);

    try
        % --- Step 3.1: Load CSV data ---
        fprintf('  Step 1: Loading CSV data...\n');
        [vm_data, da_data] = load_csv_data(file_path);
        data_length = size(vm_data, 2);

        % --- Step 3.2: Repair bad data points ---
        fprintf('  Step 2: Repairing bad data points...\n');
        [vm_clean, da_clean] = repair_data_points(vm_data, da_data, data_length, INTERPOLATION_METHOD);

        % Convert DA to voltage
        da_volt = (da_clean - 32768) * (20.0 / 65536);

        % --- Step 3.3: Detect excitation channel and frequency ---
        fprintf('  Step 3: Detecting excitation channel and frequency...\n');
        [excite_ch, excite_freq] = detect_excitation(da_volt, SAMPLING_RATE);
        fprintf('    Excitation: DA%d, Frequency: %.1f Hz\n', excite_ch, excite_freq);

        % --- Step 3.4: Steady-state detection ---
        fprintf('  Step 4: Steady-state detection...\n');
        steady_info = detect_steady_state(vm_clean, excite_freq, SAMPLING_RATE, ...
            STABILITY_THRESHOLD, CONSECUTIVE_PERIODS, CHECK_POINTS, START_PERIOD);

        if isempty(steady_info)
            fprintf('  ✗ Steady-state detection failed, skipping file\n');
            continue;
        end

        fprintf('    Steady-state starts at period: %d\n', steady_info.period);

        % Plot steady-state waveform overlay (if enabled)
        if PLOT_STEADY_STATE && should_plot_frequency(excite_freq, PLOT_FREQUENCY_LIST)
            plot_steady_state_overlay(vm_clean, da_volt, steady_info, excite_ch, ...
                excite_freq, CONSECUTIVE_PERIODS, PLOT_CHANNEL, SAMPLING_RATE);
        end

        % --- Step 3.5: FFT analysis ---
        fprintf('  Step 5: FFT analysis (mode: %s)...\n', FFT_MODE);
        [current_magnitudes_db, current_phases] = perform_fft_analysis(...
            vm_clean, da_volt, steady_info, excite_ch, excite_freq, ...
            SAMPLING_RATE, FFT_MODE, MIN_DA_THRESHOLD, COMPARE_FFT_METHODS);

        % --- Step 3.6: Store results ---
        frequencies = [frequencies, excite_freq];
        magnitudes_db(:, end+1) = current_magnitudes_db;
        phases(:, end+1) = current_phases;
        excitation_channels = [excitation_channels, excite_ch];

        fprintf('  ✓ Successfully processed frequency %.1f Hz\n\n', excite_freq);

    catch err
        fprintf('  ✗ Processing failed: %s\n\n', err.message);
        continue;
    end
end


%% ========================================================================
%  SECTION 4: POST-PROCESSING & OUTPUT
%  ========================================================================

if isempty(frequencies)
    fprintf('No files were successfully processed\n');
    return;
end

% --- Sort results by frequency ---
[frequencies, sort_idx] = sort(frequencies);
magnitudes_db = magnitudes_db(:, sort_idx);
phases = phases(:, sort_idx);
excitation_channels = excitation_channels(sort_idx);

% --- Normalize magnitude data ---
fprintf('Normalizing magnitude data...\n');
magnitudes_db_normalized = normalize_magnitudes(magnitudes_db, frequencies);

% --- Analysis summary ---
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Successfully processed: %d frequency points\n', length(frequencies));
fprintf('Frequency range: %.1f - %.1f Hz\n', min(frequencies), max(frequencies));

% --- Generate Bode plots ---
fprintf('\nGenerating Bode plots...\n');
phases_processed = plot_bode_results(frequencies, magnitudes_db_normalized, phases, ...
    CHANNEL_COLORS, magnitudes_db, excitation_channels, DISPLAY_CHANNELS, ...
    MODEL_WN_SQUARED, MODEL_TWO_ZETA_WN);

% --- Save to workspace ---
assignin('base', 'openloop_frequencies', frequencies);
assignin('base', 'openloop_magnitudes_db_original', magnitudes_db);
assignin('base', 'openloop_magnitudes_db_normalized', magnitudes_db_normalized);
assignin('base', 'openloop_phases', phases);
fprintf('Results saved to workspace variables\n');

% --- Calculate linear magnitude ---
fprintf('\nCalculating linear magnitude data...\n');
magnitudes_linear = 10.^(magnitudes_db / 20);

% --- Generate output filename from folder name ---
[~, folder_name, ~] = fileparts(csv_folder);
output_filename = fullfile(OUTPUT_BASE_FOLDER, [folder_name '.m']);

% --- Save to .m file (P1.m format) ---
fprintf('Saving Bode data to: %s...\n', output_filename);
save_bode_data_to_file(output_filename, frequencies, magnitudes_linear, phases, phases_processed);
fprintf('✓ Bode data saved successfully\n');

fprintf('\n=== ALL OPERATIONS COMPLETE ===\n');

end


%% ========================================================================
%  SECTION 5: HELPER FUNCTIONS
%  ========================================================================

%% ------------------------------------------------------------------------
%  5.1 DATA LOADING & PREPROCESSING
%  ------------------------------------------------------------------------

function [vm_data, da_data] = load_csv_data(file_path)
% LOAD_CSV_DATA Load VM and DA data from CSV file
%
% Input:
%   file_path - Path to CSV file
%
% Output:
%   vm_data - VM voltage data (6 x N)
%   da_data - DA digital data (6 x N)

    raw_data = readtable(file_path);
    data_length = height(raw_data);

    % Initialize data matrices
    vm_data = zeros(6, data_length);
    da_data = zeros(6, data_length);

    % Extract data for each channel
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
end


function [vm_clean, da_clean] = repair_data_points(vm_data, da_data, data_length, interpolation_method)
% REPAIR_DATA_POINTS Repair bad data points using interpolation
%
% Input:
%   vm_data - Raw VM data (6 x N)
%   da_data - Raw DA data (6 x N)
%   data_length - Total number of data points
%   interpolation_method - Interpolation method ('linear'|'spline'|'pchip'|'makima')
%
% Output:
%   vm_clean - Repaired VM data (6 x N)
%   da_clean - Repaired DA data (6 x N)

    % Identify bad data indices (every 10000th point)
    bad_indices = 1:10000:data_length;
    num_bad_points = length(bad_indices);

    fprintf('    Using interpolation method: %s\n', interpolation_method);

    % Copy original data
    vm_clean = vm_data;
    da_clean = da_data;

    % Save original bad point values for comparison
    vm_bad_original = vm_data(:, bad_indices);
    da_bad_original = da_data(:, bad_indices);

    if num_bad_points > 0
        % Find good data indices
        good_indices = setdiff(1:data_length, bad_indices);

        % Interpolate for each channel
        for ch = 1:6
            if length(good_indices) >= 4
                try
                    % Use selected interpolation method
                    vm_clean(ch, bad_indices) = interp1(good_indices, ...
                        vm_data(ch, good_indices), bad_indices, interpolation_method, 'extrap');
                    da_clean(ch, bad_indices) = interp1(good_indices, ...
                        da_data(ch, good_indices), bad_indices, interpolation_method, 'extrap');
                catch
                    % Fallback to linear interpolation if advanced method fails
                    fprintf('    Warning: Ch%d interpolation failed, using linear fallback\n', ch);
                    vm_clean(ch, bad_indices) = interp1(good_indices, ...
                        vm_data(ch, good_indices), bad_indices, 'linear', 'extrap');
                    da_clean(ch, bad_indices) = interp1(good_indices, ...
                        da_data(ch, good_indices), bad_indices, 'linear', 'extrap');
                end
            else
                % Insufficient data points, use simple linear interpolation
                for idx = bad_indices
                    if idx > 1 && idx < data_length
                        vm_clean(ch, idx) = (vm_data(ch, idx-1) + vm_data(ch, idx+1)) / 2;
                        da_clean(ch, idx) = (da_data(ch, idx-1) + da_data(ch, idx+1)) / 2;
                    end
                end
            end
        end

        % Calculate repair statistics
        vm_error_rms = sqrt(mean((vm_clean(:, bad_indices) - vm_bad_original).^2, 2));
        da_error_rms = sqrt(mean((da_clean(:, bad_indices) - da_bad_original).^2, 2));

        fprintf('    VM repair RMS error: %.6f V (average)\n', mean(vm_error_rms));
        fprintf('    DA repair RMS error: %.6f V (average)\n', mean(da_error_rms));
    end

    fprintf('    Total data points: %d, Repaired points: %d\n', data_length, num_bad_points);
end


%% ------------------------------------------------------------------------
%  5.2 SIGNAL ANALYSIS
%  ------------------------------------------------------------------------

function [excite_ch, excite_freq] = detect_excitation(da_voltage, sampling_rate)
% DETECT_EXCITATION Detect excitation channel and frequency
%
% Input:
%   da_voltage - DA voltage data (6 x N)
%   sampling_rate - Sampling rate (Hz)
%
% Output:
%   excite_ch - Excitation channel index (1-6)
%   excite_freq - Dominant frequency (Hz)

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
        error('No valid excitation channel detected');
    end

    excite_ch = best_channel;
    excite_freq = best_freq;
end


function steady_info = detect_steady_state(vm_signal, target_freq, sampling_rate, ...
    stability_threshold, consecutive_periods, check_points, start_period)
% DETECT_STEADY_STATE Detect steady-state region in VM signal
%
% Input:
%   vm_signal - VM voltage data (6 x N or 1 x N)
%   target_freq - Signal frequency (Hz)
%   sampling_rate - Sampling rate (Hz)
%   stability_threshold - Maximum allowed difference between periods (V)
%   consecutive_periods - Number of consecutive stable periods required
%   check_points - Number of check points per period
%   start_period - Starting period index for detection
%
% Output:
%   steady_info - Structure containing:
%       .period - Steady-state starting period index
%       .index - Steady-state starting sample index
%       .max_periods - Maximum available periods
%       .period_samples - Samples per period

    % Convert to 6-channel matrix if single channel input
    if isvector(vm_signal)
        vm_clean = vm_signal(:)';
        vm_clean = repmat(vm_clean, 6, 1);
    else
        vm_clean = vm_signal;
    end

    clean_length = size(vm_clean, 2);

    % Calculate period parameters
    period_samples = round(sampling_rate / target_freq);
    max_periods = floor(clean_length / period_samples);
    check_positions = round(linspace(1, period_samples, check_points));

    fprintf('    Period samples: %d, Max periods: %d\n', period_samples, max_periods);

    % Handle insufficient data case
    if max_periods < 5
        fprintf('    ⚠ Warning: Insufficient periods (only %d), using last period as steady-state\n', max_periods);
        fprintf('    ⚠ Data may contain transient response, reliability is low!\n');

        steady_period = max(1, max_periods - 1);
        steady_info = struct('period', steady_period, ...
                            'index', steady_period * period_samples + 1, ...
                            'max_periods', max_periods, ...
                            'period_samples', period_samples);
        return;
    end

    steady_periods = [];

    % Detect steady-state for each VM channel
    for vm_ch = 1:6
        signal = vm_clean(vm_ch, :);

        % Test each period starting from start_period
        for test_period = start_period:(max_periods - consecutive_periods)
            all_stable = true;

            % Check stability across consecutive periods
            for i = 1:consecutive_periods
                current_period = test_period + i - 1;
                next_period = current_period + 1;

                current_start = current_period * period_samples + 1;
                next_start = next_period * period_samples + 1;

                max_diff = 0;

                % Compare adjacent periods at check points
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

                % Mark as unstable if difference exceeds threshold
                if max_diff >= stability_threshold
                    all_stable = false;
                    break;
                end
            end

            % Record and stop if stable period found
            if all_stable
                steady_periods(end+1) = test_period;
                break;
            end
        end
    end

    % Select most conservative steady-state point (maximum)
    if ~isempty(steady_periods)
        recommended_period = max(steady_periods);
        clean_index = recommended_period * period_samples + 1;

        steady_info = struct(...
            'period', recommended_period, ...
            'index', clean_index, ...
            'max_periods', max_periods, ...
            'period_samples', period_samples);

        fprintf('    Steady-state detected: Period %d, Index %d\n', recommended_period, clean_index);
    else
        % Use last periods if no stable period found
        fprintf('    Warning: No stable period found (threshold %.4fV)\n', stability_threshold);
        fprintf('    Using last %d periods as steady-state (may be unstable)\n', consecutive_periods);

        steady_period = max(1, max_periods - consecutive_periods);
        steady_info = struct('period', steady_period, ...
                            'index', steady_period * period_samples + 1, ...
                            'max_periods', max_periods, ...
                            'period_samples', period_samples);
    end
end


%% ------------------------------------------------------------------------
%  5.3 FFT ANALYSIS
%  ------------------------------------------------------------------------

function [magnitudes_db, phases] = perform_fft_analysis(...
    vm_clean, da_volt, steady_info, excite_ch, excite_freq, ...
    sampling_rate, fft_mode, min_da_threshold, compare_fft_methods)
% PERFORM_FFT_ANALYSIS Calculate transfer function using FFT
%
% Transfer Function Calculation:
%   H(jω) = VM(jω) / DA(jω)
%   - DA: Excitation signal (input)
%   - VM: System response (output)
%   - Goal: Calculate magnitude |H| and phase ∠H at excitation frequency
%
% Two FFT Modes:
%   1. 'full' (default): FFT on multiple complete periods
%      - Higher frequency resolution
%      - Better noise suppression with more data points
%   2. 'averaged': Average periods first, then FFT on single period
%      - Reduces inter-period variation
%      - Lower frequency resolution
%
% Input:
%   vm_clean - Cleaned VM data (6 x N)
%   da_volt - DA voltage data (6 x N)
%   steady_info - Steady-state information structure
%   excite_ch - Excitation channel index (1-6)
%   excite_freq - Excitation frequency (Hz)
%   sampling_rate - Sampling rate (Hz)
%   fft_mode - 'full' or 'averaged'
%   min_da_threshold - Minimum DA threshold for valid FFT
%   compare_fft_methods - Enable method comparison
%
% Output:
%   magnitudes_db - Magnitude in dB (6 x 1)
%   phases - Phase in degrees (6 x 1)

    magnitudes_db = zeros(6, 1);
    phases = zeros(6, 1);

    % Extract steady-state parameters
    period_samples = steady_info.period_samples;
    steady_start = steady_info.index;
    available_length = size(vm_clean, 2) - steady_start + 1;
    available_periods = floor(available_length / period_samples);

    if available_periods < 1
        fprintf('    Warning: Insufficient data for one complete period\n');
        return;
    end

    % --- Averaged FFT Mode ---
    if strcmp(fft_mode, 'averaged')
        fprintf('    Using period-averaged FFT: %d periods\n', available_periods);

        da_signal = da_volt(excite_ch, :);

        % Extract and average all periods
        vm_periods = zeros(6, available_periods, period_samples);
        da_periods = zeros(available_periods, period_samples);

        for p = 1:available_periods
            start_idx = steady_start + (p-1) * period_samples;
            end_idx = start_idx + period_samples - 1;

            for ch = 1:6
                vm_periods(ch, p, :) = vm_clean(ch, start_idx:end_idx);
            end
            da_periods(p, :) = da_signal(start_idx:end_idx);
        end

        % Calculate averaged periods
        vm_avg_periods = squeeze(mean(vm_periods, 2));  % 6 x period_samples
        da_avg_period = mean(da_periods, 1);            % 1 x period_samples

        % Calculate period-to-period standard deviation
        vm_std = squeeze(std(vm_periods, 0, 2));
        fprintf('    VM inter-period std: %.6f (average)\n', mean(vm_std(:)));

        % FFT for each channel
        for ch = 1:6
            vm_fft = fft(vm_avg_periods(ch, :));
            da_fft = fft(da_avg_period);

            % Calculate target bin dynamically (not always bin 2!)
            N_avg = period_samples;
            freq_resolution_avg = sampling_rate / N_avg;
            target_bin = round(excite_freq / freq_resolution_avg) + 1;

            % Validate frequency bin
            actual_freq = (target_bin - 1) * freq_resolution_avg;
            if abs(actual_freq - excite_freq) > 0.01
                fprintf('    Warning CH%d: Frequency mismatch - target %.2f Hz, actual %.2f Hz\n', ...
                        ch, excite_freq, actual_freq);
            end

            % Calculate transfer function
            vm_complex = vm_fft(target_bin);
            da_complex = da_fft(target_bin);

            if abs(da_complex) > min_da_threshold
                transfer_function = vm_complex / da_complex;
                magnitude_linear = abs(transfer_function);
                magnitudes_db(ch) = 20 * log10(magnitude_linear);
                phases(ch) = angle(transfer_function) * 180 / pi;
            else
                magnitudes_db(ch) = -Inf;
                phases(ch) = 0;
            end
        end

        % Compare with full FFT if requested
        if compare_fft_methods
            fprintf('    Calculating full FFT for comparison...\n');
            [magnitudes_full, phases_full] = calculate_full_fft(...
                vm_clean, da_volt, excite_ch, excite_freq, ...
                steady_start, available_periods, period_samples, ...
                sampling_rate, min_da_threshold);

            fprintf('    === FFT Method Comparison ===\n');
            for ch = 1:6
                mag_diff = magnitudes_db(ch) - magnitudes_full(ch);
                phase_diff = phases(ch) - phases_full(ch);
                fprintf('    CH%d: ΔMag=%.3f dB, ΔPhase=%.2f°\n', ch, mag_diff, phase_diff);
            end
        end

    % --- Full FFT Mode ---
    else
        fprintf('    Using full FFT: %d periods\n', available_periods);

        for ch = 1:6
            vm_signal = vm_clean(ch, :);
            da_signal = da_volt(excite_ch, :);

            % Extract complete periods with boundary check
            max_index = size(vm_clean, 2);
            end_index = steady_start + available_periods * period_samples - 1;

            % Boundary validation
            if end_index > max_index
                fprintf('    Warning CH%d: Requested end_index (%d) exceeds data length (%d)\n', ...
                        ch, end_index, max_index);
                end_index = max_index;
            end

            % Ensure we have at least one complete period
            actual_length = end_index - steady_start + 1;
            if actual_length < period_samples
                fprintf('    Warning CH%d: Insufficient data (%d samples) for one period (%d samples)\n', ...
                        ch, actual_length, period_samples);
                magnitudes_db(ch) = -Inf;
                phases(ch) = 0;
                continue;
            end

            vm_period_data = vm_signal(steady_start:end_index);
            da_period_data = da_signal(steady_start:end_index);

            % Perform FFT
            vm_fft = fft(vm_period_data);
            da_fft = fft(da_period_data);

            % Find target frequency bin
            N = length(vm_period_data);
            freq_axis = (0:N-1) * sampling_rate / N;
            freq_resolution = freq_axis(2) - freq_axis(1);
            target_bin = round(excite_freq / freq_resolution) + 1;

            % Validate frequency bin
            actual_freq = (target_bin - 1) * freq_resolution;
            if abs(actual_freq - excite_freq) > 0.01
                fprintf('    Warning CH%d: Frequency mismatch - target %.2f Hz, actual %.2f Hz\n', ...
                        ch, excite_freq, actual_freq);
            end

            % Calculate transfer function
            vm_complex = vm_fft(target_bin);
            da_complex = da_fft(target_bin);

            if abs(da_complex) > min_da_threshold
                transfer_function = vm_complex / da_complex;
                magnitude_linear = abs(transfer_function);
                magnitudes_db(ch) = 20 * log10(magnitude_linear);
                phases(ch) = angle(transfer_function) * 180 / pi;
            else
                magnitudes_db(ch) = -Inf;
                phases(ch) = 0;
            end
        end
    end
end


function [magnitudes_db, phases] = calculate_full_fft(...
    vm_clean, da_volt, excite_ch, excite_freq, ...
    steady_start, available_periods, period_samples, ...
    sampling_rate, min_da_threshold)
% CALCULATE_FULL_FFT Helper function for full FFT calculation
% (Used for comparison in averaged mode)

    magnitudes_db = zeros(6, 1);
    phases = zeros(6, 1);

    for ch = 1:6
        % Boundary check
        max_index = size(vm_clean, 2);
        end_index = steady_start + available_periods * period_samples - 1;

        if end_index > max_index
            fprintf('    Warning (compare) CH%d: end_index (%d) > data length (%d)\n', ...
                    ch, end_index, max_index);
            end_index = max_index;
        end

        % Ensure sufficient data
        actual_length = end_index - steady_start + 1;
        if actual_length < period_samples
            magnitudes_db(ch) = -Inf;
            phases(ch) = 0;
            continue;
        end

        vm_full = vm_clean(ch, steady_start:end_index);
        da_full = da_volt(excite_ch, steady_start:end_index);

        vm_fft_full = fft(vm_full);
        da_fft_full = fft(da_full);

        N_full = length(vm_full);
        freq_res = sampling_rate / N_full;
        target_bin_full = round(excite_freq / freq_res) + 1;

        % Validate frequency bin
        actual_freq = (target_bin_full - 1) * freq_res;
        if abs(actual_freq - excite_freq) > 0.01
            fprintf('    Warning (compare) CH%d: Freq mismatch - target %.2f Hz, actual %.2f Hz\n', ...
                    ch, excite_freq, actual_freq);
        end

        if abs(da_fft_full(target_bin_full)) > min_da_threshold
            H_full = vm_fft_full(target_bin_full) / da_fft_full(target_bin_full);
            magnitudes_db(ch) = 20 * log10(abs(H_full));
            phases(ch) = angle(H_full) * 180 / pi;
        else
            magnitudes_db(ch) = -Inf;
            phases(ch) = 0;
        end
    end
end


%% ------------------------------------------------------------------------
%  5.4 DATA NORMALIZATION & PROCESSING
%  ------------------------------------------------------------------------

function magnitudes_normalized = normalize_magnitudes(magnitudes_db, frequencies)
% NORMALIZE_MAGNITUDES Normalize magnitude data to lowest frequency
%
% Input:
%   magnitudes_db - Magnitude in dB (6 x N)
%   frequencies - Frequency vector (1 x N)
%
% Output:
%   magnitudes_normalized - Normalized magnitude in dB (6 x N)

    if isempty(magnitudes_db) || isempty(frequencies)
        magnitudes_normalized = magnitudes_db;
        return;
    end

    % Find lowest frequency index
    [~, min_freq_idx] = min(frequencies);

    magnitudes_normalized = zeros(size(magnitudes_db));

    fprintf('Normalization reference: %.2f Hz\n', frequencies(min_freq_idx));

    % Normalize each channel
    for ch = 1:6
        reference_value = magnitudes_db(ch, min_freq_idx);

        if isfinite(reference_value)
            % Normalize: subtract lowest frequency value
            magnitudes_normalized(ch, :) = magnitudes_db(ch, :) - reference_value;
            fprintf('  CH%d: Reference = %.2f dB\n', ch, reference_value);
        else
            % Keep original if reference is invalid
            magnitudes_normalized(ch, :) = magnitudes_db(ch, :);
            fprintf('  CH%d: Invalid reference, keeping original data\n', ch);
        end
    end

    fprintf('Normalization complete! All channels referenced to 0 dB at lowest frequency\n');
end


%% ------------------------------------------------------------------------
%  5.5 VISUALIZATION
%  ------------------------------------------------------------------------

function phases_processed = plot_bode_results(frequencies, magnitudes_db, phases, ...
    colors, original_magnitudes_db, excitation_channels, display_channels, ...
    model_wn_squared, model_two_zeta_wn)
% PLOT_BODE_RESULTS Generate Bode magnitude and phase plots
%
% Input:
%   frequencies - Frequency vector (1 x N)
%   magnitudes_db - Normalized magnitude in dB (6 x N)
%   phases - Phase in degrees (6 x N)
%   colors - Color array for channels
%   original_magnitudes_db - Original magnitude (for normalization values)
%   excitation_channels - Excitation channel for each frequency
%   display_channels - Channels to display
%   model_wn_squared - Model natural frequency squared
%   model_two_zeta_wn - Model damping parameter
%
% Output:
%   phases_processed - Phase with 180° correction applied (6 x N)

    % Process phase data
    phases_processed = phases;

    % Apply 180° phase correction based on excitation channel
    for freq_idx = 1:length(frequencies)
        excite_ch = excitation_channels(freq_idx);

        if ismember(excite_ch, [1, 3, 6])
            % For excitation channels 1, 3, 6: subtract 180° from excitation channel only
            phases_processed(excite_ch, freq_idx) = phases_processed(excite_ch, freq_idx) - 180;
        else
            % For excitation channels 2, 4, 5: subtract 180° from non-excitation channels
            for ch = 1:6
                if ch ~= excite_ch
                    phases_processed(ch, freq_idx) = phases_processed(ch, freq_idx) - 180;
                end
            end
        end
    end

    % Calculate normalization values (linear magnitude at lowest frequency)
    [~, min_freq_idx] = min(frequencies);
    normalization_values_db = original_magnitudes_db(:, min_freq_idx);
    normalization_values_linear = 10.^(normalization_values_db/20);

    % Create figure with two subplots
    figure('Name', 'Open-loop Bode Plot', 'Position', [100, 100, 900, 720]);

    %% --- Magnitude Plot ---
    subplot(2,1,1);
    hold on;

    % Plot each channel
    for ch = display_channels
        norm_val_linear = normalization_values_linear(ch);
        if isfinite(norm_val_linear)
            legend_text = sprintf('P%d (%.3f)', ch, norm_val_linear);
        else
            legend_text = sprintf('P%d (N/A)', ch);
        end

        semilogx(frequencies, magnitudes_db(ch, :), ...
                'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
                'MarkerSize', 10, 'DisplayName', legend_text);
    end

    % Add theoretical model response
    if ~isempty(frequencies)
        omega = 2 * pi * frequencies;
        s = 1j * omega;

        H_s = model_wn_squared ./ (s.^2 + model_two_zeta_wn * s + model_wn_squared);
        H_magnitude_db = 20 * log10(abs(H_s));

        semilogx(frequencies, H_magnitude_db, '-', ...
                'Color', 'k', 'LineWidth', 3, 'DisplayName', 'Model');
    end

    xlabel('Frequency (Hz)', 'FontWeight', 'bold', 'FontSize', 40);
    ylabel('Magnitude (dB)', 'FontWeight', 'bold', 'FontSize', 40);
    legend('Location', 'southwest', 'FontWeight', 'bold','FontSize', 24);

    % Set logarithmic X-axis
    set(gca, 'XScale', 'log');
    if ~isempty(frequencies)
        freq_max = max(frequencies);
        xlim([0.1, freq_max]);

        log_min = -1;
        log_max = ceil(log10(freq_max));
        log_ticks = 10.^(log_min+1:log_max);
        set(gca, 'XTick', log_ticks);
    end

    % Set Y-axis limits
    if ~isempty(magnitudes_db)
        y_min = min(magnitudes_db(:));
        if isfinite(y_min)
            ylim([y_min - 5, 2]);
        end
    end

    % Format axes
    set(gca, 'FontWeight', 'bold', 'FontSize', 24, 'LineWidth', 2);
    ax = gca;
    ax.XAxis.LineWidth = 3;
    ax.YAxis.LineWidth = 3;
    ax.XAxis.FontWeight = 'bold';
    ax.YAxis.FontWeight = 'bold';
    box on;
    ax.Box = 'on';
    ax.BoxStyle = 'full';

    %% --- Phase Plot ---
    subplot(2,1,2);
    hold on;

    % Plot each channel
    for ch = display_channels
        norm_val_linear = normalization_values_linear(ch);
        if isfinite(norm_val_linear)
            legend_text = sprintf('P%d (%.3f)', ch, norm_val_linear);
        else
            legend_text = sprintf('P%d (N/A)', ch);
        end

        semilogx(frequencies, phases_processed(ch, :), ...
                'Color', colors(ch), 'LineWidth', 2, 'Marker', 'o', ...
                'MarkerSize', 10, 'DisplayName', legend_text);
    end

    % Add theoretical model phase response
    if ~isempty(frequencies)
        omega = 2 * pi * frequencies;
        s = 1j * omega;

        H_s = model_wn_squared ./ (s.^2 + model_two_zeta_wn * s + model_wn_squared);
        H_phase_deg = angle(H_s) * 180 / pi;

        semilogx(frequencies, H_phase_deg, '-', ...
                'Color', 'k', 'LineWidth', 3, 'DisplayName', 'Model');
    end

    xlabel('Frequency (Hz)', 'FontWeight', 'bold', 'FontSize', 40);
    ylabel('Phase (deg)', 'FontWeight', 'bold', 'FontSize', 40);

    % Set logarithmic X-axis
    set(gca, 'XScale', 'log');
    if ~isempty(frequencies)
        xlim([0.1, freq_max]);
        set(gca, 'XTick', log_ticks);
    end

    % Format axes
    set(gca, 'FontWeight', 'bold', 'FontSize', 24, 'LineWidth', 2);
    ax = gca;
    ax.XAxis.LineWidth = 3;
    ax.YAxis.LineWidth = 3;
    ax.XAxis.FontWeight = 'bold';
    ax.YAxis.FontWeight = 'bold';
    box on;
    ax.Box = 'on';
    ax.BoxStyle = 'full';

    fprintf('Bode plots completed\n');
    fprintf('Frequency range: %.2f - %.2f Hz\n', min(frequencies), max(frequencies));
    fprintf('Frequency points: %d\n', length(frequencies));
end


function plot_steady_state_overlay(vm_clean, da_volt, steady_info, excite_ch, ...
    target_freq, num_periods, plot_channel, sampling_rate)
% PLOT_STEADY_STATE_OVERLAY Visualize steady-state waveform overlay
%
% Input:
%   vm_clean - Cleaned VM data (6 x N)
%   da_volt - DA voltage data (6 x N)
%   steady_info - Steady-state information structure
%   excite_ch - Excitation channel index
%   target_freq - Signal frequency (Hz)
%   num_periods - Number of periods to display
%   plot_channel - 0=all channels, 1-6=specific, [3,5]=multiple
%   sampling_rate - Sampling rate (Hz)

    % Extract steady-state parameters
    steady_start = steady_info.index;
    period_samples = steady_info.period_samples;
    max_periods = steady_info.max_periods;

    % Calculate available periods
    available_from_steady = floor((size(vm_clean, 2) - steady_start + 1) / period_samples);
    periods_to_plot = min(num_periods, available_from_steady);

    if periods_to_plot < 1
        fprintf('    Warning: Insufficient data after steady-state, cannot plot overlay\n');
        return;
    end

    % Time axis (normalized to one period)
    time_axis = (0:period_samples-1) / sampling_rate * 1000;  % Convert to ms

    % Determine channels to plot
    if isequal(plot_channel, 0)
        channels_to_plot = 1:6;
    elseif isvector(plot_channel) && length(plot_channel) > 1
        channels_to_plot = plot_channel;
    else
        channels_to_plot = plot_channel;
    end

    % Create figure
    figure('Name', sprintf('Steady-State Waveform Overlay - %.1f Hz', target_freq), ...
           'Position', [50, 50, 1000, 700]);

    % Color settings
    colors = lines(max(periods_to_plot, 3));

    num_subplots = length(channels_to_plot);
    plot_rows = ceil(sqrt(num_subplots));
    plot_cols = ceil(num_subplots / plot_rows);

    for idx = 1:num_subplots
        ch = channels_to_plot(idx);
        ax = subplot(plot_rows, plot_cols, idx);
        hold on;

        % Plot VM data for each period
        legend_entries = {};
        all_vm_data = [];

        for p = 1:periods_to_plot
            period_start = steady_start + (p-1) * period_samples;
            period_end = period_start + period_samples - 1;

            if period_end <= size(vm_clean, 2)
                vm_data = vm_clean(ch, period_start:period_end);
                all_vm_data(p, :) = vm_data;

                % Plot VM waveform with color gradient
                color_adjusted = colors(p,:) * (0.3 + 0.7 * (p/periods_to_plot));
                plot(time_axis, vm_data, '-', ...
                    'Color', color_adjusted, 'LineWidth', 2.5);
                legend_entries{end+1} = sprintf('Period %d', steady_info.period + p - 1);
            end
        end

        % Plot DA waveform (excitation channel) on right Y-axis
        da_start = steady_start;
        da_end = da_start + period_samples - 1;

        if da_end <= size(da_volt, 2)
            da_data = da_volt(excite_ch, da_start:da_end);

            yyaxis right;
            plot(time_axis, da_data, 'k-', ...
                'LineWidth', 3, 'DisplayName', sprintf('DA%d', excite_ch));
            ylabel('DA (V)', 'FontWeight', 'bold', 'FontSize', 16);
            set(gca, 'YColor', 'k', 'LineWidth', 2);

            yyaxis left;
        end

        % Labels and formatting
        xlabel('Time (ms)', 'FontWeight', 'bold', 'FontSize', 16);
        ylabel('VM', 'FontWeight', 'bold', 'FontSize', 16);

        % Title
        if ch == excite_ch
            title_str = sprintf('Ch%d (excited)', ch);
        else
            title_str = sprintf('Ch%d', ch);
        end
        title(title_str, 'FontWeight', 'bold', 'FontSize', 18);

        grid off;
        legend(legend_entries, 'Location', 'best', 'FontSize', 14, 'LineWidth', 1.5);

        % Format axes
        set(gca, 'FontWeight', 'bold', 'FontSize', 14, 'LineWidth', 2);
        ax.XAxis.LineWidth = 2.5;
        ax.YAxis(1).LineWidth = 2.5;
        if length(ax.YAxis) > 1
            ax.YAxis(2).LineWidth = 2.5;
        end
        box on;
        set(gca, 'BoxStyle', 'full');
    end

    % Overall title with status indicator
    consecutive_periods = num_periods;
    if steady_info.max_periods < 5
        status_str = ' [Insufficient Data]';
        title_color = [1, 0.5, 0];
    elseif steady_info.period >= steady_info.max_periods - consecutive_periods
        status_str = ' [Fallback]';
        title_color = 'r';
    else
        status_str = ' [Steady Detected]';
        title_color = 'k';
    end

    % Adjust subplot spacing for title
    set(gcf, 'Units', 'normalized');
    all_axes = findobj(gcf, 'Type', 'axes');
    for i = 1:length(all_axes)
        pos = get(all_axes(i), 'Position');
        set(all_axes(i), 'Position', [pos(1), pos(2)-0.02, pos(3)*1.02, pos(4)*0.92]);
    end

    % Set overall title
    sgtitle(sprintf('Steady-State Overlay @ %.1f Hz (Period %d, %d cycles)%s', ...
            target_freq, steady_info.period, periods_to_plot, status_str), ...
            'FontWeight', 'bold', 'FontSize', 18, 'Color', title_color);

    fprintf('    Waveform overlay displayed: %d periods starting from period %d\n', ...
            periods_to_plot, steady_info.period);
end


function should_plot = should_plot_frequency(freq, freq_list)
% SHOULD_PLOT_FREQUENCY Determine if frequency should be plotted
%
% Input:
%   freq - Current frequency (Hz)
%   freq_list - List of frequencies to plot (empty = plot all)
%
% Output:
%   should_plot - Boolean flag

    if isempty(freq_list)
        should_plot = true;  % Empty list = plot all frequencies
    else
        % Tolerance for floating-point comparison
        tolerance = 0.1;  % Hz
        should_plot = any(abs(freq - freq_list(:)) < tolerance);
    end
end


%% ------------------------------------------------------------------------
%  5.6 FILE I/O
%  ------------------------------------------------------------------------

function save_bode_data_to_file(filename, frequencies, magnitudes_linear, phases, phases_processed)
% SAVE_BODE_DATA_TO_FILE Save Bode data to .m file (P1.m format)
%
% Input:
%   filename - Output file path
%   frequencies - Frequency vector (1 x N)
%   magnitudes_linear - Linear magnitude (6 x N)
%   phases - Original phase in degrees (6 x N)
%   phases_processed - Processed phase in degrees (6 x N)

    % Open file for writing
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot create file: %s', filename);
    end

    try
        % Write header comments
        fprintf(fid, '%% Bode Plot Data\n');
        fprintf(fid, '%% Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, '%% Data dimension: 6 channels x %d frequency points\n', length(frequencies));
        fprintf(fid, '%% Channels: P1-P6\n\n');

        % Write frequency data
        fprintf(fid, '%% Frequency points (Hz)\n');
        fprintf(fid, 'frequencies = [');

        items_per_line = 10;
        for i = 1:length(frequencies)
            if mod(i-1, items_per_line) == 0
                if i > 1
                    fprintf(fid, ' ...\n               ');
                end
            end

            if i < length(frequencies)
                fprintf(fid, '%.6f, ', frequencies(i));
            else
                fprintf(fid, '%.6f', frequencies(i));
            end
        end
        fprintf(fid, '];\n\n');

        % Write linear magnitude data
        fprintf(fid, '%% Linear magnitude (6 channels x %d frequencies)\n', length(frequencies));
        fprintf(fid, '%% Row 1-6: P1-P6\n');
        fprintf(fid, 'magnitudes_linear = [\n');
        for ch = 1:6
            fprintf(fid, '    ');
            for i = 1:length(frequencies)
                if i < length(frequencies)
                    fprintf(fid, '%.6e, ', magnitudes_linear(ch, i));
                else
                    fprintf(fid, '%.6e', magnitudes_linear(ch, i));
                end
            end
            if ch < 6
                fprintf(fid, '; ...\n');
            else
                fprintf(fid, '\n');
            end
        end
        fprintf(fid, '];\n\n');

        % Write original phase data
        fprintf(fid, '%% Original phase (degrees)\n');
        fprintf(fid, 'phases = [\n');
        for ch = 1:6
            fprintf(fid, '    ');
            for i = 1:length(frequencies)
                if i < length(frequencies)
                    fprintf(fid, '%.4f, ', phases(ch, i));
                else
                    fprintf(fid, '%.4f', phases(ch, i));
                end
            end
            if ch < 6
                fprintf(fid, '; ...\n');
            else
                fprintf(fid, '\n');
            end
        end
        fprintf(fid, '];\n\n');

        % Write processed phase data
        fprintf(fid, '%% Processed phase (degrees, with 180deg correction)\n');
        fprintf(fid, 'phases_processed = [\n');
        for ch = 1:6
            fprintf(fid, '    ');
            for i = 1:length(frequencies)
                if i < length(frequencies)
                    fprintf(fid, '%.4f, ', phases_processed(ch, i));
                else
                    fprintf(fid, '%.4f', phases_processed(ch, i));
                end
            end
            if ch < 6
                fprintf(fid, '; ...\n');
            else
                fprintf(fid, '\n');
            end
        end
        fprintf(fid, '];\n\n');

        % Write usage examples
        fprintf(fid, '%% Usage Example:\n');
        fprintf(fid, '%% 1. Plot magnitude (linear) for all channels:\n');
        fprintf(fid, '%%    figure; loglog(frequencies, magnitudes_linear''); legend(''P1'',''P2'',''P3'',''P4'',''P5'',''P6'');\n');
        fprintf(fid, '%% 2. Plot magnitude (dB) for channel 1:\n');
        fprintf(fid, '%%    figure; semilogx(frequencies, 20*log10(magnitudes_linear(1,:)));\n');
        fprintf(fid, '%% 3. Plot processed phase for all channels:\n');
        fprintf(fid, '%%    figure; semilogx(frequencies, phases_processed''); legend(''P1'',''P2'',''P3'',''P4'',''P5'',''P6'');\n');

        fclose(fid);
        fprintf('    File saved successfully: %s\n', filename);

    catch ME
        fclose(fid);
        rethrow(ME);
    end
end
