# OpenLoop Bode Analysis - Enhancement Plan

**Project:** Openloop_cali
**Created:** 2025-10-09
**Status:** Planning Phase

---

## 📋 Overview

This document outlines the planned enhancements for `openloop_bode.m`. These improvements will be implemented in a **separate testing project** first to ensure stability before merging into production.

---

## 🎯 Enhancement Goals

1. **FFT Spectrum Visualization** - Add diagnostic plots to verify frequency analysis correctness
2. **Terminal Output Management** - Implement verbosity levels for cleaner, more organized console output
3. **Adaptive Threshold Detection** - Support percentage-based steady-state detection with comparison mode

---

## 📊 Task Breakdown

### **Phase 1: FFT Spectrum Visualization** ⭐ Priority 1

#### **Objective**
Add optional FFT spectrum plots to verify that the target frequency is correctly identified and dominant in both DA and VM signals.

#### **Tasks**

- [ ] **Task 1.1: Add configuration parameters to SECTION 1**
  ```matlab
  % --- FFT Spectrum Diagnostic Settings ---
  PLOT_FFT_SPECTRUM = false;              % Enable FFT spectrum visualization
  PLOT_FFT_CHANNELS = [1];                % Channels to plot: 0=all, [1,2]=specific
  PLOT_FFT_FREQUENCIES = [];              % Frequencies to plot: empty=all, [0.1,1.0]=specific
  FFT_PLOT_FREQ_RANGE = [0.01, 1000];     % Frequency range to display (Hz)
  FFT_PLOT_SHOW_HARMONICS = true;         % Highlight harmonics of target frequency
  ```

- [ ] **Task 1.2: Create `plot_fft_spectrum` function**
  - **Location:** Add to SECTION 5.5 (Visualization)
  - **Function signature:**
    ```matlab
    function plot_fft_spectrum(vm_fft, da_fft, freq_axis, target_freq,
                                excite_ch, plot_channels, show_harmonics)
    ```

  - **Inputs:**
    - `vm_fft`: FFT result for VM channels (6 x N complex array)
    - `da_fft`: FFT result for DA excitation channel (1 x N complex array)
    - `freq_axis`: Frequency axis vector (Hz)
    - `target_freq`: Target excitation frequency (Hz)
    - `excite_ch`: Excitation channel index (1-6)
    - `plot_channels`: Channels to display
    - `show_harmonics`: Boolean flag to mark harmonics

  - **Outputs:**
    - Figure with subplots showing frequency spectrum

- [ ] **Task 1.3: Design figure layout**

  **Layout Structure:**
  ```
  ┌─────────────────────────────────────────────────────┐
  │ FFT Spectrum Analysis @ 0.1 Hz                      │
  ├─────────────────────────────────────────────────────┤
  │ DA Channel Spectrum (Top subplot)                   │
  │ - Magnitude vs Frequency (log scale)                │
  │ - Red vertical line at target frequency             │
  │ - Text annotation: "Target: 0.1 Hz, Mag: 1.24 V"   │
  │ - Optional: Mark 2nd, 3rd harmonics if enabled      │
  ├─────────────────────────────────────────────────────┤
  │ VM Channels Spectrum (Bottom subplots)              │
  │ - Grid layout: 2x3 for 6 channels OR single channel │
  │ - Same format as DA subplot                         │
  │ - Text annotation: "|H|=2.34, ∠H=-45.2°"           │
  │ - Highlight SNR or noise floor if possible          │
  └─────────────────────────────────────────────────────┘
  ```

- [ ] **Task 1.4: Implement diagnostic checks**
  - Verify target frequency bin is correct (print warning if mismatch)
  - Calculate SNR: ratio of target bin magnitude to mean noise floor
  - Detect if other significant frequency components exist (>10% of target)
  - Print diagnostic summary to console:
    ```
    FFT Diagnostics:
      Target frequency: 0.100 Hz (bin 123)
      Frequency match: ✓ (error < 0.01 Hz)
      DA magnitude: 1.24 V
      VM CH1 SNR: 45.2 dB
      VM CH1 |H|: 2.34, ∠H: -45.2°
      Harmonics detected: None
    ```

- [ ] **Task 1.5: Integrate into main processing loop**
  - Add call to `plot_fft_spectrum` in SECTION 3 after FFT analysis
  - Use `should_plot_frequency` helper to filter by frequency
  - Conditional execution based on `PLOT_FFT_SPECTRUM` flag

- [ ] **Task 1.6: Testing checklist**
  - [ ] Test with single frequency file
  - [ ] Test with all frequencies (ensure not too many figures)
  - [ ] Test with specific channel selection
  - [ ] Verify harmonic detection works
  - [ ] Confirm SNR calculation is reasonable

---

### **Phase 2: Terminal Output Management** ⭐ Priority 2

#### **Objective**
Implement a verbosity level system to control console output detail, making it easier to focus on relevant information during different stages of development and debugging.

#### **Tasks**

- [ ] **Task 2.1: Add verbosity configuration to SECTION 1**
  ```matlab
  % --- Console Output Settings ---
  VERBOSE_LEVEL = 2;  % 0=minimal, 1=normal, 2=detailed, 3=debug
  ```

- [ ] **Task 2.2: Create helper function `vprintf`**
  ```matlab
  function vprintf(level, required_level, format, varargin)
  % VPRINTF Verbosity-controlled fprintf
  %   Only prints if current VERBOSE_LEVEL >= required_level
      if level >= required_level
          fprintf(format, varargin{:});
      end
  end
  ```
  - **Location:** Add to SECTION 5 (Helper Functions)
  - Make it accessible to all functions (nested or pass as parameter)

- [ ] **Task 2.3: Define output message levels**

  **Level 0: Minimal** - Production mode, only critical info
  - Overall progress: `[1/10] 0.1Hz.csv ✓`
  - Final summary
  - Errors only

  **Level 1: Normal** - Default user mode
  - File processing start/end
  - Excitation detection result
  - Steady-state detection result
  - Success/failure status

  **Level 2: Detailed** - Current default behavior
  - All Level 1 messages
  - Step-by-step progress (Step 1-5)
  - Key parameter values
  - Warnings

  **Level 3: Debug** - Development/troubleshooting
  - All Level 2 messages
  - Interpolation RMS errors
  - FFT bin validation details
  - Transfer function values
  - Diagnostic calculations

- [ ] **Task 2.4: Refactor existing print statements**

  Systematically update all `fprintf` calls in these sections:

  - [ ] SECTION 2: Initialization
    - Level 0: "=== OPENLOOP BODE ANALYSIS ==="
    - Level 1: "Data folder:", "Found X files"

  - [ ] SECTION 3: Main processing loop
    - Level 0: "[X/Y] filename ✓/✗"
    - Level 1: "Processing: filename"
    - Level 2: "Step 1-5: ..." (current behavior)
    - Level 3: Add detailed diagnostics

  - [ ] `load_csv_data` function
    - Level 3: Data matrix dimensions

  - [ ] `repair_data_points` function
    - Level 2: Interpolation method
    - Level 3: RMS errors, point counts

  - [ ] `detect_excitation` function
    - Level 1: Final result
    - Level 3: Energy calculations per channel

  - [ ] `detect_steady_state` function
    - Level 2: Period samples, max periods
    - Level 2: Final detection result
    - Level 3: Per-channel stability check details

  - [ ] `perform_fft_analysis` function
    - Level 2: FFT mode, period count
    - Level 3: Frequency bin validation
    - Level 3: Transfer function values per channel

  - [ ] SECTION 4: Post-processing
    - Level 1: Normalization, summary
    - Level 2: Per-channel normalization values

- [ ] **Task 2.5: Add progress indicator for Level 0**
  ```matlab
  % For minimal output, use compact progress format
  % [1/10] 0.1Hz.csv ✓  [2/10] 0.5Hz.csv ✓  [3/10] 1.0Hz.csv ✗
  ```

- [ ] **Task 2.6: Format output with visual separators**
  - Use consistent indentation (2 spaces per level)
  - Add section separators for Level 1+
  - Use symbols: ✓ (success), ✗ (failure), ⚠ (warning)

- [ ] **Task 2.7: Testing checklist**
  - [ ] Test each verbosity level (0-3)
  - [ ] Verify output is clean and readable at each level
  - [ ] Ensure no critical information is lost at Level 0
  - [ ] Confirm debug details appear only at Level 3

---

### **Phase 3: Adaptive Threshold Detection** ⭐ Priority 3

#### **Objective**
Implement percentage-based steady-state detection threshold with a comparison mode to help users choose between fixed and adaptive methods.

#### **Tasks**

- [ ] **Task 3.1: Add configuration parameters to SECTION 1**
  ```matlab
  % --- Steady-State Detection Mode ---
  STABILITY_MODE = 'fixed';  % 'fixed' | 'percentage' | 'comparison'

  % Fixed threshold mode (original)
  STABILITY_THRESHOLD_FIXED = 2e-3;  % Absolute threshold (V)

  % Percentage threshold mode
  STABILITY_THRESHOLD_PERCENT = 0.05;  % Threshold as % of peak-to-peak
  PERCENTAGE_BASE = 'peak_to_peak';    % 'peak_to_peak' | 'rms' (future option)

  % Comparison mode settings
  COMPARISON_SHOW_DETAILS = true;  % Show per-channel comparison
  ```

- [ ] **Task 3.2: Modify `detect_steady_state` function signature**
  ```matlab
  function steady_info = detect_steady_state(vm_signal, target_freq, sampling_rate, ...
      stability_mode, threshold_fixed, threshold_percent, ...
      consecutive_periods, check_points, start_period)
  ```

  - Add `stability_mode` parameter
  - Add `threshold_fixed` parameter
  - Add `threshold_percent` parameter
  - Keep backward compatibility

- [ ] **Task 3.3: Implement percentage threshold calculation**

  Add new internal function:
  ```matlab
  function threshold_value = calculate_adaptive_threshold(signal, period_samples, threshold_percent)
  % Calculate threshold based on first period peak-to-peak
      if length(signal) < period_samples
          threshold_value = Inf;  % Invalid
          return;
      end

      first_period = signal(1:period_samples);
      peak_to_peak = max(first_period) - min(first_period);
      threshold_value = peak_to_peak * (threshold_percent / 100);
  end
  ```

- [ ] **Task 3.4: Implement comparison mode logic**

  **When `STABILITY_MODE = 'comparison'`:**

  1. Run steady-state detection with **fixed** threshold
     - Store result: `steady_info_fixed`

  2. Run steady-state detection with **percentage** threshold
     - Store result: `steady_info_percentage`

  3. Print comparison report:
     ```
     ┌─────────────────────────────────────────────────────┐
     │ STEADY-STATE COMPARISON @ 0.1 Hz                    │
     ├─────────────────────────────────────────────────────┤
     │ [Method 1: Fixed Threshold = 2.000 mV]              │
     │   CH1: Period 3, P2P=8.24V, Threshold=2.00mV (0.024%)│
     │   CH2: Period 3, P2P=7.98V, Threshold=2.00mV (0.025%)│
     │   ...                                               │
     │   Final Decision: Period 3                          │
     ├─────────────────────────────────────────────────────┤
     │ [Method 2: Percentage Threshold = 0.050%]           │
     │   CH1: Period 5, P2P=8.24V, Threshold=4.12mV        │
     │   CH2: Period 4, P2P=7.98V, Threshold=3.99mV        │
     │   ...                                               │
     │   Final Decision: Period 5                          │
     ├─────────────────────────────────────────────────────┤
     │ ⚠ DIFFERENCE: Fixed detected earlier (Δ=-2 periods)│
     │ → Using FIXED method for FFT analysis               │
     └─────────────────────────────────────────────────────┘
     ```

  4. Use one method for actual FFT (default: fixed)
     - Could add parameter `COMPARISON_USE_METHOD = 'fixed'` to choose

- [ ] **Task 3.5: Update main processing loop**
  - Pass new parameters to `detect_steady_state`
  - Handle comparison mode output
  - Conditional printing based on `VERBOSE_LEVEL`

- [ ] **Task 3.6: Add summary statistics (optional enhancement)**

  After processing all files, print summary:
  ```
  ┌────────────────────────────────────────────────┐
  │ THRESHOLD COMPARISON SUMMARY (10 frequencies)  │
  ├────────────────────────────────────────────────┤
  │ Fixed method earlier:      6 files (avg -2.3p)│
  │ Percentage method earlier: 3 files (avg +1.7p)│
  │ Same result:               1 files            │
  │                                                │
  │ Fixed fallback count:      2                   │
  │ Percentage fallback count: 0                   │
  │                                                │
  │ Recommendation: Consider switching to          │
  │ percentage mode (reduces fallback cases)       │
  └────────────────────────────────────────────────┘
  ```

- [ ] **Task 3.7: Testing checklist**
  - [ ] Test `'fixed'` mode (should behave identically to current code)
  - [ ] Test `'percentage'` mode with different percentages
  - [ ] Test `'comparison'` mode output formatting
  - [ ] Verify percentage calculation for various signal amplitudes
  - [ ] Test edge cases: very small signals, very large signals
  - [ ] Confirm fallback behavior in both modes

---

## 🧪 Testing Strategy

### **Test Environment Setup**

1. **Create separate testing branch/folder**
   ```
   Openloop_cali_test/
   ├── openloop_bode.m (modified)
   ├── processed_csv/
   │   └── P1/ (copy test data)
   └── test_results/
       ├── fft_spectrum_plots/
       ├── output_logs/
       └── comparison_reports/
   ```

2. **Prepare test datasets**
   - Small dataset: 3-5 frequency points for quick iteration
   - Full dataset: All frequencies for final validation
   - Edge cases: Very low frequency, very high frequency

### **Phase Testing**

**Phase 1 Testing:**
- Run with `PLOT_FFT_SPECTRUM = true` on small dataset
- Verify all spectrum plots are correct
- Check diagnostic messages
- Validate SNR calculations

**Phase 2 Testing:**
- Test each verbosity level independently
- Compare output length and clarity
- Ensure no information loss

**Phase 3 Testing:**
- Run comparison mode on full dataset
- Analyze which frequencies benefit from percentage mode
- Fine-tune percentage value based on results
- Document recommended settings

### **Integration Testing**

- [ ] Run all three enhancements together
- [ ] Test with `VERBOSE_LEVEL=0` + `PLOT_FFT_SPECTRUM=true`
- [ ] Test with `VERBOSE_LEVEL=3` + `STABILITY_MODE='comparison'`
- [ ] Verify no performance degradation
- [ ] Confirm output files (P1.m) are identical to original

---

## 📝 Documentation Updates

After successful testing, update documentation:

- [ ] Update header comments in `openloop_bode.m`
- [ ] Add usage examples for new parameters
- [ ] Document verbosity levels in detail
- [ ] Provide guidance on choosing threshold mode
- [ ] Add troubleshooting section for FFT spectrum interpretation

---

## ✅ Merge Criteria

Before merging to production:

- [ ] All phase tests pass
- [ ] Code review completed
- [ ] No breaking changes to existing functionality
- [ ] Performance impact < 10%
- [ ] Documentation updated
- [ ] Default parameters set to safe values:
  - `VERBOSE_LEVEL = 1` (not 2)
  - `PLOT_FFT_SPECTRUM = false`
  - `STABILITY_MODE = 'fixed'` (maintain current behavior)

---

## 📅 Estimated Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: FFT Spectrum | 6 tasks | 3-4 hours |
| Phase 2: Output Management | 7 tasks | 2-3 hours |
| Phase 3: Adaptive Threshold | 7 tasks | 4-5 hours |
| Testing & Integration | - | 2-3 hours |
| **Total** | **20 tasks** | **11-15 hours** |

---

## 📌 Notes & Considerations

### **Backward Compatibility**
- All new features are **opt-in** via configuration flags
- Default behavior matches current implementation
- Existing scripts/workflows will not break

### **Performance Considerations**
- FFT spectrum plotting may slow down processing if enabled for all frequencies
- Recommend using `PLOT_FFT_FREQUENCIES` to limit plots to specific cases
- Comparison mode runs detection twice → ~2x time for that step

### **Future Enhancements** (Not in current scope)
- Save comparison results to CSV for batch analysis
- Auto-detect optimal percentage threshold
- RMS-based percentage threshold (alternative to peak-to-peak)
- Interactive mode to adjust threshold on-the-fly

---

## 🔗 References

- Original script: `openloop_bode.m`
- Related files: `hsdata_reader.py`, Bode plot output `P1.m`
- MATLAB documentation: FFT, semilogx, subplot functions

---

**End of Enhancement Plan**
