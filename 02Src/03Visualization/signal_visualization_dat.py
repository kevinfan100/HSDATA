#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Visualization Analysis System (Final Version)
Generate VD/VM and VM/DA response plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import List, Dict, Tuple
import re
import argparse
import importlib.util
from pathlib import Path

# Dynamic import HSDataReader
spec = importlib.util.spec_from_file_location("hsdata_reader", Path(__file__).parent.parent / "01Core" / "hsdata_reader.py")
hsdata_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsdata_reader)
HSDataReader = hsdata_reader.HSDataReader

class SignalVisualizer:
    """Signal Visualization Analyzer"""
    
    def __init__(self, sampling_freq=100000):
        """
        Initialize visualizer
        
        Args:
            sampling_freq (int): Sampling frequency (Hz)
        """
        self.sampling_freq = sampling_freq
        
    def extract_frequency_from_filename(self, filename: str) -> float:
        """Extract frequency from filename"""
        import re
        # Try to match multiple formats
        patterns = [
            r'dc([0-9]+\.?[0-9]*)\.dat',  # dc10.dat
            r'pi([0-9]+\.?[0-9]*)\.dat',  # pi10.dat
            r'ndc(\d+)\.dat',             # ndc10.dat
            r'([0-9]+\.?[0-9]*)\.dat'     # 10.dat
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return float(match.group(1))
        
        # If none match, try to estimate frequency from file content
        print(f"Warning: Cannot extract frequency from filename {filename}, trying to estimate from content...")
        return self.estimate_frequency_from_content(filename)
    
    def estimate_frequency_from_content(self, filename: str) -> float:
        """
        Estimate frequency from file content
        This is a simplified estimation method, actual use may require more complex algorithms
        """
        try:
            # Read a small portion of data to estimate frequency
            reader = HSDataReader(filename)
            records = reader.read_data_records()
            
            if len(records) < 1000:
                raise ValueError("Insufficient data to estimate frequency")
            
            # Use first VM channel data for frequency estimation
            vm_signal = np.array([record['vm'][0] for record in records[:10000]])
            
            # Use FFT to estimate main frequency
            fft_result = np.fft.fft(vm_signal)
            freq_axis = np.fft.fftfreq(len(vm_signal), 1/self.sampling_freq)
            
            # Only consider positive frequency part
            positive_freqs = freq_axis[:len(freq_axis)//2]
            positive_fft = np.abs(fft_result[:len(freq_axis)//2])
            
            # Find frequency corresponding to maximum amplitude
            max_idx = np.argmax(positive_fft)
            estimated_freq = positive_freqs[max_idx]
            
            print(f"Estimated frequency: {estimated_freq:.1f}Hz")
            return estimated_freq
            
        except Exception as e:
            print(f"Frequency estimation failed: {e}")
            # If estimation fails, use default frequency
            print("Using default frequency: 10Hz")
            return 10.0
    
    def find_period_boundaries(self, signal_data: np.ndarray, target_freq: float, 
                              num_periods: int = 10) -> Tuple[int, int]:
        """
        Find data boundaries for specified number of periods
        
        Args:
            signal_data: Signal data
            target_freq: Target frequency
            num_periods: Number of periods
            
        Returns:
            (start_idx, end_idx): Data boundaries
        """
        # Calculate points per period
        period_points = int(self.sampling_freq / target_freq)
        
        # Calculate total points needed
        total_points = period_points * num_periods
        
        # Ensure not exceeding data length
        if total_points > len(signal_data):
            total_points = len(signal_data)
            num_periods = total_points // period_points
        
        # Start from middle to avoid unstable parts at beginning and end
        start_idx = (len(signal_data) - total_points) // 2
        end_idx = start_idx + total_points
        
        return start_idx, end_idx
    
    def average_periods(self, signal_data: np.ndarray, target_freq: float, 
                       num_periods: int = 5) -> np.ndarray:
        """
        Average multiple periods with proper algorithm
        
        Args:
            signal_data: Signal data
            target_freq: Target frequency
            num_periods: Number of periods for averaging
            
        Returns:
            Averaged single period data
        """
        # Calculate points per period
        period_points = int(self.sampling_freq / target_freq)
        
        # Find data boundaries
        start_idx, end_idx = self.find_period_boundaries(signal_data, target_freq, num_periods)
        data_segment = signal_data[start_idx:end_idx]
        
        # Ensure data length is integer multiple of period points
        num_complete_periods = len(data_segment) // period_points
        if num_complete_periods < 1:
            return data_segment
        
        # Reshape data into period matrix
        periods_matrix = data_segment[:num_complete_periods * period_points].reshape(
            num_complete_periods, period_points)
        
        # Use median for robust averaging
        averaged_period = np.median(periods_matrix, axis=0)
        
        return averaged_period
    
    def convert_da_to_current(self, da_data: np.ndarray) -> np.ndarray:
        """
        Convert DA data to current
        
        Args:
            da_data: DA data (16bit, 0-65535)
            
        Returns:
            Current data (A)
        """
        # DA conversion parameters - corrected
        da_max = 65535  # 16bit maximum value
        voltage_range = 10  # ±5V = 10V total range
        current_ratio = 0.2  # Voltage to current conversion ratio (A/V)
        
        # Conversion steps:
        # 1. DA value -> voltage (-5V to +5V)
        voltage = (da_data / da_max - 0.5) * voltage_range
        
        # 2. Voltage -> current
        current = voltage * current_ratio
        
        return current
    
    def load_data(self, file_path: str) -> Tuple[List[Dict], float]:
        """
        Load data and extract frequency
        
        Args:
            file_path: Data file path
            
        Returns:
            (records, target_freq): Data records and target frequency
        """
        print(f"Loading data: {file_path}")
        
        # Read data
        reader = HSDataReader(file_path)
        records = reader.read_data_records()
        
        # Skip first few records that might have initialization issues
        skip_count = 1000
        if len(records) > skip_count:
            records = records[skip_count:]
            print(f"Skipped first {skip_count} records to avoid initialization issues")
        
        # Debug: Check data ranges
        if records:
            vm_sample = records[0]['vm']
            vd_sample = records[0]['vd']
            da_sample = records[0]['da']
            print(f"Sample data ranges:")
            print(f"  VM: {np.min(vm_sample):.6f} to {np.max(vm_sample):.6f} V")
            print(f"  VD: {np.min(vd_sample):.6f} to {np.max(vd_sample):.6f} V")
            print(f"  DA: {np.min(da_sample)} to {np.max(da_sample)} (raw)")
        
        # Extract frequency
        filename = os.path.basename(file_path)
        target_freq = self.extract_frequency_from_filename(filename)
        
        print(f"Target frequency: {target_freq}Hz")
        print(f"Data length: {len(records)} records")
        
        return records, target_freq
    
    def plot_vd_vm_overlay(self, records: List[Dict], target_freq: float, num_periods: int = 5):
        """
        Plot VD/VM overlay (Y-axis VM, X-axis VD, six channels overlay)
        
        Args:
            records: Data records
            target_freq: Target frequency
            num_periods: Number of periods for averaging
        """
        print(f"Analyzing VD/VM overlay...")
        
        # Find channel with maximum energy as input
        vd_energy = np.zeros(6)
        for i in range(6):
            vd_signal = np.array([record['vd'][i] for record in records])
            vd_energy[i] = np.sum(vd_signal**2)
        
        input_channel = np.argmax(vd_energy)
        print(f"Selected VD{input_channel} as input channel")
        
        # Extract input signal and average
        vd_input = np.array([record['vd'][input_channel] for record in records])
        vd_averaged = self.average_periods(vd_input, target_freq, num_periods)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title(f'VM/VD Response at {target_freq}Hz', fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # Plot each VM channel
        for vm_channel in range(6):
            # Extract and average VM signal
            vm_signal = np.array([record['vm'][vm_channel] for record in records])
            vm_averaged = self.average_periods(vm_signal, target_freq, num_periods)
            
            # Plot VD vs VM
            plt.plot(vd_averaged, vm_averaged, color=colors[vm_channel], 
                    linewidth=2, label=f'ch{vm_channel+1}')
        
        plt.xlabel('VD (V)')
        plt.ylabel('VM (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')  # Keep aspect ratio
        plt.show()
        
        return vd_averaged, [np.array([record['vm'][i] for record in records]) for i in range(6)]
    
    def plot_vm_da_separate(self, records: List[Dict], target_freq: float, num_periods: int = 10):
        """
        Plot VM/DA separate (Y-axis VM, X-axis DA, six channels separate display)
        
        Args:
            records: Data records
            target_freq: Target frequency
            num_periods: Number of periods to display
        """
        print(f"Analyzing VM/DA separate...")
        
        # Find data boundaries
        vm_signal = np.array([record['vm'][0] for record in records])
        start_idx, end_idx = self.find_period_boundaries(vm_signal, target_freq, num_periods)
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'VM vs Control Effort Current at {target_freq}Hz', fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # Plot each channel
        for channel in range(6):
            row = channel // 3
            col = channel % 3
            ax = axes[row, col]
            
            # Extract VM data
            vm_signal = np.array([record['vm'][channel] for record in records])
            vm_segment = vm_signal[start_idx:end_idx]
            
            # Extract and convert DA data
            da_signal = np.array([record['da'][channel] for record in records])
            da_current = self.convert_da_to_current(da_signal)
            da_segment = da_current[start_idx:end_idx]
            
            # Plot VM vs DA
            ax.plot(da_segment, vm_segment, color=colors[channel], linewidth=1.5, alpha=0.8)
            ax.set_title(f'ch{channel+1}')
            ax.set_xlabel('Current (A)')
            ax.set_ylabel('VM Voltage (V)')
            ax.grid(True, alpha=0.3)
            
            # Calculate and display statistics
            vm_pp = np.max(vm_segment) - np.min(vm_segment)
            da_pp = np.max(da_segment) - np.min(da_segment)
            ax.text(0.02, 0.98, f'VM P-P: {vm_pp:.3f}V\nDA P-P: {da_pp:.3f}A', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return vm_signal, da_current
    
    def plot_da_time_domain(self, records: List[Dict], target_freq: float, num_periods: int = 10):
        """
        Plot DA time domain (fixed 10 periods of DA output, in current units)
        
        Args:
            records: Data records
            target_freq: Target frequency
            num_periods: Number of periods to display
        """
        print(f"Analyzing DA time domain response...")
        
        # Find data boundaries
        vm_signal = np.array([record['vm'][0] for record in records])
        start_idx, end_idx = self.find_period_boundaries(vm_signal, target_freq, num_periods)
        
        # Create time axis
        period_points = int(self.sampling_freq / target_freq)
        total_points = end_idx - start_idx
        time_axis = np.linspace(0, total_points / self.sampling_freq, total_points)
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Control Effort Current Output at {target_freq}Hz', fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # Plot each channel
        for channel in range(6):
            row = channel // 3
            col = channel % 3
            ax = axes[row, col]
            
            # Extract and convert DA data
            da_signal = np.array([record['da'][channel] for record in records])
            da_current = self.convert_da_to_current(da_signal)
            da_segment = da_current[start_idx:end_idx]
            
            # Plot DA current time domain response
            ax.plot(time_axis * 1000, da_segment, color=colors[channel], linewidth=1.5)
            ax.set_title(f'ch{channel+1}')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('DA Current (A)')
            ax.grid(True, alpha=0.3)
            
            # Calculate and display statistics
            da_pp = np.max(da_segment) - np.min(da_segment)
            ax.text(0.02, 0.98, f'P-P: {da_pp:.3f}A', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return da_current

def main():
    """Main program"""
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Signal Visualization Analysis System')
    parser.add_argument('--file', '-f', type=str, default='pi10.dat',
                       help='Data file path (default: pi10.dat)')
    parser.add_argument('--sampling-freq', '-s', type=int, default=100000,
                       help='Sampling frequency (default: 100000 Hz)')
    parser.add_argument('--periods', '-p', type=int, default=10,
                       help='Display periods (default: 10)')
    parser.add_argument('--average-periods', '-a', type=int, default=5,
                       help='Average periods (default: 5)')
    
    args = parser.parse_args()
    
    print("Signal Visualization Analysis System (Final Version)")
    print("=" * 50)
    print(f"Data file: {args.file}")
    print(f"Sampling frequency: {args.sampling_freq} Hz")
    print(f"Display periods: {args.periods}")
    print(f"Average periods: {args.average_periods}")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = SignalVisualizer(sampling_freq=args.sampling_freq)
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found {args.file}")
        print("Please check the file path")
        return
    
    try:
        # Load data (only read once)
        records, target_freq = visualizer.load_data(args.file)
        
        # 1. Plot VD/VM overlay
        print("\n1. Generating VD/VM overlay...")
        vd_averaged, vm_signals = visualizer.plot_vd_vm_overlay(
            records, target_freq, num_periods=args.average_periods)
        
        # 2. Plot VM/DA separate
        print("\n2. Generating VM/DA separate...")
        vm_signal, da_current = visualizer.plot_vm_da_separate(
            records, target_freq, num_periods=args.periods)
        
        # 3. Plot DA time domain
        print("\n3. Generating DA time domain...")
        da_current_full = visualizer.plot_da_time_domain(
            records, target_freq, num_periods=args.periods)
        
        print("\nAnalysis complete!")
        print(f"Successfully processed file: {args.file}")
        print(f"Target frequency: {target_freq} Hz")
        print(f"Data records: {len(records)}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 