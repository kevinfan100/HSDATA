import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

def visualize_csv_data(csv_file, start_index=394106, num_samples=500, sampling_freq=100000):
    """
    Read CSV file and visualize VM, VD, DA data in three separate windows
    
    Parameters:
    csv_file: CSV file path
    start_index: Starting index number
    num_samples: Number of samples to extract
    sampling_freq: Sampling frequency (Hz)
    """
    
    print(f"Reading CSV file: {csv_file}")
    print(f"Starting from index {start_index}, extracting {num_samples} samples")
    print(f"Sampling frequency: {sampling_freq} Hz")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read CSV file, total {len(df)} records")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check index range
    if start_index >= len(df):
        print(f"Error: Starting index {start_index} is out of range (0-{len(df)-1})")
        return
    
    end_index = min(start_index + num_samples, len(df))
    actual_samples = end_index - start_index
    
    print(f"Actual sampling range: {start_index} to {end_index-1} ({actual_samples} samples)")
    
    # Extract specified range of data
    data_subset = df.iloc[start_index:end_index]
    
    # Separate VM, VD, DA data
    vm_columns = [col for col in df.columns if col.startswith('vm_')]
    vd_columns = [col for col in df.columns if col.startswith('vd_')]
    da_columns = [col for col in df.columns if col.startswith('da_')]
    
    print(f"Found columns:")
    print(f"VM columns: {vm_columns}")
    print(f"VD columns: {vd_columns}")
    print(f"DA columns: {da_columns}")
    
    # Create time axis (in seconds)
    time_axis = np.arange(actual_samples) / sampling_freq
    
    # Set up matplotlib for better display
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'CSV Data Visualization (Index {start_index} to {end_index-1}, Sampling Rate: {sampling_freq} Hz)', 
                 fontsize=16, fontweight='bold')
    
    # Plot VM data
    ax1.set_title('VM Data', fontsize=14, fontweight='bold')
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, col in enumerate(vm_columns):
        color = colors[i % len(colors)]
        ax1.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=1.5, color=color)
    ax1.set_ylabel('VM Values')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_axis[-1])
    
    # Plot VD data
    ax2.set_title('VD Data', fontsize=14, fontweight='bold')
    for i, col in enumerate(vd_columns):
        color = colors[i % len(colors)]
        ax2.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=1.5, color=color)
    ax2.set_ylabel('VD Values')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_axis[-1])
    
    # Plot DA data
    ax3.set_title('DA Data', fontsize=14, fontweight='bold')
    for i, col in enumerate(da_columns):
        color = colors[i % len(colors)]
        ax3.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=1.5, color=color)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('DA Values')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, time_axis[-1])
    
    plt.tight_layout()
    
    plt.show()
    
    # Display statistics
    print("\n=== Statistics ===")
    print(f"VM data range: {data_subset[vm_columns].min().min():.6f} to {data_subset[vm_columns].max().max():.6f}")
    print(f"VD data range: {data_subset[vd_columns].min().min():.6f} to {data_subset[vd_columns].max().max():.6f}")
    print(f"DA data range: {data_subset[da_columns].min().min():.0f} to {data_subset[da_columns].max().max():.0f}")
    
    # Calculate and display time duration
    duration = actual_samples / sampling_freq
    print(f"Time duration: {duration:.6f} seconds ({duration*1000:.3f} ms)")
    
    return data_subset

def create_individual_plots(csv_file, start_index=394106, num_samples=500, sampling_freq=100000):
    """
    Create individual plots for VM, VD, and DA data in separate windows
    """
    
    print(f"\nCreating individual plots...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    data_subset = df.iloc[start_index:start_index+num_samples]
    
    # Separate data
    vm_columns = [col for col in df.columns if col.startswith('vm_')]
    vd_columns = [col for col in df.columns if col.startswith('vd_')]
    da_columns = [col for col in df.columns if col.startswith('da_')]
    
    time_axis = np.arange(num_samples) / sampling_freq
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Create individual plots
    for data_type, columns in [('VM', vm_columns), ('VD', vd_columns), ('DA', da_columns)]:
        plt.figure(figsize=(12, 8))
        plt.title(f'{data_type} Data (Index {start_index} to {start_index+num_samples-1})', 
                 fontsize=16, fontweight='bold')
        
        for i, col in enumerate(columns):
            color = colors[i % len(colors)]
            plt.plot(time_axis, data_subset[col], label=f'ch{i+1}', linewidth=2, color=color)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel(f'{data_type} Values')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    # Set file path and parameters
    csv_file = "what_200_0.1_combined_20250717_192317.csv"
    start_index = 215791
    num_samples = 5000
    sampling_freq = 100000  # 100 kHz
    
    # Execute visualization
    data = visualize_csv_data(csv_file, start_index, num_samples, sampling_freq)
    
    # Create individual plots
    create_individual_plots(csv_file, start_index, num_samples, sampling_freq) 