#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信號視覺化分析系統 (最終版)
用於生成VD/VM、VM/DA和DA時域的響應圖
"""

from hsdata_reader import HSDataReader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import List, Dict, Tuple

class SignalVisualizer:
    """信號視覺化分析器"""
    
    def __init__(self, sampling_freq=100000):
        """
        初始化視覺化器
        
        Args:
            sampling_freq (int): 採樣頻率 (Hz)
        """
        self.sampling_freq = sampling_freq
        
    def extract_frequency_from_filename(self, filename: str) -> float:
        """從檔名提取頻率"""
        import re
        # 嘗試匹配多種格式
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
        
        # 如果都無法匹配，嘗試從檔案內容推測頻率
        print(f"警告: 無法從檔名 {filename} 提取頻率，嘗試從檔案內容推測...")
        return self.estimate_frequency_from_content(filename)
    
    def estimate_frequency_from_content(self, filename: str) -> float:
        """
        從檔案內容推測頻率
        這是一個簡化的推測方法，實際使用時可能需要更複雜的算法
        """
        try:
            # 讀取一小部分數據來推測頻率
            reader = HSDataReader(filename)
            records = reader.read_data_records()
            
            if len(records) < 1000:
                raise ValueError("數據不足，無法推測頻率")
            
            # 使用第一個VM通道的數據進行頻率推測
            vm_signal = np.array([record['vm'][0] for record in records[:10000]])
            
            # 使用FFT推測主頻率
            fft_result = np.fft.fft(vm_signal)
            freq_axis = np.fft.fftfreq(len(vm_signal), 1/self.sampling_freq)
            
            # 只考慮正頻率部分
            positive_freqs = freq_axis[:len(freq_axis)//2]
            positive_fft = np.abs(fft_result[:len(fft_result)//2])
            
            # 找到最大幅值對應的頻率
            max_idx = np.argmax(positive_fft)
            estimated_freq = positive_freqs[max_idx]
            
            print(f"推測頻率: {estimated_freq:.1f}Hz")
            return estimated_freq
            
        except Exception as e:
            print(f"頻率推測失敗: {e}")
            # 如果推測失敗，使用預設頻率
            print("使用預設頻率: 10Hz")
            return 10.0
    
    def find_period_boundaries(self, signal_data: np.ndarray, target_freq: float, 
                              num_periods: int = 10) -> Tuple[int, int]:
        """
        找到指定週期數的數據邊界
        
        Args:
            signal_data: 信號數據
            target_freq: 目標頻率
            num_periods: 週期數
            
        Returns:
            (start_idx, end_idx): 數據邊界
        """
        # 計算一個週期的採樣點數
        period_points = int(self.sampling_freq / target_freq)
        
        # 計算總需要的點數
        total_points = period_points * num_periods
        
        # 確保不超過數據長度
        if total_points > len(signal_data):
            total_points = len(signal_data)
            num_periods = total_points // period_points
        
        # 從數據中間開始取，避免開始和結束的不穩定部分
        start_idx = (len(signal_data) - total_points) // 2
        end_idx = start_idx + total_points
        
        return start_idx, end_idx
    
    def average_periods(self, signal_data: np.ndarray, target_freq: float, 
                       num_periods: int = 5) -> np.ndarray:
        """
        對多個週期進行平均，並過濾異常週期（包含最大值與最大跳變檢查），
        嘗試所有可能的num_periods組合，選出最平滑（最大跳變最小）的平均結果。
        
        Args:
            signal_data: 信號數據
            target_freq: 目標頻率
            num_periods: 用於平均的週期數
            
        Returns:
            平均後的單週期數據
        """
        import itertools
        # 計算一個週期的採樣點數
        period_points = int(self.sampling_freq / target_freq)
        
        # 找到數據邊界
        start_idx, end_idx = self.find_period_boundaries(signal_data, target_freq, num_periods+5)
        data_segment = signal_data[start_idx:end_idx]
        
        # 確保數據長度是週期點數的整數倍
        num_complete_periods = len(data_segment) // period_points
        if num_complete_periods < num_periods:
            return data_segment
        
        # 重塑數據為週期矩陣
        periods_matrix = data_segment[:num_complete_periods * period_points].reshape(
            num_complete_periods, period_points)
        
        # 過濾異常週期（最大/最小值超過合理範圍，或最大跳變過大）
        reasonable_limit = 1.0  # 合理最大絕對值
        max_jump_limit = 0.2    # 合理最大跳變
        valid_periods = []
        for period in periods_matrix:
            max_abs = np.max(np.abs(period))
            max_jump = np.max(np.abs(np.diff(period)))
            if (max_abs < reasonable_limit) and (max_jump < max_jump_limit):
                valid_periods.append(period)
        if len(valid_periods) < num_periods:
            valid_periods = periods_matrix
        else:
            valid_periods = np.array(valid_periods)
        
        # 嘗試所有num_periods組合，選最大跳變最小的
        best_score = np.inf
        best_avg = None
        n = valid_periods.shape[0]
        for idxs in itertools.combinations(range(n), num_periods):
            group = valid_periods[list(idxs)]
            avg = np.median(group, axis=0)
            score = np.max(np.abs(np.diff(avg)))
            if score < best_score:
                best_score = score
                best_avg = avg
        if best_avg is not None:
            return best_avg
        else:
            # 萬一都不行，退回全部平均
            return np.median(valid_periods, axis=0)
    
    def convert_da_to_current(self, da_data: np.ndarray) -> np.ndarray:
        """
        將DA數據轉換為電流
        
        Args:
            da_data: DA數據 (16bit, 0-65535)
            
        Returns:
            電流數據 (A)
        """
        # DA轉換參數
        da_max = 65535  # 16bit最大值
        voltage_range = 20  # ±10V = 20V總範圍
        current_ratio = 0.6  # 電壓到電流的轉換比例
        
        # 轉換步驟：
        # 1. DA值 -> 電壓 (-10V to +10V)
        voltage = (da_data / da_max - 0.5) * voltage_range
        
        # 2. 電壓 -> 電流
        current = voltage * current_ratio
        
        return current
    
    def load_data(self, file_path: str) -> Tuple[List[Dict], float]:
        """
        載入數據並提取頻率
        
        Args:
            file_path: 數據檔案路徑
            
        Returns:
            (records, target_freq): 數據記錄和目標頻率
        """
        print(f"正在載入數據: {file_path}")
        
        # 讀取數據
        reader = HSDataReader(file_path)
        records = reader.read_data_records()
        
        # 提取頻率
        filename = os.path.basename(file_path)
        target_freq = self.extract_frequency_from_filename(filename)
        
        print(f"目標頻率: {target_freq}Hz")
        print(f"數據長度: {len(records)} 記錄")
        
        return records, target_freq
    
    def plot_vd_vm_overlay(self, records: List[Dict], target_freq: float, num_periods: int = 5):
        """
        繪製VD/VM疊圖 (縱軸VM，橫軸VD，六個通道疊圖)
        
        Args:
            records: 數據記錄
            target_freq: 目標頻率
            num_periods: 用於平均的週期數
        """
        print(f"正在分析VD/VM疊圖...")
        
        # 找到能量最大的VD通道作為輸入
        vd_energy = np.zeros(6)
        for i in range(6):
            vd_signal = np.array([record['vd'][i] for record in records])
            vd_energy[i] = np.sum(vd_signal**2)
        
        input_channel = np.argmax(vd_energy)
        print(f"選擇VD{input_channel}作為輸入通道")
        
        # 提取輸入信號並平均
        vd_input = np.array([record['vd'][input_channel] for record in records])
        vd_averaged = self.average_periods(vd_input, target_freq, num_periods)
        
        # 創建圖形
        plt.figure(figsize=(12, 8))
        plt.title(f'VM/VD Response at {target_freq}Hz)', 
                 fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # 繪製每個VM通道
        for vm_channel in range(6):
            # 提取並平均VM信號
            vm_signal = np.array([record['vm'][vm_channel] for record in records])
            vm_averaged = self.average_periods(vm_signal, target_freq, num_periods)
            
            # 繪製VD vs VM
            plt.plot(vd_averaged, vm_averaged, color=colors[vm_channel], 
                    linewidth=2, label=f'ch{vm_channel+1}')
        
        plt.xlabel('VD (V)')
        plt.ylabel('VM (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')  # 保持縱橫比
        plt.show()
        
        return vd_averaged, [np.array([record['vm'][i] for record in records]) for i in range(6)]
    
    def plot_vm_da_separate(self, records: List[Dict], target_freq: float, num_periods: int = 10):
        """
        繪製VM/DA分開圖 (縱軸VM，橫軸DA，六個通道分開呈現)
        
        Args:
            records: 數據記錄
            target_freq: 目標頻率
            num_periods: 顯示的週期數
        """
        print(f"正在分析VM/DA分開圖...")
        
        # 找到數據邊界
        vm_signal = np.array([record['vm'][0] for record in records])
        start_idx, end_idx = self.find_period_boundaries(vm_signal, target_freq, num_periods)
        
        # 創建圖形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'VM vs Control Effort Current at {target_freq}Hz', fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # 繪製每個通道
        for channel in range(6):
            row = channel // 3
            col = channel % 3
            ax = axes[row, col]
            
            # 提取VM數據
            vm_signal = np.array([record['vm'][channel] for record in records])
            vm_segment = vm_signal[start_idx:end_idx]
            
            # 提取並轉換DA數據
            da_signal = np.array([record['da'][channel] for record in records])
            da_current = self.convert_da_to_current(da_signal)
            da_segment = da_current[start_idx:end_idx]
            
            # 繪製VM vs DA
            ax.plot(da_segment, vm_segment, color=colors[channel], linewidth=1.5, alpha=0.8)
            ax.set_title(f'ch{channel+1}')
            ax.set_xlabel('Current (A)')
            ax.set_ylabel('VM Voltage (V)')
            ax.grid(True, alpha=0.3)
            
            # 計算並顯示統計信息
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
        繪製DA時域圖 (固定10個週期的DA輸出，以電流為單位)
        
        Args:
            records: 數據記錄
            target_freq: 目標頻率
            num_periods: 顯示的週期數
        """
        print(f"正在分析DA時域響應...")
        
        # 找到數據邊界
        vm_signal = np.array([record['vm'][0] for record in records])
        start_idx, end_idx = self.find_period_boundaries(vm_signal, target_freq, num_periods)
        
        # 創建時間軸
        period_points = int(self.sampling_freq / target_freq)
        total_points = end_idx - start_idx
        time_axis = np.linspace(0, total_points / self.sampling_freq, total_points)
        
        # 創建圖形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Control Effort Current Output at {target_freq}Hz', fontsize=16)
        
        colors = ['black', 'blue', 'green', 'red', 'magenta', 'cyan']
        
        # 繪製每個通道
        for channel in range(6):
            row = channel // 3
            col = channel % 3
            ax = axes[row, col]
            
            # 提取並轉換DA數據
            da_signal = np.array([record['da'][channel] for record in records])
            da_current = self.convert_da_to_current(da_signal)
            da_segment = da_current[start_idx:end_idx]
            
            # 繪製DA電流時域響應
            ax.plot(time_axis * 1000, da_segment, color=colors[channel], linewidth=1.5)
            ax.set_title(f'ch{channel+1}')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('DA Current (A)')
            ax.grid(True, alpha=0.3)
            
            # 計算並顯示統計信息
            da_pp = np.max(da_segment) - np.min(da_segment)
            ax.text(0.02, 0.98, f'P-P: {da_pp:.3f}A', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return da_current

def main():
    """主程式"""
    import argparse
    import sys
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='信號視覺化分析系統')
    parser.add_argument('--file', '-f', type=str, default='pi10.dat',
                       help='數據檔案路徑 (預設: 您的檔案名.dat)')
    parser.add_argument('--sampling-freq', '-s', type=int, default=100000,
                       help='採樣頻率 (預設: 100000 Hz)')
    parser.add_argument('--periods', '-p', type=int, default=10,
                       help='顯示週期數 (預設: 10)')
    parser.add_argument('--average-periods', '-a', type=int, default=5,
                       help='平均週期數 (預設: 5)')
    
    args = parser.parse_args()
    
    print("信號視覺化分析系統 (最終版)")
    print("=" * 50)
    print(f"數據檔案: {args.file}")
    print(f"採樣頻率: {args.sampling_freq} Hz")
    print(f"顯示週期數: {args.periods}")
    print(f"平均週期數: {args.average_periods}")
    print("=" * 50)
    
    # 初始化視覺化器
    visualizer = SignalVisualizer(sampling_freq=args.sampling_freq)
    
    # 檢查檔案是否存在
    if not os.path.exists(args.file):
        print(f"錯誤: 找不到檔案 {args.file}")
        print("請檢查檔案路徑是否正確")
        return
    
    try:
        # 載入數據（只讀取一次）
        records, target_freq = visualizer.load_data(args.file)
        
        # 1. 繪製VD/VM疊圖
        print("\n1. 生成VD/VM疊圖...")
        vd_averaged, vm_signals = visualizer.plot_vd_vm_overlay(
            records, target_freq, num_periods=args.average_periods)
        
        # 2. 繪製VM/DA分開圖
        print("\n2. 生成VM/DA分開圖...")
        vm_signal, da_current = visualizer.plot_vm_da_separate(
            records, target_freq, num_periods=args.periods)
        
        # 3. 繪製DA時域圖
        print("\n3. 生成DA時域圖...")
        da_current_full = visualizer.plot_da_time_domain(
            records, target_freq, num_periods=args.periods)
        
        print("\n分析完成！")
        print(f"成功處理檔案: {args.file}")
        print(f"目標頻率: {target_freq}Hz")
        print(f"數據記錄數: {len(records)}")
        
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 