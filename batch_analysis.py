#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次信號分析腳本
用於處理多個數據檔案
"""

import os
import subprocess
import sys

def batch_analyze_files(file_list, sampling_freq=100000, periods=10, average_periods=5):
    """
    批次分析多個檔案
    
    Args:
        file_list: 檔案路徑列表
        sampling_freq: 採樣頻率
        periods: 顯示週期數
        average_periods: 平均週期數
    """
    print("批次信號分析開始")
    print("=" * 50)
    
    for i, file_path in enumerate(file_list, 1):
        if not os.path.exists(file_path):
            print(f"[{i}/{len(file_list)}] 跳過: {file_path} (檔案不存在)")
            continue
            
        print(f"\n[{i}/{len(file_list)}] 處理: {file_path}")
        print("-" * 30)
        
        try:
            # 執行分析
            cmd = [
                sys.executable, "signal_visualization_final.py",
                "-f", file_path,
                "-s", str(sampling_freq),
                "-p", str(periods),
                "-a", str(average_periods)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ 成功處理: {file_path}")
            else:
                print(f"✗ 處理失敗: {file_path}")
                print(f"錯誤: {result.stderr}")
                
        except Exception as e:
            print(f"✗ 執行錯誤: {file_path}")
            print(f"錯誤: {e}")
    
    print("\n批次分析完成！")

def main():
    """主程式"""
    # 您可以在這裡列出要處理的檔案
    files_to_analyze = [
        "pi10.dat",
        "pi50.dat", 
        "pi100.dat",
        "dc10.dat",
        "dc50.dat",
        # 添加更多檔案...
    ]
    
    # 或者從目錄中自動找到所有.dat檔案
    def find_dat_files(directory="."):
        """找到目錄中所有的.dat檔案"""
        dat_files = []
        for file in os.listdir(directory):
            if file.endswith('.dat'):
                dat_files.append(file)
        return sorted(dat_files)
    
    # 使用自動找到的檔案（取消註解下面這行）
    # files_to_analyze = find_dat_files()
    
    if not files_to_analyze:
        print("沒有找到要處理的檔案")
        return
    
    print(f"找到 {len(files_to_analyze)} 個檔案要處理:")
    for file in files_to_analyze:
        print(f"  - {file}")
    
    # 開始批次處理
    batch_analyze_files(files_to_analyze)

if __name__ == "__main__":
    main() 