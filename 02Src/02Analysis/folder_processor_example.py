#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folder Processor Usage Examples
批次處理器使用範例
"""

import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))
from folder_processor import FolderProcessor

def example_1_process_single_folder():
    """範例 1: 處理單一資料夾"""
    print("=== 範例 1: 處理單一資料夾 ===")
    
    processor = FolderProcessor()
    
    # 指定要處理的資料夾路徑
    folder_path = "01Data/01Raw_dat/0717_pi"
    
    # 處理資料夾
    results = processor.process_folder(folder_path)
    
    print(f"處理結果: {results}")
    print(f"成功處理: {processor.processed_count} 個檔案")
    print(f"處理失敗: {processor.error_count} 個檔案")

def example_2_process_all_raw_folders():
    """範例 2: 處理所有 raw data 資料夾"""
    print("\n=== 範例 2: 處理所有 raw data 資料夾 ===")
    
    processor = FolderProcessor()
    
    # 處理所有 raw data 資料夾
    results = processor.process_all_raw_folders()
    
    print(f"總處理結果: {results}")
    print(f"總成功處理: {processor.processed_count} 個檔案")
    print(f"總處理失敗: {processor.error_count} 個檔案")

def example_3_process_specific_folders():
    """範例 3: 處理指定的資料夾"""
    print("\n=== 範例 3: 處理指定的資料夾 ===")
    
    processor = FolderProcessor()
    
    # 指定要處理的資料夾名稱
    folder_names = ["0715_ndc_data", "0716_hsdata_dc"]
    
    # 處理指定的資料夾
    results = processor.process_specific_folders(folder_names)
    
    print(f"處理結果: {results}")
    print(f"成功處理: {processor.processed_count} 個檔案")
    print(f"處理失敗: {processor.error_count} 個檔案")

def example_4_get_folder_info():
    """範例 4: 取得資料夾資訊"""
    print("\n=== 範例 4: 取得資料夾資訊 ===")
    
    processor = FolderProcessor()
    
    # 取得資料夾資訊
    folder_path = "01Data/01Raw_dat/0715_ndc_data"
    info = processor.get_folder_info(folder_path)
    
    print(f"資料夾資訊: {info}")

def example_5_interactive_usage():
    """範例 5: 互動式使用"""
    print("\n=== 範例 5: 互動式使用 ===")
    
    processor = FolderProcessor()
    
    print("請選擇操作:")
    print("1. 處理單一資料夾")
    print("2. 處理所有 raw data 資料夾")
    print("3. 處理指定的資料夾")
    print("4. 取得資料夾資訊")
    
    choice = input("請輸入選項 (1-4): ").strip()
    
    if choice == "1":
        folder_path = input("請輸入資料夾路徑: ").strip()
        results = processor.process_folder(folder_path)
        print(f"處理結果: {results}")
        
    elif choice == "2":
        results = processor.process_all_raw_folders()
        print(f"處理結果: {results}")
        
    elif choice == "3":
        folder_names_input = input("請輸入資料夾名稱 (用逗號分隔): ").strip()
        folder_names = [name.strip() for name in folder_names_input.split(",")]
        results = processor.process_specific_folders(folder_names)
        print(f"處理結果: {results}")
        
    elif choice == "4":
        folder_path = input("請輸入資料夾路徑: ").strip()
        info = processor.get_folder_info(folder_path)
        print(f"資料夾資訊: {info}")
        
    else:
        print("無效的選項")

if __name__ == "__main__":
    print("HSData Folder Processor Usage Examples")
    print("HSData 批次處理器使用範例")
    print("=" * 50)
    
    # 執行範例
    example_1_process_single_folder()
    # example_2_process_all_raw_folders()
    # example_3_process_specific_folders()
    # example_4_get_folder_info()
    
    # 互動式使用 (可選)
    # example_5_interactive_usage() 