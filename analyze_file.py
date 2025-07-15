#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檔案分析工具
用於分析 HSData 檔案的標頭訊息
"""

import struct
import os
import datetime

def analyze_file(file_path):
    """分析檔案標頭結構"""
    print(f"分析檔案: {file_path}")
    print("="*50)
    
    # 獲取檔案大小
    file_size = os.path.getsize(file_path)
    print(f"檔案大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    with open(file_path, 'rb') as f:
        # 讀取標頭 (8 + 4 + 4 + 8 = 24 bytes)
        header_data = f.read(24)
        
        if len(header_data) < 24:
            print("錯誤: 檔案太小，無法讀取完整標頭")
            return
        
        # 解析標頭
        magic = header_data[0:8]
        version = struct.unpack('<I', header_data[8:12])[0]  # 32-bit unsigned int, little-endian
        record_count = struct.unpack('<I', header_data[12:16])[0]  # 32-bit unsigned int, little-endian
        timestamp = struct.unpack('<Q', header_data[16:24])[0]  # 64-bit unsigned int, little-endian
        
        print("\n標頭訊息:")
        print(f"1. Magic: {magic} (hex: {magic.hex()})")
        print(f"   Magic (string): '{magic.decode('ascii', errors='ignore')}'")
        print(f"2. Version: {version}")
        print(f"3. Record Count: {record_count:,}")
        print(f"4. Timestamp: {timestamp}")
        
        # 轉換時間戳為可讀格式
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            print(f"   建立時間: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except (ValueError, OSError):
            print(f"   建立時間: 無法解析時間戳")
        
        # 驗證記錄大小
        remaining_size = file_size - 24
        if record_count > 0:
            record_size = remaining_size // record_count
            print(f"\n記錄大小計算:")
            print(f"  剩餘資料大小: {remaining_size:,} bytes")
            print(f"  記錄數量: {record_count:,}")
            print(f"  每筆記錄大小: {record_size} bytes")
            
            if remaining_size % record_count == 0:
                print(f"  ✓ 記錄大小計算正確")
            else:
                print(f"  ⚠ 記錄大小計算有餘數: {remaining_size % record_count} bytes")

if __name__ == "__main__":
    analyze_file("dc100_0715.dat") 