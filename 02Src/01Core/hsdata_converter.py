#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 檔案轉換工具 - 精簡版
命令行工具，用於將 HSData 二進制檔案轉換為 CSV 格式

使用方法:
    python hsdata_converter.py <input_file> [options]
"""

import argparse
import sys
import os
from datetime import datetime
import importlib.util
from pathlib import Path

# 動態導入 HSDataReader
spec = importlib.util.spec_from_file_location("hsdata_reader", Path(__file__).parent / "hsdata_reader.py")
hsdata_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsdata_reader)
HSDataReader = hsdata_reader.HSDataReader

def print_file_info(reader: HSDataReader):
    """顯示檔案資訊"""
    info = reader.get_info()
    print("\n" + "="*60)
    print("檔案信息")
    print("="*60)
    print(f"檔案路徑: {info['file_path']}")
    print(f"檔案大小: {info['file_size_mb']} MB")
    print(f"創建時間: {info['creation_time']}")
    print(f"記錄數量: {info['record_count']:,}")
    print(f"版本號: {info['version']}")
    print(f"Magic Number: {info['magic']}")
    print("="*60)

def export_csv(reader: HSDataReader, output_path: str = None):
    """導出 CSV 檔案"""
    print(f"\n開始導出 CSV...")
    try:
        result_path = reader.to_csv(output_path)
        print(f"導出完成: {result_path}\n")
        return True
    except Exception as e:
        print(f"❌ CSV 導出失敗: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='HSData 二進制檔案轉換工具 - 精簡版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python hsdata_converter.py 01Data/01Raw_dat/0716_hsdata_ndc/ndc100.dat
  python hsdata_converter.py 01Data/01Raw_dat/0716_hsdata_ndc/ndc100.dat --output my_data.csv
  python hsdata_converter.py 01Data/01Raw_dat/0716_hsdata_ndc/ndc100.dat --info --no-export
        """
    )
    parser.add_argument('input_file', help='輸入的 HSData 檔案路徑')
    parser.add_argument('-o', '--output', help='輸出 CSV 檔案路徑 (預設: 自動生成到對應的 processed 資料夾)')
    parser.add_argument('--info', action='store_true', help='顯示檔案信息')
    parser.add_argument('--no-export', action='store_true', help='不導出 CSV，僅顯示信息')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ 錯誤: 檔案不存在 - {args.input_file}")
        sys.exit(1)
    
    try:
        print(f"正在讀取檔案: {args.input_file}")
        reader = HSDataReader(args.input_file)
        
        if not reader.validate_format():
            print("❌ 錯誤: 無效的 HSData 檔案格式")
            sys.exit(1)
        
        if args.info:
            print_file_info(reader)
        
        reader.read_data()
        
        if not args.no_export:
            success = export_csv(reader, args.output)
            if not success:
                sys.exit(1)
        
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 