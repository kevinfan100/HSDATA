#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 檔案轉換工具
命令行工具，用於將 HSData 二進制檔案轉換為 CSV 格式

使用方法:
    python hsdata_converter.py <input_file> [options]
"""

import argparse
import sys
import os
from datetime import datetime
from hsdata_reader import HSDataReader
import logging

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hsdata_converter.log', encoding='utf-8')
        ]
    )

def print_file_info(reader: HSDataReader):
    info = reader.get_file_info()
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

def print_statistics(reader: HSDataReader):
    stats = reader.get_statistics()
    print("\n" + "="*60)
    print("數據統計")
    print("="*60)
    print(f"總記錄數: {stats['total_records']:,}")
    for label in ['vm', 'vd', 'da']:
        print(f"\n{label.upper()} 數據統計:")
        for i in range(6):
            mean = stats[label]['mean'][i]
            std = stats[label]['std'][i]
            minv = stats[label]['min'][i]
            maxv = stats[label]['max'][i]
            if label == 'da':
                print(f"  {label.upper()}_{i}: 平均={mean:.1f}, 標準差={std:.1f}, 範圍=[{minv}, {maxv}]")
            else:
                print(f"  {label.upper()}_{i}: 平均={mean:.6f}, 標準差={std:.6f}, 範圍=[{minv:.6f}, {maxv:.6f}]")
    print("="*60)

def export_csv(reader: HSDataReader, output_path: str):
    print(f"\n開始導出 CSV...")
    try:
        result_path = reader.export_to_csv(output_path)
        print(f"✓ 導出完成: {result_path}")
        print("CSV 導出成功！")
    except Exception as e:
        print(f"❌ CSV 導出失敗: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description='HSData 二進制檔案轉換工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat --output data.csv
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat --info --stats
        """
    )
    parser.add_argument('input_file', help='輸入的 HSData 檔案路徑')
    parser.add_argument('-o', '--output', help='輸出 CSV 檔案路徑 (預設: 自動生成)')
    parser.add_argument('--info', action='store_true', help='顯示檔案信息')
    parser.add_argument('--stats', action='store_true', help='顯示統計信息')
    parser.add_argument('--no-export', action='store_true', help='不導出 CSV，僅顯示信息')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細輸出')
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.input_file):
        print(f"❌ 錯誤: 檔案不存在 - {args.input_file}")
        sys.exit(1)
    try:
        print(f"正在讀取檔案: {args.input_file}")
        reader = HSDataReader(args.input_file)
        if not reader.validate_file_format():
            print("❌ 錯誤: 無效的 HSData 檔案格式")
            sys.exit(1)
        if args.info:
            print_file_info(reader)
        print("正在讀取數據記錄...")
        reader.read_data_records()
        if args.stats:
            print_statistics(reader)
        if not args.no_export:
            if args.output:
                output_path = args.output
            else:
                base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{base_name}.csv"
            success = export_csv(reader, output_path)
            if not success:
                sys.exit(1)
        print("\n✅ 處理完成！")
    except Exception as e:
        logger.error(f"處理失敗: {e}")
        print(f"❌ 錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 