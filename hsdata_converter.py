#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 檔案轉換工具
命令行工具，用於將 HSData 二進制檔案轉換為 CSV 格式

使用方法:
    python hsdata_converter.py <input_file> [options]

作者：AI Assistant
版本：1.0
"""

import argparse
import sys
import os
from datetime import datetime
import json
from hsdata_reader import HSDataReader
import logging

def setup_logging(verbose: bool = False):
    """設置日誌"""
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
    """打印檔案信息"""
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
    """打印統計信息"""
    stats = reader.get_statistics()
    print("\n" + "="*60)
    print("數據統計")
    print("="*60)
    print(f"總記錄數: {stats['total_records']:,}")
    
    # VM 統計
    print("\nVM 數據統計:")
    for i in range(6):
        print(f"  VM_{i}: 平均={stats['vm']['mean'][i]:.6f}, "
              f"標準差={stats['vm']['std'][i]:.6f}, "
              f"範圍=[{stats['vm']['min'][i]:.6f}, {stats['vm']['max'][i]:.6f}]")
    
    # VD 統計
    print("\nVD 數據統計:")
    for i in range(6):
        print(f"  VD_{i}: 平均={stats['vd']['mean'][i]:.6f}, "
              f"標準差={stats['vd']['std'][i]:.6f}, "
              f"範圍=[{stats['vd']['min'][i]:.6f}, {stats['vd']['max'][i]:.6f}]")
    
    # DA 統計
    print("\nDA 數據統計:")
    for i in range(6):
        print(f"  DA_{i}: 平均={stats['da']['mean'][i]:.1f}, "
              f"標準差={stats['da']['std'][i]:.1f}, "
              f"範圍=[{stats['da']['min'][i]}, {stats['da']['max'][i]}]")
    print("="*60)

def export_csv(reader: HSDataReader, output_path: str, format_type: str):
    """導出 CSV 檔案"""
    print(f"\n開始導出 {format_type} 格式的 CSV...")
    
    try:
        if format_type == 'all':
            # 導出所有格式
            combined_path = reader.export_to_csv(output_path, 'combined')
            print(f"✓ 合併格式: {combined_path}")
            
            separate_base = reader.export_to_csv(output_path, 'separate')
            print(f"✓ 分離格式: {separate_base}_vm.csv, {separate_base}_vd.csv, {separate_base}_da.csv")
            
            detailed_path = reader.export_to_csv(output_path.replace('.csv', '_detailed.csv'), 'detailed')
            print(f"✓ 詳細格式: {detailed_path}")
        else:
            # 導出指定格式
            result_path = reader.export_to_csv(output_path, format_type)
            print(f"✓ 導出完成: {result_path}")
        
        print("CSV 導出成功！")
        
    except Exception as e:
        print(f"❌ CSV 導出失敗: {e}")
        return False
    
    return True

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='HSData 二進制檔案轉換工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat --format combined --output data.csv
  python hsdata_converter.py HSData-2025-07-15-12-15-19.dat --format all --info --stats
        """
    )
    
    parser.add_argument('input_file', help='輸入的 HSData 檔案路徑')
    parser.add_argument('-o', '--output', help='輸出 CSV 檔案路徑 (預設: 自動生成)')
    parser.add_argument('-f', '--format', 
                       choices=['combined', 'separate', 'detailed', 'all'],
                       default='combined',
                       help='輸出格式 (預設: combined)')
    parser.add_argument('--info', action='store_true', help='顯示檔案信息')
    parser.add_argument('--stats', action='store_true', help='顯示統計信息')
    parser.add_argument('--no-export', action='store_true', help='不導出 CSV，僅顯示信息')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    # 設置日誌
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # 檢查輸入檔案
    if not os.path.exists(args.input_file):
        print(f"❌ 錯誤: 檔案不存在 - {args.input_file}")
        sys.exit(1)
    
    try:
        # 創建讀取器
        print(f"正在讀取檔案: {args.input_file}")
        reader = HSDataReader(args.input_file)
        
        # 驗證檔案格式
        if not reader.validate_file_format():
            print("❌ 錯誤: 無效的 HSData 檔案格式")
            sys.exit(1)
        
        # 顯示檔案信息
        if args.info:
            print_file_info(reader)
        
        # 讀取數據記錄
        print("正在讀取數據記錄...")
        reader.read_data_records()
        
        # 顯示統計信息
        if args.stats:
            print_statistics(reader)
        
        # 導出 CSV
        if not args.no_export:
            # 生成輸出檔案路徑
            if args.output:
                output_path = args.output
            else:
                base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{base_name}_{args.format}_{timestamp}.csv"
            
            # 執行導出
            success = export_csv(reader, output_path, args.format)
            if not success:
                sys.exit(1)
        
        print("\n✅ 處理完成！")
        
    except Exception as e:
        logger.error(f"處理失敗: {e}")
        print(f"❌ 錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 