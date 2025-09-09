#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 二進制檔案讀取器 - 批次處理版本
支援整個資料夾的批次處理和自動路徑結構
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import os
import glob

class HSDataReader:
    """HSData 二進制檔案讀取器 - 精簡版"""
    
    # 檔案格式常數
    MAGIC_NUMBER = b'HSDATA\x00\x00'
    VERSION = 1
    HEADER_SIZE = 24
    RECORD_SIZE = 60

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.header: Optional[Dict] = None
        self.data_records: List[Dict] = []
        self._validate_file()

    def _validate_file(self):
        """驗證檔案存在性和基本格式"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"檔案不存在: {self.file_path}")
        
        file_size = self.file_path.stat().st_size
        if file_size < self.HEADER_SIZE:
            raise ValueError(f"檔案太小: {file_size} bytes")

    def read_header(self) -> Dict:
        """讀取檔案頭部資訊"""
        with open(self.file_path, 'rb') as f:
            header_data = f.read(self.HEADER_SIZE)
            
            if len(header_data) < self.HEADER_SIZE:
                raise ValueError("檔案頭部數據不完整")
            
            magic = header_data[0:8]
            version = struct.unpack('<I', header_data[8:12])[0]
            record_count = struct.unpack('<I', header_data[12:16])[0]
            timestamp = struct.unpack('<Q', header_data[16:24])[0]
            
            self.header = {
                'magic': magic.decode('ascii', errors='ignore').rstrip('\x00'),
                'version': version,
                'record_count': record_count,
                'timestamp': timestamp,
                'creation_time': datetime.fromtimestamp(timestamp),
                'file_size': self.file_path.stat().st_size
            }
            
            return self.header

    def validate_format(self) -> bool:
        """驗證檔案格式"""
        if self.header is None:
            self.read_header()
        
        if not self.header['magic'].startswith('HSDATA'):
            print(f"❌ 無效的 magic number: {self.header['magic']}")
            return False
        
        if self.header['version'] != self.VERSION:
            print(f"❌ 不支援的版本號: {self.header['version']}")
            return False
        
        return True

    def read_data(self) -> List[Dict]:
        """讀取所有數據記錄"""
        records = []
        with open(self.file_path, 'rb') as f:
            f.seek(self.HEADER_SIZE)
            idx = 0
            
            while True:
                record_data = f.read(self.RECORD_SIZE)
                if len(record_data) < self.RECORD_SIZE:
                    break
                
                vm = struct.unpack('<6f', record_data[0:24])
                vd = struct.unpack('<6f', record_data[24:48])
                da = struct.unpack('<6H', record_data[48:60])
                
                records.append({
                    'index': idx, 
                    'vm': np.array(vm), 
                    'vd': np.array(vd), 
                    'da': np.array(da)
                })
                idx += 1
                
                # 簡單的進度顯示
                if idx % 50000 == 0:
                    print(f"已讀取 {idx} 筆記錄...")
        
        self.data_records = records
        print(f"✓ 數據讀取完成，共 {len(records)} 筆記錄")
        return records
    
    def read_data_records(self) -> List[Dict]:
        """讀取所有數據記錄 (別名方法，與原始程式碼相容)"""
        return self.read_data()

    def get_info(self) -> Dict:
        """取得檔案基本資訊"""
        if self.header is None:
            self.read_header()
        
        return {
            'file_path': str(self.file_path),
            'file_size_mb': round(self.header['file_size'] / (1024 * 1024), 2),
            'creation_time': self.header['creation_time'],
            'record_count': self.header['record_count'],
            'version': self.header['version'],
            'magic': self.header['magic']
        }

    def to_csv(self, output_path: Optional[str] = None) -> str:
        """轉換為 CSV 檔案，自動處理路徑"""
        if not self.data_records:
            self.read_data()
        
        # 自動生成輸出路徑
        if output_path is None:
            output_path = self._get_auto_csv_path()
        
        # 確保輸出目錄存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for r in self.data_records:
            row = [r['index']] + r['vm'].tolist() + r['vd'].tolist() + r['da'].tolist()
            data.append(row)
        
        columns = ['index'] + [f'vm_{i}' for i in range(6)] + [f'vd_{i}' for i in range(6)] + [f'da_{i}' for i in range(6)]
        df = pd.DataFrame(data, columns=columns)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"CSV 導出完成: {output_path}")
        return str(output_path)

    def _get_auto_csv_path(self) -> Path:
        """自動生成 CSV 輸出路徑"""
        # 取得專案根目錄
        project_root = self._find_project_root()
        
        # 確保檔案路徑是絕對路徑
        file_path_abs = self.file_path.resolve()
        
        # 取得相對路徑（從 raw data 開始）
        try:
            raw_data_path = project_root / "01Data" / "01Raw_dat"
            relative_path = file_path_abs.relative_to(raw_data_path)
            
            # 替換副檔名並放到 processed 目錄
            csv_filename = relative_path.with_suffix('.csv')
            csv_path = project_root / "01Data" / "02Processed_csv" / csv_filename
            
            return csv_path
            
        except ValueError as e:
            print(f"無法取得相對路徑: {e}")
            # 如果無法取得相對路徑，使用預設位置
            return Path(f"{self.file_path.stem}.csv")

    def _find_project_root(self) -> Path:
        """尋找專案根目錄"""
        current = Path.cwd()
        
        # 向上尋找包含 01Data 目錄的資料夾
        while current != current.parent:
            if (current / "01Data").exists():
                return current
            current = current.parent
        
        # 如果找不到，使用當前工作目錄
        return Path.cwd()

    def to_dataframe(self) -> pd.DataFrame:
        """取得 pandas DataFrame 格式的數據"""
        if not self.data_records:
            self.read_data()
        
        data = []
        for r in self.data_records:
            row = [r['index']] + r['vm'].tolist() + r['vd'].tolist() + r['da'].tolist()
            data.append(row)
        
        columns = ['index'] + [f'vm_{i}' for i in range(6)] + [f'vd_{i}' for i in range(6)] + [f'da_{i}' for i in range(6)]
        return pd.DataFrame(data, columns=columns)


class BatchProcessor:
    """批次處理器 - 處理整個資料夾"""
    
    def __init__(self, input_folder: str | Path, output_folder: Optional[str | Path] = None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else None
        self.processed_files = []
        self.failed_files = []
        
        if not self.input_folder.exists():
            raise FileNotFoundError(f"輸入資料夾不存在: {self.input_folder}")

    def find_dat_files(self, recursive: bool = True) -> List[Path]:
        """尋找所有 .dat 檔案"""
        if recursive:
            pattern = "**/*.dat"
        else:
            pattern = "*.dat"
        
        dat_files = list(self.input_folder.glob(pattern))
        print(f"找到 {len(dat_files)} 個 .dat 檔案")
        return dat_files

    def process_folder(self, recursive: bool = True, skip_existing: bool = True) -> Dict:
        """批次處理整個資料夾"""
        dat_files = self.find_dat_files(recursive)
        
        if not dat_files:
            print("❌ 未找到任何 .dat 檔案")
            return {"processed": 0, "failed": 0, "skipped": 0}
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"\n開始批次處理 {len(dat_files)} 個檔案...")
        print("=" * 60)
        
        for i, file_path in enumerate(dat_files, 1):
            print(f"\n[{i}/{len(dat_files)}] 處理檔案: {file_path.name}")
            
            try:
                # 決定輸出路徑
                output_path = self._get_output_path(file_path)
                
                # 處理檔案
                reader = HSDataReader(file_path)
                
                # 驗證格式
                header = reader.read_header()
                if not reader.validate_format():
                    print(f"❌ 格式驗證失敗: {file_path}")
                    self.failed_files.append({"file": file_path, "error": "格式驗證失敗"})
                    failed_count += 1
                    continue
                
                # 讀取數據並轉換
                data_records = reader.read_data()
                csv_path = reader.to_csv(str(output_path))
                
                self.processed_files.append({
                    "input": file_path,
                    "output": csv_path,
                    "records": len(data_records),
                    "size_mb": reader.get_info()["file_size_mb"]
                })
                
                processed_count += 1
                print(f"✅ 處理完成: {file_path.name}")
                
            except Exception as e:
                print(f"❌ 處理失敗: {file_path.name} - {str(e)}")
                self.failed_files.append({"file": file_path, "error": str(e)})
                failed_count += 1
        
        # 顯示總結
        print("\n" + "=" * 60)
        print("批次處理完成！")
        print(f"✅ 成功處理: {processed_count} 個檔案")
        print(f"❌ 處理失敗: {failed_count} 個檔案")
        
        if self.failed_files:
            print("\n失敗檔案清單:")
            for failed in self.failed_files:
                print(f"  - {failed['file'].name}: {failed['error']}")
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files
        }

    def _get_output_path(self, input_file: Path) -> Path:
        """取得輸出檔案路徑"""
        if self.output_folder:
            # 使用指定的輸出資料夾
            # 保持相對路徑結構
            try:
                relative_path = input_file.relative_to(self.input_folder)
                output_path = self.output_folder / relative_path.with_suffix('.csv')
            except ValueError:
                # 如果無法取得相對路徑，直接使用檔名
                output_path = self.output_folder / input_file.with_suffix('.csv').name
        else:
            # 使用自動路徑（原有邏輯）
            reader = HSDataReader(input_file)
            output_path = Path(reader._get_auto_csv_path())
        
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

def main():
    """主程式 - 批次處理版本"""
    
    # 設定路徑
    input_folder = "C:/Users/lu921/Desktop/git_repos/HSDATA/01Data/01Raw_dat"
    output_folder = "C:/Users/lu921/Desktop/git_repos/HSDATA/01Data/02Processed_csv"
    
    # 方式1: 使用指定的輸出資料夾
    try:
        print("🚀 開始批次處理...")
        processor = BatchProcessor(input_folder, output_folder)
        
        # 處理整個資料夾 (遞迴搜尋子資料夾)
        results = processor.process_folder(
            recursive=True,      # 遞迴搜尋子資料夾
            skip_existing=True   # 跳過已存在的檔案
        )
        
    except Exception as e:
        print(f"❌ 批次處理失敗: {e}")
        return
    
    # 方式2: 使用自動路徑 (註解掉，可依需求切換)
    """
    try:
        print("🚀 開始批次處理 (自動路徑)...")
        processor = BatchProcessor(input_folder)  # 不指定輸出資料夾
        
        results = processor.process_folder(recursive=True, skip_existing=True)
        report_path = processor.generate_summary_report()
        
        print(f"\n🎉 批次處理全部完成！")
        print(f"📊 詳細報告: {report_path}")
        
    except Exception as e:
        print(f"❌ 批次處理失敗: {e}")
        return
    """


def single_file_example():
    """單檔處理範例 (保留原功能)"""
    file_path = "C:/Users/lu921/Desktop/git_repos/HSDATA/01Data/01Raw_dat/jump/500_jump_newB.dat"
    
    try:
        reader = HSDataReader(file_path)
        header = reader.read_header()
        
        if reader.validate_format():
            print("檔案格式驗證通過")
            info = reader.get_info()
            print(f"檔案大小: {info['file_size_mb']} MB")
            
            data_records = reader.read_data()
            df = reader.to_dataframe()
            csv_path = reader.to_csv()
            print(f"CSV 檔案已儲存至: {csv_path}")
        
    except Exception as e:
        print(f"錯誤: {e}")


if __name__ == "__main__":
    main()  # 執行批次處理
    # single_file_example()  # 或執行單檔處理範例