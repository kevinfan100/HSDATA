#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 二進制檔案讀取器 - 精簡版
支援自動路徑處理和資料夾結構
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

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