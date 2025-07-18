#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSData 二進制檔案讀取器
用於讀取和轉換 HSData 二進制檔案為 CSV 格式

版本：1.0
"""

import struct
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HSDataReader:
    """HSData 二進制檔案讀取器，只支援合併格式匯出"""
    MAGIC_NUMBER = b'HSDATA\x00\x00'
    VERSION = 1
    HEADER_SIZE = 24
    RECORD_SIZE = 60

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header = None
        self.data_records = []
        self._validate_file_exists()

    def _validate_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"檔案不存在: {self.file_path}")
        file_size = os.path.getsize(self.file_path)
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
                'file_size': os.path.getsize(self.file_path)
            }
            logger.info(f"檔案頭部讀取成功: {self.header}")
            return self.header

    def validate_file_format(self) -> bool:
        """驗證檔案格式"""
        if self.header is None:
            self.read_header()
        if not self.header['magic'].startswith('HSDATA'):
            logger.error(f"無效的 magic number: {self.header['magic']}")
            return False
        if self.header['version'] != self.VERSION:
            logger.error(f"不支援的版本號: {self.header['version']}")
            return False
        return True

    def read_data_records(self) -> List[Dict]:
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
                records.append({'index': idx, 'vm': np.array(vm), 'vd': np.array(vd), 'da': np.array(da)})
                idx += 1
                if idx % 10000 == 0:
                    logger.info(f"已讀取 {idx} 筆記錄")
        self.data_records = records
        logger.info(f"數據記錄讀取完成，共 {len(records)} 筆")
        return records

    def get_file_info(self) -> Dict:
        """取得檔案基本資訊"""
        if self.header is None:
            self.read_header()
        return {
            'file_path': self.file_path,
            'file_size_mb': round(self.header['file_size'] / (1024 * 1024), 2),
            'creation_time': self.header['creation_time'],
            'record_count': self.header['record_count'],
            'version': self.header['version'],
            'magic': self.header['magic']
        }

    def get_statistics(self) -> Dict:
        """取得數據統計資訊"""
        if not self.data_records:
            self.read_data_records()
        vm_data = np.array([r['vm'] for r in self.data_records])
        vd_data = np.array([r['vd'] for r in self.data_records])
        da_data = np.array([r['da'] for r in self.data_records])
        return {
            'total_records': len(self.data_records),
            'vm': {'mean': np.mean(vm_data, 0).tolist(), 'std': np.std(vm_data, 0).tolist(), 'min': np.min(vm_data, 0).tolist(), 'max': np.max(vm_data, 0).tolist()},
            'vd': {'mean': np.mean(vd_data, 0).tolist(), 'std': np.std(vd_data, 0).tolist(), 'min': np.min(vd_data, 0).tolist(), 'max': np.max(vd_data, 0).tolist()},
            'da': {'mean': np.mean(da_data, 0).tolist(), 'std': np.std(da_data, 0).tolist(), 'min': np.min(da_data, 0).tolist(), 'max': np.max(da_data, 0).tolist()}
        }

    def export_to_csv(self, output_path: str) -> str:
        """匯出合併格式 CSV"""
        if not self.data_records:
            self.read_data_records()
        data = []
        for r in self.data_records:
            row = [r['index']] + r['vm'].tolist() + r['vd'].tolist() + r['da'].tolist()
            data.append(row)
        columns = ['index'] + [f'vm_{i}' for i in range(6)] + [f'vd_{i}' for i in range(6)] + [f'da_{i}' for i in range(6)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 導出完成: {output_path}")
        return output_path