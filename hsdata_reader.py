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
import csv
import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HSDataReader:
    """HSData 二進制檔案讀取器"""
    
    # 檔案格式常量
    MAGIC_NUMBER = b'HSDATA\x00\x00'
    VERSION = 1
    HEADER_SIZE = 24  # 修正：實際頭部大小是24 bytes
    RECORD_SIZE = 60
    
    def __init__(self, file_path: str):
        """
        初始化讀取器
        
        Args:
            file_path (str): HSData 檔案路徑
        """
        self.file_path = file_path
        self.header = None
        self.data_records = []
        self._validate_file_exists()
    
    def _validate_file_exists(self):
        """驗證檔案是否存在"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"檔案不存在: {self.file_path}")
        
        file_size = os.path.getsize(self.file_path)
        if file_size < self.HEADER_SIZE:
            raise ValueError(f"檔案太小，無法包含有效的檔案頭部: {file_size} bytes")
        
        # 驗證檔案大小是否符合預期
        expected_size = self.HEADER_SIZE + (150000 * self.RECORD_SIZE)  # 根據實際數據
        if file_size != expected_size:
            print(f"警告: 檔案大小不匹配，預期 {expected_size:,} bytes，實際 {file_size:,} bytes")
    
    def read_header(self) -> Dict:
        """
        讀取檔案頭部
        
        Returns:
            Dict: 包含檔案頭部信息的字典
        """
        try:
            with open(self.file_path, 'rb') as f:
                # 讀取 24 bytes 的檔案頭部
                header_data = f.read(self.HEADER_SIZE)
                
                if len(header_data) < self.HEADER_SIZE:
                    raise ValueError("檔案頭部數據不完整")
            
                magic = header_data[0:8]
                version = struct.unpack('<I', header_data[8:12])[0]
                record_count = struct.unpack('<I', header_data[12:16])[0]
                timestamp = struct.unpack('<Q', header_data[16:24])[0]
                
                self.header = {
                    'magic': magic.decode('ascii').rstrip('\x00'),
                    'version': version,
                    'record_count': record_count,
                    'timestamp': timestamp,
                    'creation_time': datetime.fromtimestamp(timestamp),
                    'file_size': os.path.getsize(self.file_path)
                }
                
                logger.info(f"檔案頭部讀取成功: {self.header}")
                return self.header
                
        except Exception as e:
            logger.error(f"讀取檔案頭部失敗: {e}")
            raise
    
    def validate_file_format(self) -> bool:
        """
        驗證檔案格式
        
        Returns:
            bool: 檔案格式是否有效
        """
        try:
            if self.header is None:
                self.read_header()
            
            # 檢查 magic number（只檢查前6個字元為HSDATA）
            if not self.header['magic'].startswith('HSDATA'):
                logger.error(f"無效的 magic number: {self.header['magic']}")
                return False
            
            # 檢查版本號
            if self.header['version'] != self.VERSION:
                logger.error(f"不支援的版本號: {self.header['version']}")
                return False
            
            # 檢查檔案大小
            expected_size = self.HEADER_SIZE + (self.header['record_count'] * self.RECORD_SIZE)
            if self.header['file_size'] != expected_size:
                logger.warning(f"檔案大小不匹配: 預期 {expected_size:,}, 實際 {self.header['file_size']:,}")
            
            logger.info("檔案格式驗證通過")
            return True
            
        except Exception as e:
            logger.error(f"檔案格式驗證失敗: {e}")
            return False
    
    def read_data_records(self) -> List[Dict]:
        """
        讀取所有數據記錄
        
        Returns:
            List[Dict]: 包含所有數據記錄的列表
        """
        records = []
        
        try:
            with open(self.file_path, 'rb') as f:
                # 跳過檔案頭部（24 bytes）
                f.seek(self.HEADER_SIZE)
                
                record_index = 0
                # 讀取所有數據記錄
                while True:
                    # 讀取 60 bytes 的數據記錄
                    record_data = f.read(self.RECORD_SIZE)
                    if len(record_data) < self.RECORD_SIZE:
                        break
                    
                    # 解包數據記錄
                    # 格式：'6f 6f 6H' (6個float + 6個float + 6個unsigned short)
                    vm = struct.unpack('<6f', record_data[0:24])
                    vd = struct.unpack('<6f', record_data[24:48])
                    da = struct.unpack('<6H', record_data[48:60])
                    
                    records.append({
                        'index': record_index,
                        'vm': np.array(vm),
                        'vd': np.array(vd),
                        'da': np.array(da)
                    })
                    
                    record_index += 1
                    
                    # 進度顯示
                    if record_index % 10000 == 0:
                        logger.info(f"已讀取 {record_index} 筆記錄")
            
            self.data_records = records
            logger.info(f"數據記錄讀取完成，共 {len(records)} 筆")
            return records
            
        except Exception as e:
            logger.error(f"讀取數據記錄失敗: {e}")
            raise
    
    def get_file_info(self) -> Dict:
        """
        獲取檔案基本信息
        
        Returns:
            Dict: 檔案信息字典
        """
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
        """獲取數據統計信息"""
        if not self.data_records:
            self.read_data_records()
        
        # 準備數據
        vm_data = np.array([record['vm'] for record in self.data_records])
        vd_data = np.array([record['vd'] for record in self.data_records])
        da_data = np.array([record['da'] for record in self.data_records])
        
        # 計算統計信息
        stats = {
            'total_records': len(self.data_records),
            'vm': {
                'mean': np.mean(vm_data, axis=0).tolist(),
                'std': np.std(vm_data, axis=0).tolist(),
                'min': np.min(vm_data, axis=0).tolist(),
                'max': np.max(vm_data, axis=0).tolist()
            },
            'vd': {
                'mean': np.mean(vd_data, axis=0).tolist(),
                'std': np.std(vd_data, axis=0).tolist(),
                'min': np.min(vd_data, axis=0).tolist(),
                'max': np.max(vd_data, axis=0).tolist()
            },
            'da': {
                'mean': np.mean(da_data, axis=0).tolist(),
                'std': np.std(da_data, axis=0).tolist(),
                'min': np.min(da_data, axis=0).tolist(),
                'max': np.max(da_data, axis=0).tolist()
            }
        }
        
        return stats

    def export_to_csv(self, output_path: str, format_type: str = 'combined') -> str:
        """
        導出數據為 CSV 格式
        
        Args:
            output_path (str): 輸出檔案路徑
            format_type (str): 輸出格式 ('combined', 'separate', 'detailed')
        
        Returns:
            str: 輸出檔案路徑
        """
        if not self.data_records:
            self.read_data_records()
        
        if format_type == 'combined':
            return self._export_combined_csv(output_path)
        elif format_type == 'separate':
            return self._export_separate_csv(output_path)
        elif format_type == 'detailed':
            return self._export_detailed_csv(output_path)
        else:
            raise ValueError(f"不支援的格式類型: {format_type}")
    
    def _export_combined_csv(self, output_path: str) -> str:
        """導出合併格式的 CSV"""
        try:
            # 準備數據
            data = []
            for record in self.data_records:
                row = [record['index']]
                row.extend(record['vm'].tolist())
                row.extend(record['vd'].tolist())
                row.extend(record['da'].tolist())
                data.append(row)
            
            # 準備列名
            columns = ['index']
            columns.extend([f'vm_{i}' for i in range(6)])
            columns.extend([f'vd_{i}' for i in range(6)])
            columns.extend([f'da_{i}' for i in range(6)])
            
            # 創建 DataFrame 並導出
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"合併格式 CSV 導出完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"導出合併格式 CSV 失敗: {e}")
            raise
    
    def _export_separate_csv(self, output_path: str) -> str:
        """導出分離格式的 CSV（分別導出 vm、vd、da）"""
        try:
            base_path = output_path.replace('.csv', '')
            
            # 準備數據
            vm_data = np.array([record['vm'] for record in self.data_records])
            vd_data = np.array([record['vd'] for record in self.data_records])
            da_data = np.array([record['da'] for record in self.data_records])
            
            # 導出 vm 數據
            vm_df = pd.DataFrame(vm_data, columns=[f'vm_{i}' for i in range(6)])
            vm_df.insert(0, 'index', range(len(vm_data)))
            vm_path = f"{base_path}_vm.csv"
            vm_df.to_csv(vm_path, index=False, encoding='utf-8-sig')
            
            # 導出 vd 數據
            vd_df = pd.DataFrame(vd_data, columns=[f'vd_{i}' for i in range(6)])
            vd_df.insert(0, 'index', range(len(vd_data)))
            vd_path = f"{base_path}_vd.csv"
            vd_df.to_csv(vd_path, index=False, encoding='utf-8-sig')
            
            # 導出 da 數據
            da_df = pd.DataFrame(da_data, columns=[f'da_{i}' for i in range(6)])
            da_df.insert(0, 'index', range(len(da_data)))
            da_path = f"{base_path}_da.csv"
            da_df.to_csv(da_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"分離格式 CSV 導出完成:")
            logger.info(f"  VM: {vm_path}")
            logger.info(f"  VD: {vd_path}")
            logger.info(f"  DA: {da_path}")
            
            return base_path
            
        except Exception as e:
            logger.error(f"導出分離格式 CSV 失敗: {e}")
            raise