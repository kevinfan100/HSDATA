#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料夾批次處理器
用於處理整個資料夾中的 HSData 檔案
"""

from pathlib import Path
from typing import List, Dict
import sys

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("hsdata_reader", Path(__file__).parent.parent / "01Core" / "hsdata_reader.py")
hsdata_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsdata_reader)
HSDataReader = hsdata_reader.HSDataReader

class FolderProcessor:
    """資料夾批次處理器"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
    
    def process_folder(self, folder_path: str | Path) -> Dict:
        """處理整個資料夾"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ 資料夾不存在: {folder_path}")
            return {}
        
        # 找到所有 .dat 檔案
        dat_files = list(folder_path.glob("*.dat"))
        
        if not dat_files:
            print(f"❌ 資料夾中沒有 .dat 檔案: {folder_path}")
            return {}
        
        print(f"找到 {len(dat_files)} 個 .dat 檔案")
        
        results = {}
        for i, dat_file in enumerate(dat_files, 1):
            print(f"\n[{i}/{len(dat_files)}] 處理: {dat_file.name}")
            
            try:
                reader = HSDataReader(dat_file)
                
                if reader.validate_format():
                    # 自動轉換為 CSV（會自動放到對應的 processed 資料夾）
                    csv_path = reader.to_csv()
                    results[str(dat_file)] = {
                        'status': 'success',
                        'csv_path': csv_path,
                        'info': reader.get_info()
                    }
                    self.processed_count += 1
                    print(f"  ✓ 完成")
                else:
                    results[str(dat_file)] = {
                        'status': 'invalid_format',
                        'error': '檔案格式無效'
                    }
                    self.error_count += 1
                    print(f"  ❌ 檔案格式無效")
                    
            except Exception as e:
                results[str(dat_file)] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.error_count += 1
                print(f"  ❌ 錯誤: {e}")
        
        print(f"\n處理完成！成功: {self.processed_count}, 失敗: {self.error_count}")
        return results
    
    def process_all_raw_folders(self) -> Dict:
        """處理所有 raw data 資料夾"""
        project_root = self._find_project_root()
        raw_data_dir = project_root / "01Data" / "01Raw_dat"
        
        if not raw_data_dir.exists():
            print(f"❌ Raw data 目錄不存在: {raw_data_dir}")
            return {}
        
        all_results = {}
        
        # 處理每個子資料夾
        for subfolder in raw_data_dir.iterdir():
            if subfolder.is_dir():
                print(f"\n=== 處理資料夾: {subfolder.name} ===")
                results = self.process_folder(subfolder)
                all_results[subfolder.name] = results
        
        return all_results
    
    def process_specific_folders(self, folder_names: List[str]) -> Dict:
        """處理指定的資料夾"""
        project_root = self._find_project_root()
        raw_data_dir = project_root / "01Data" / "01Raw_dat"
        
        if not raw_data_dir.exists():
            print(f"❌ Raw data 目錄不存在: {raw_data_dir}")
            return {}
        
        all_results = {}
        
        for folder_name in folder_names:
            folder_path = raw_data_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                print(f"\n=== 處理資料夾: {folder_name} ===")
                results = self.process_folder(folder_path)
                all_results[folder_name] = results
            else:
                print(f"❌ 資料夾不存在: {folder_name}")
        
        return all_results
    
    def _find_project_root(self) -> Path:
        """尋找專案根目錄"""
        current = Path.cwd()
        
        while current != current.parent:
            if (current / "01Data").exists():
                return current
            current = current.parent
        
        return Path.cwd()
    
    def get_folder_info(self, folder_path: str | Path) -> Dict:
        """取得資料夾資訊"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {'error': '資料夾不存在'}
        
        dat_files = list(folder_path.glob("*.dat"))
        
        return {
            'folder_path': str(folder_path),
            'dat_count': len(dat_files),
            'dat_files': [f.name for f in dat_files],
            'total_size_mb': sum(f.stat().st_size for f in dat_files) / (1024 * 1024)
        } 