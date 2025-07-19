# HSData 二進制存儲方案（修正版）

## 概述

將 `HS_RDataThread` 函數中的數據存儲格式從文本格式改為二進制格式，以大幅減少檔案大小。本方案已修正原始計畫中的錯誤，確保實施的準確性。

## 目標

- 將檔案大小減少 75-80%
- 保持數據完整性
- 實現標準化的檔案格式
- 便於後續擴展
- 修正現有代碼中的邏輯錯誤

## 數據結構定義

### HSDataRecord 結構體
```cpp
struct HSDataRecord {
    float vm[6];              // 6個 float，24 bytes
    float vd[6];              // 6個 float，24 bytes  
    unsigned short da[6];     // 6個 unsigned short，12 bytes
    // 總計：60 bytes per record
};
```

### HSDataFileHeader 結構體
```cpp
struct HSDataFileHeader {
    char magic[8];           // "HSDATA\0\0" 用於識別檔案格式
    uint32_t version;        // 版本號 = 1
    uint32_t record_count;   // 數據記錄數量
    uint64_t timestamp;      // 檔案創建時間戳
};
```

## 檔案格式設計

### 二進制檔案結構
```
[檔案頭部 32 bytes] [數據記錄1 60 bytes] [數據記錄2 60 bytes] ... [數據記錄N 60 bytes]
```

### 檔案大小對比

| 格式 | 每筆數據大小 | 10000筆數據總大小 | 節省空間 |
|------|-------------|------------------|----------|
| 文本格式 | 200-250 字符 | 2-2.5 MB | - |
| 二進制格式 | 60 bytes | 600 KB | 75-80% |

## 發現的問題與修正

### 問題1：檔案創建邏輯重複
**問題描述**：`HS_RDataThread` 函數中檔案創建邏輯出現兩次（第7341行和第7575行）
**修正方案**：統一檔案創建邏輯，只在 while 迴圈外創建一次

### 問題2：存儲格式混亂
**問題描述**：同時使用文本格式（fprintf）和二進制格式（myfile.write）
**修正方案**：移除文本格式，統一使用二進制格式

### 問題3：檔案指針衝突
**問題描述**：使用 FILE* 和 std::fstream 兩個不同的檔案指針
**修正方案**：統一使用 FILE* 進行二進制操作

### 問題4：代碼結構不清
**問題描述**：大量註解代碼和重複邏輯
**修正方案**：清理註解代碼，簡化邏輯結構

## 修正後的實現步驟

### 步驟1：添加必要的頭文件
在 `PT3DView.cpp` 頂部添加：
```cpp
#include <cstdint>  // 為了使用 uint32_t 和 uint64_t
```

### 步驟2：添加結構體定義
在 `PT3DView.h` 中添加：
```cpp
// HSData 二進制存儲結構體
struct HSDataRecord {
    float vm[6];              // 6個 float，24 bytes
    float vd[6];              // 6個 float，24 bytes  
    unsigned short da[6];     // 6個 unsigned short，12 bytes
};

struct HSDataFileHeader {
    char magic[8];           // "HSDATA\0\0"
    uint32_t version;        // 版本號 = 1
    uint32_t record_count;   // 數據記錄數量
    uint64_t timestamp;      // 檔案創建時間戳
};
```

### 步驟3：清理現有代碼
**移除以下代碼：**
```cpp
// 移除這些行
std::fstream myfile;
myfile = std::fstream(str, std::ios::out | std::ios::binary);
myfile.write((char*)acReadBuf, batch_size * sizeof(UCHAR));
if (flag_fileopen) myfile.close();

// 移除重複的檔案創建邏輯（第7575行附近）
if (flag_fileopen == false) {
    // ... 重複的檔案創建代碼
}

// 移除文本格式的 fprintf
fprintf(pfile1, "%.6lf, %.6lf, ...", ...);
```

### 步驟4：修正檔案創建邏輯
**修改檔案創建部分（第7341行附近）：**
```cpp
if (m_flag_mot_savedata)
{
    time(&t_now);
    localtime_s(&timeinfo, &t_now);
    strftime(tmp, sizeof(tmp), "HSData-%Y-%m-%d-%H-%M-%S.dat", &timeinfo);
    str = tmp;
    str = sPath + str;

    // 改為二進制模式打開檔案
    if ((_wfopen_s(&pfile1, str, L"wb")) != 0)
    {
        AfxMessageBox(_T("file can not be opened"));
        return 0;
    }

    // 寫入檔案頭部
    HSDataFileHeader header;
    strcpy_s(header.magic, "HSDATA");
    header.version = 1;
    header.record_count = 0;  // 稍後更新
    header.timestamp = (uint64_t)t_now;
    
    if (fwrite(&header, sizeof(header), 1, pfile1) != 1) {
        AfxMessageBox(_T("Header write error"));
        fclose(pfile1);
        return 0;
    }
    
    flag_fileopen = true;
}
```

### 步驟5：修正數據存儲邏輯
**修改數據寫入部分（第7600行附近）：**
```cpp
if (m_flag_mot_savedata)
{
    for (i = 0; i < RT_packet_num; i++)
    {
        HSDataRecord record;
        for(int j = 0; j < 6; j++) {
            record.vm[j] = HSdata_Vm[j][i];
            record.vd[j] = HSdata_Vd[j][i];
            record.da[j] = MTCtrl_DA[j][i];
        }
        
        if (fwrite(&record, sizeof(HSDataRecord), 1, pfile1) != 1) {
            AfxMessageBox(_T("Data write error"));
            return 0;
        }
    }
    m_trackdata_save_counter++;
}
```

### 步驟6：修正檔案關閉邏輯
**修改檔案關閉部分（第7680行附近）：**
```cpp
if (flag_fileopen) {
    // 計算總記錄數
    uint32_t total_records = m_trackdata_save_counter * RT_packet_num;
    
    // 回到檔案開頭更新記錄數量
    fseek(pfile1, 0, SEEK_SET);
    HSDataFileHeader header;
    strcpy_s(header.magic, "HSDATA");
    header.version = 1;
    header.record_count = total_records;
    header.timestamp = (uint64_t)t_now;
    
    if (fwrite(&header, sizeof(header), 1, pfile1) != 1) {
        AfxMessageBox(_T("Header update error"));
    }
    
    fclose(pfile1);
    flag_fileopen = false;
}
```

## 實施順序

1. **第一步**：添加頭文件 `#include <cstdint>`
2. **第二步**：在 `PT3DView.h` 中添加結構體定義
3. **第三步**：清理現有代碼（移除重複和混亂部分）
4. **第四步**：修正檔案創建邏輯（統一創建，添加檔案頭部）
5. **第五步**：修正數據存儲邏輯（替換為二進制格式）
6. **第六步**：修正檔案關閉邏輯（更新記錄數量）

## 錯誤處理增強

### 檔案寫入檢查
```cpp
if (fwrite(&record, sizeof(HSDataRecord), 1, pfile1) != 1) {
    AfxMessageBox(_T("Data write error"));
    return 0;
}
```

### 檔案頭部更新檢查
```cpp
if (fwrite(&header, sizeof(header), 1, pfile1) != 1) {
    AfxMessageBox(_T("Header update error"));
}
```

## 優點

- **檔案大小減少 75-80%**
- **保持數據完整性**
- **檔案格式標準化**（有頭部信息）
- **便於後續擴展**（可以添加讀取功能）
- **修正了現有代碼的邏輯錯誤**
- **統一了存儲格式**
- **增強了錯誤處理**

## 注意事項

- 需要包含 `<cstdint>` 頭文件以使用 `uint32_t` 和 `uint64_t`
- 檔案打開模式從 `"w"` 改為 `"wb"`（二進制模式）
- 確保數據對齊和字節序一致性
- 移除了所有重複和混亂的代碼
- 統一使用 FILE* 進行檔案操作

---

# Python 解讀檔案方案

## 概述

本節說明如何使用 Python 解讀由 C++ 程式產生的 HSData 二進制檔案，提供完整的架構設計和程式碼框架。

## 檔案結構分析

### 檔案頭部結構 (32 bytes)
```python
# 檔案頭部格式
struct HSDataFileHeader {
    char magic[8];           # "HSDATA\0\0"
    uint32_t version;        # 版本號 = 1
    uint32_t record_count;   # 數據記錄數量
    uint64_t timestamp;      # 檔案創建時間戳
}
```

### 數據記錄結構 (60 bytes)
```python
# 每筆數據記錄格式
struct HSDataRecord {
    float vm[6];              # 6個 float，24 bytes
    float vd[6];              # 6個 float，24 bytes  
    unsigned short da[6];     # 6個 unsigned short，12 bytes
}
```

### 完整檔案結構
```
[檔案頭部 32 bytes] [數據記錄1 60 bytes] [數據記錄2 60 bytes] ... [數據記錄N 60 bytes]
```

## 基礎架構設計

### 核心類別框架
```python
import struct
import numpy as np
from datetime import datetime
from typing import Tuple, List

class HSDataReader:
    """HSData 二進制檔案讀取器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header = None
        self.data_records = []
    
    def read_header(self) -> dict:
        """讀取檔案頭部"""
        # 基礎框架：讀取 32 bytes 的檔案頭部
        pass
    
    def read_data_records(self) -> List[dict]:
        """讀取所有數據記錄"""
        # 基礎框架：讀取所有 60 bytes 的數據記錄
        pass
    
    def validate_file_format(self) -> bool:
        """驗證檔案格式"""
        # 基礎框架：檢查 magic number 和版本號
        pass
    
    def get_file_info(self) -> dict:
        """獲取檔案基本信息"""
        # 基礎框架：返回檔案統計信息
        pass
```

## 核心解讀邏輯框架

### 檔案頭部讀取
```python
def read_header(self) -> dict:
    """讀取檔案頭部"""
    with open(self.file_path, 'rb') as f:
        # 讀取 32 bytes 的檔案頭部
        header_data = f.read(32)
        
        # 使用 struct 解包
        # 格式：'8s I I Q' (8字串 + uint32 + uint32 + uint64)
        magic, version, record_count, timestamp = struct.unpack('<8s I I Q', header_data)
        
        return {
            'magic': magic.decode('ascii').rstrip('\x00'),
            'version': version,
            'record_count': record_count,
            'timestamp': timestamp,
            'creation_time': datetime.fromtimestamp(timestamp)
        }
```

### 數據記錄讀取
```python
def read_data_records(self) -> List[dict]:
    """讀取所有數據記錄"""
    records = []
    
    with open(self.file_path, 'rb') as f:
        # 跳過檔案頭部
        f.seek(32)
        
        # 讀取所有數據記錄
        while True:
            # 讀取 60 bytes 的數據記錄
            record_data = f.read(60)
            if len(record_data) < 60:
                break
                
            # 解包數據記錄
            # 格式：'6f 6f 6H' (6個float + 6個float + 6個unsigned short)
            vm = struct.unpack('<6f', record_data[0:24])
            vd = struct.unpack('<6f', record_data[24:48])
            da = struct.unpack('<6H', record_data[48:60])
            
            records.append({
                'vm': np.array(vm),
                'vd': np.array(vd),
                'da': np.array(da)
            })
    
    return records
```

## 使用範例框架

### 基本使用範例
```python
def main():
    """使用範例"""
    # 創建讀取器
    reader = HSDataReader("HSData-2024-01-15-14-30-25.dat")
    
    # 驗證檔案格式
    if not reader.validate_file_format():
        print("無效的檔案格式")
        return
    
    # 讀取檔案信息
    file_info = reader.get_file_info()
    print(f"檔案創建時間: {file_info['creation_time']}")
    print(f"數據記錄數量: {file_info['record_count']}")
    
    # 讀取所有數據
    data_records = reader.read_data_records()
    print(f"實際讀取記錄數: {len(data_records)}")
    
    # 數據分析範例
    if data_records:
        # 轉換為 numpy 陣列便於分析
        vm_data = np.array([record['vm'] for record in data_records])
        vd_data = np.array([record['vd'] for record in data_records])
        da_data = np.array([record['da'] for record in data_records])
        
        # 基礎統計
        print(f"vm 數據形狀: {vm_data.shape}")
        print(f"vd 數據形狀: {vd_data.shape}")
        print(f"da 數據形狀: {da_data.shape}")
```

## 關鍵技術要點

### 字節序處理
```python
# 使用 '<' 表示 little-endian（與 C++ 一致）
struct.unpack('<8s I I Q', header_data)
```

### 數據類型對應
```python
# C++ 到 Python 的類型對應
# uint32_t -> I (unsigned int)
# uint64_t -> Q (unsigned long long)
# float -> f (float)
# unsigned short -> H (unsigned short)
```

### 檔案偏移計算
```python
# 檔案頭部：0-31 bytes (32 bytes)
# 數據記錄：從第 32 byte 開始，每筆 60 bytes
# 總檔案大小 = 32 + (記錄數 × 60)
```

## 擴展功能框架

### 分析器類別
```python
class HSDataAnalyzer(HSDataReader):
    """HSData 分析器"""
    
    def plot_data(self, channel: int = 0):
        """繪製數據圖表"""
        pass
    
    def export_to_csv(self, output_path: str):
        """導出為 CSV 格式"""
        pass
    
    def get_statistics(self) -> dict:
        """獲取統計信息"""
        pass
    
    def filter_data(self, condition: callable) -> List[dict]:
        """數據過濾"""
        pass
```

### 數據可視化框架
```python
class HSDataVisualizer:
    """HSData 可視化工具"""
    
    def plot_time_series(self, data: np.ndarray, title: str = ""):
        """繪製時間序列圖"""
        pass
    
    def plot_3d_trajectory(self, x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray):
        """繪製 3D 軌跡圖"""
        pass
    
    def create_dashboard(self, data_records: List[dict]):
        """創建數據儀表板"""
        pass
```

## 錯誤處理與驗證

### 檔案格式驗證
```python
def validate_file_format(self) -> bool:
    """驗證檔案格式"""
    try:
        header = self.read_header()
        return (header['magic'] == 'HSDATA' and 
                header['version'] == 1)
    except:
        return False
```

### 數據完整性檢查
```python
def check_data_integrity(self) -> bool:
    """檢查數據完整性"""
    # 檢查檔案大小是否符合預期
    # 檢查記錄數量是否正確
    # 檢查數據範圍是否合理
    pass
```

## 效能優化建議

### 記憶體管理
- 使用生成器逐筆讀取大檔案
- 分批處理大量數據
- 使用 numpy 進行向量化運算

### 檔案讀取優化
- 使用 `mmap` 進行記憶體映射
- 並行讀取多個檔案
- 快取常用數據

## 應用場景

### 1. 數據分析
- 統計分析
- 趨勢分析
- 異常檢測

### 2. 數據可視化
- 時間序列圖表
- 3D 軌跡圖
- 實時監控儀表板

### 3. 數據轉換
- 導出為 CSV/Excel
- 轉換為其他格式
- 數據壓縮

### 4. 機器學習
- 特徵工程
- 模型訓練
- 預測分析

這個 Python 解讀方案提供了完整的架構框架，可以根據具體需求進行擴展和實現。 