# HSData 二進制檔案轉換工具

這是一個用於讀取和轉換 HSData 二進制檔案為 CSV 格式的 Python 工具。

## 功能特點

- ✅ 讀取 HSData 二進制檔案格式
- ✅ 驗證檔案格式和數據完整性
- ✅ 提供多種 CSV 輸出格式
- ✅ 生成詳細的統計信息
- ✅ 支援命令行和程式化使用
- ✅ 完整的錯誤處理和日誌記錄

## 檔案結構

```
HSDATA/
├── hsdata_reader.py      # 核心讀取器類別
├── hsdata_converter.py   # 命令行工具
├── example_usage.py      # 使用範例
├── requirements.txt      # 依賴套件
├── README.md            # 說明文檔
└── HSData-2025-07-15-12-15-19.dat  # 範例數據檔案
```

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 命令行工具

#### 基本使用
```bash
# 轉換為合併格式 CSV（預設）
python hsdata_converter.py HSData-2025-07-15-12-15-19.dat

# 指定輸出檔案
python hsdata_converter.py HSData-2025-07-15-12-15-19.dat -o output.csv

# 顯示檔案信息和統計
python hsdata_converter.py HSData-2025-07-15-12-15-19.dat --info --stats
```

#### 輸出格式選項
```bash
# 合併格式（所有數據在一行）
python hsdata_converter.py input.dat --format combined

# 分離格式（vm、vd、da 分別導出）
python hsdata_converter.py input.dat --format separate

# 詳細格式（包含統計信息）
python hsdata_converter.py input.dat --format detailed

# 所有格式
python hsdata_converter.py input.dat --format all
```

#### 其他選項
```bash
# 僅顯示信息，不導出 CSV
python hsdata_converter.py input.dat --info --stats --no-export

# 詳細輸出模式
python hsdata_converter.py input.dat --verbose
```

### 2. 程式化使用

```python
from hsdata_reader import HSDataReader

# 創建讀取器
reader = HSDataReader("HSData-2025-07-15-12-15-19.dat")

# 驗證檔案格式
if reader.validate_file_format():
    # 讀取數據
    data_records = reader.read_data_records()
    
    # 獲取統計信息
    stats = reader.get_statistics()
    
    # 導出 CSV
    reader.export_to_csv("output.csv", "combined")
```

### 3. 使用範例腳本

```bash
python example_usage.py
```

## 輸出格式說明

### 1. 合併格式 (combined)
單一 CSV 檔案，每行包含：
- `index`: 記錄索引
- `vm_0` 到 `vm_5`: 6個 VM 值
- `vd_0` 到 `vd_5`: 6個 VD 值  
- `da_0` 到 `da_5`: 6個 DA 值

### 2. 分離格式 (separate)
生成三個 CSV 檔案：
- `*_vm.csv`: 僅包含 VM 數據
- `*_vd.csv`: 僅包含 VD 數據
- `*_da.csv`: 僅包含 DA 數據

### 3. 詳細格式 (detailed)
包含統計信息的完整數據：
- 原始數據（vm_0 到 da_5）
- 統計信息（平均值、標準差、最小值、最大值）

## 數據結構

### 檔案頭部 (32 bytes)
```cpp
struct HSDataFileHeader {
    char magic[8];           // "HSDATA\0\0"
    uint32_t version;        // 版本號 = 1
    uint32_t record_count;   // 數據記錄數量
    uint64_t timestamp;      // 檔案創建時間戳
}
```

### 數據記錄 (60 bytes)
```cpp
struct HSDataRecord {
    float vm[6];              // 6個 float，24 bytes
    float vd[6];              // 6個 float，24 bytes  
    unsigned short da[6];     // 6個 unsigned short，12 bytes
}
```

## 錯誤處理

工具包含完整的錯誤處理機制：

- 檔案不存在檢查
- 檔案格式驗證
- 數據完整性檢查
- 詳細的錯誤訊息和日誌記錄

## 效能優化

- 使用 numpy 進行向量化運算
- 分批讀取大檔案
- 記憶體效率優化
- 進度顯示功能

## 日誌記錄

工具會生成詳細的日誌檔案 `hsdata_converter.log`，包含：
- 操作時間戳
- 錯誤和警告訊息
- 處理進度
- 統計信息

## 系統需求

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0

## 授權

本工具僅供學習和研究使用。

## 支援

如有問題或建議，請檢查：
1. 檔案格式是否正確
2. 依賴套件是否已安裝
3. 日誌檔案中的錯誤訊息

## 更新日誌

### v1.0
- 初始版本
- 支援基本的二進制檔案讀取
- 提供多種 CSV 輸出格式
- 完整的錯誤處理和日誌記錄 