# 多模態情緒分析系統

## 系統簡介

這個系統可以分析影片中的**視覺（臉部表情）**、**音訊（聲音特徵）**和**文字（對話內容）**三種模態的情緒，並比較它們之間的一致性。

### ✨ 核心特色

-  **只分析四種基本情緒**：快樂、生氣、悲傷、中性（更容易理解）
-  **支援自訂文字檔案**：不需要依賴語音轉錄，可以直接提供對話內容
-  **詳細情緒解釋**：告訴你為什麼判斷是那種情緒（關鍵詞、分數、原因）
-  **完整視覺化**：時間序列折線圖、圓餅圖、相關性熱圖
-  **相關性分析**：計算視覺和音訊情緒的同步程度
-  **中英文支援**：圖表和輸出都有中文標籤

---

##  快速開始

### 1. 安裝環境

```bash
# 創建 conda 環境
conda create --name hw python=3.9 -y
conda activate hw

# 安裝套件
pip install -r requirements.txt

# 安裝 FFmpeg
# macOS: brew install ffmpeg
# Windows: 從 https://www.gyan.dev/ffmpeg/builds/ 下載
# Ubuntu: sudo apt-get install ffmpeg
```

### 2. 準備檔案

**檔案結構：**
```
your_project/
├── multimodal_emotion_analysis.py  # 主程式（改進版）
├── quick_start.py                  # 快速啟動腳本
├── requirements.txt
├── vlog.mp4                        # 你的影片
├── transcript.txt                  # 對話文字（選填）
└── emotion_analysis_results/       # 結果資料夾（自動生成）
```

**建立文字檔案（選填但推薦）：**

`transcript.txt` 範例：
```txt
今天真的很開心！專案終於完成了。
雖然中間遇到很多困難，有時候真的很煩。
但是最後看到成果，覺得一切都值得了。
```

### 3. 執行分析

```bash
# 基本使用（每 2 秒分析，使用自訂文字）
python quick_start.py vlog.mp4 transcript.txt

# 更精細分析（每 1 秒分析，推薦用於報告）
python quick_start.py vlog.mp4 transcript.txt 1

# 不使用文字檔案（自動語音轉錄）
python quick_start.py vlog.mp4

# 快速分析（每 3 秒）
python quick_start.py vlog.mp4 transcript.txt 3
```

---

## 📊 四種基本情緒

| 情緒 | 中文 | 英文 | 圖表顏色 | 說明 |
|-----|------|------|---------|------|
| 😊 | 快樂 | happy | 🟡 金黃色 | 開心、高興、滿意 |
| 😠 | 生氣 | angry | 🔴 紅色 | 憤怒、煩躁、不滿 |
| 😢 | 悲傷 | sad | 🔵 藍色 | 難過、沮喪、失望 |
| 😐 | 中性 | neutral | ⚪ 灰色 | 平靜、普通、無明顯情緒 |

**原本的 7 種情緒映射：**
- `fear` (恐懼) → `sad` (悲傷)
- `disgust` (厭惡) → `angry` (生氣)
- `surprise` (驚訝) → `happy` (快樂)

---

## 💡 為什麼判斷是那種情緒？

### 文字情緒分析 - 詳細解釋

執行後你會看到：

```bash
💬 正在分析文字情緒...

   情緒分析結果：
   快樂 (happy  ): ████████████████░░░░ 78.5%
   中性 (neutral): ████░░░░░░░░░░░░░░░░ 18.2%
   悲傷 (sad    ): █░░░░░░░░░░░░░░░░░░░  2.8%
   生氣 (angry  ): ░░░░░░░░░░░░░░░░░░░░  0.5%

   💡 為什麼是「快樂」？
   原始模型檢測到的情緒：
      • joy: 65.3%
      • happiness: 13.2%
      
      1. 檢測到相關詞彙：開心、高興、完成、成功
      2. 情緒強度明顯高於其他情緒（高出 60.3%）
      3. 根據整體語境判斷為「快樂」

   ⚠️ 同時檢測到其他情緒：
      • 中性 (18.2%)：根據整體語境
```

### 視覺情緒分析 - 摘要說明

```bash
✅ 視覺分析完成！共分析 30 幀

   視覺情緒分析摘要：
   在 30 個時間點中：
   快樂: ████████████████░░░░  24 次 (80.0%)
   中性: ███░░░░░░░░░░░░░░░░░   5 次 (16.7%)
   悲傷: █░░░░░░░░░░░░░░░░░░░   1 次 (3.3%)

   💡 分析要點：
   1. 整體表情以「快樂」為主
   2. 表情有 6 次變化，情緒相對穩定
   3. 在 15.3 秒處，「快樂」情緒最強烈 (92.4%)
```

### 音訊情緒分析 - 特徵說明

```bash
✅ 音訊分析完成！共分析 30 段

   音訊情緒分析摘要：
   在 30 個時間段中：
   快樂: ██████████████░░░░░░  21 次 (70.0%)
   中性: █████░░░░░░░░░░░░░░░   7 次 (23.3%)

   💡 分析要點：
   1. 聲音特徵主要顯示「快樂」情緒
   2. 聲音能量較高且音調明亮
   3. 語調有 8 次明顯變化
```

### 關鍵詞偵測

程式會自動檢測以下關鍵詞：

**快樂 😊**
- 中文：開心、快樂、高興、喜歡、愛、棒、好、讚、太好了、成功、完成、滿意、幸福、笑
- 英文：happy, joy, good, great, love, like, awesome, wonderful, haha, lol

**生氣 😠**
- 中文：生氣、憤怒、氣、討厭、煩、爛、糟、該死、可惡、受不了、火大
- 英文：angry, mad, hate, annoying, terrible, awful, stupid

**悲傷 😢**
- 中文：難過、悲傷、傷心、哭、痛苦、沮喪、失望、遺憾、孤單、寂寞
- 英文：sad, unhappy, cry, tears, depressed, disappointed, lonely

**中性 😐**
- 中文：還好、普通、一般、平常、沒什麼、正常
- 英文：okay, fine, normal, neutral, average

---

## 📈 視覺化圖表

執行後會生成 `emotion_analysis_visualization.png`，包含 **6 個子圖**：

### 圖表佈局

```
┌─────────────────────────────────────────────────────┐
│  1. 視覺情緒時間序列折線圖 (橫跨整排)                  │
├─────────────────────────────────────────────────────┤
│  2. 音訊情緒時間序列折線圖 (橫跨整排)                  │
├─────────────────────────────────────────────────────┤
│  3. 視覺 vs 音訊比較圖 (橫跨整排)                     │
├─────────────┬─────────────┬─────────────────────────┤
│ 4. 視覺圓餅圖 │ 5. 音訊圓餅圖 │  6. 相關性熱圖            │
└─────────────┴─────────────┴─────────────────────────┘
```

### 1-2. 時間序列折線圖

```
情緒強度
  1.0 ┤     ●━━●━━●     快樂 (金黃色)
      │    ●       ●
  0.5 ┤   ●         ●   悲傷 (藍色)
      │  ●           ●
  0.0 ┤●             ●●
      └──────────────────→ 時間 (秒)
      0  5  10  15  20  25
```

- **4 條折線**：快樂、生氣、悲傷、中性
- **視覺用圓點 ●**，**音訊用方塊 ■**
- 顯示情緒強度如何隨時間變化

### 3. 比較圖

```
情緒強度
  1.0 ┤  ━━━ 快樂(視覺) 實線
      │  ┄┄┄ 快樂(音訊) 虛線
  0.5 ┤  可以看出兩者是否同步！
  0.0 ┤
      └──────────────────→ 時間 (秒)
```

- **實線** = 視覺情緒
- **虛線** = 音訊情緒
- **重疊** = 一致 ✅
- **分離** = 不一致 ❌

### 4-5. 圓餅圖

顯示每種情緒的佔比：

```
      快樂 65%
     /        \
  悲傷 10%    中性 20%
     \        /
      生氣 5%
```

### 6. 相關性熱圖

```
        快樂  生氣  悲傷  中性
相關性  0.82  0.45  0.23  0.67

🟢 綠色 = 高相關 (>0.5)
🟡 黃色 = 中度相關 (0.3-0.5)
🔴 紅色 = 低相關 (<0.3)
```

---

## 📊 相關性計算說明

### 什麼是相關性？

相關性計算**視覺和音訊情緒在時間序列上的同步程度**。

### 計算方式

1. 對齊視覺和音訊的時間戳
2. 對每個情緒，提取時間序列：
   ```
   時間 0s: 視覺 happy=0.8, 音訊 happy=0.7
   時間 1s: 視覺 happy=0.6, 音訊 happy=0.5
   時間 2s: 視覺 happy=0.9, 音訊 happy=0.8
   ```
3. 計算**皮爾森相關係數** (Pearson correlation)

### 相關係數解讀

| 相關係數 | 意義 | 解釋 |
|---------|------|------|
| **0.7 ~ 1.0** | ✅ 高度一致 | 視覺和音訊同時高、同時低 |
| **0.3 ~ 0.7** | ⚠️ 中度一致 | 大致同步但有些差異 |
| **0.0 ~ 0.3** | ❌ 一致性低 | 變化趨勢不太相關 |
| **-1.0 ~ 0.0** | ❌ 相反趨勢 | 視覺高時音訊低（說反話？）|

### 實際範例

```
視覺-音訊相關性:
  快樂 (happy): 0.823 ✅ (高度一致)
  生氣 (angry): 0.456 ⚠️ (中度一致)
  悲傷 (sad): 0.234 ❌ (一致性較低)
  中性 (neutral): 0.678 ✅ (高度一致)

平均相關係數: 0.548
✅ 視覺和音訊情緒高度一致
```

**解讀：**
- 快樂和中性情緒：表情和聲音高度同步
- 生氣情緒：中度同步，可能有些時候表情不明顯
- 悲傷情緒：不太一致，可能隱藏悲傷（笑著說難過的事？）

---

## 📁 輸出結果

所有結果保存在 `emotion_analysis_results/` 資料夾：

```
emotion_analysis_results/
├── emotion_analysis_visualization.png  # 綜合視覺化圖表
├── visual_emotions.csv                 # 視覺情緒詳細數據
├── audio_emotions.csv                  # 音訊情緒詳細數據
├── analysis_summary.json               # 分析摘要（JSON格式）
└── extracted_audio.wav                 # 提取的音訊檔案
```

### CSV 檔案格式

**visual_emotions.csv:**
```csv
timestamp,dominant_emotion,happy,angry,sad,neutral
0.0,happy,0.85,0.05,0.03,0.07
1.0,happy,0.78,0.08,0.04,0.10
2.0,neutral,0.15,0.10,0.12,0.63
```

**欄位說明：**
- `timestamp`: 時間點（秒）
- `dominant_emotion`: 主要情緒
- `happy/angry/sad/neutral`: 各情緒的強度（0-1 之間）
---

## 進階使用

### 在 Python 程式中使用

```python
from multimodal_emotion_analysis import MultiModalEmotionAnalyzer

# 建立分析器
analyzer = MultiModalEmotionAnalyzer(
    video_path="my_vlog.mp4",
    text_file="transcript.txt"  # 可選
)

# 執行完整分析
analyzer.run_full_analysis(sample_rate=1)

# 或分步驟執行
visual_df = analyzer.analyze_visual_emotions(sample_rate=1)
audio_path = analyzer.extract_audio()
audio_df = analyzer.analyze_audio_emotions(audio_path)
text, emotion = analyzer.transcribe_and_analyze_text(audio_path)

# 計算相關性
correlations = analyzer.calculate_correlations(visual_df, audio_df)
print(correlations)
```

### 批次處理多個影片

```python
from pathlib import Path
from multimodal_emotion_analysis import MultiModalEmotionAnalyzer

# 處理資料夾中的所有影片
videos = Path("videos").glob("*.mp4")

for video in videos:
    # 尋找對應的文字檔案
    text_file = Path("transcripts") / f"{video.stem}.txt"
    
    print(f"\n處理: {video.name}")
    
    analyzer = MultiModalEmotionAnalyzer(
        str(video),
        text_file=str(text_file) if text_file.exists() else None,
        output_dir=f"results/{video.stem}"
    )
    
    analyzer.run_full_analysis(sample_rate=2)
    print(f" {video.name} 分析完成！")
```


### 自訂情緒映射規則

修改 `multimodal_emotion_analysis.py`：

```python
# 在 MultiModalEmotionAnalyzer.__init__ 中
self.deepface_mapping = {
    'happy': 'happy',
    'angry': 'angry',
    'sad': 'sad',
    'neutral': 'neutral',
    'fear': 'sad',      # 可以改成 'angry'
    'disgust': 'angry', # 可以改成 'sad'
    'surprise': 'happy' # 可以改成 'neutral'
}
```

---

## 使用建議

### 取樣率選擇

| 影片長度 | 情緒變化 | 推薦取樣率 | 說明 |
|---------|---------|-----------|------|
| < 1 分鐘 | 任何 | **1 秒** | 精細分析 |
| 1-3 分鐘 | 快速變化 | **1 秒** | 捕捉細節 |
| 1-3 分鐘 | 一般變化 | **2 秒** | 平衡速度 |
| 3-5 分鐘 | 一般變化 | **2-3 秒** | 效率優先 |
| > 5 分鐘 | 緩慢變化 | **3 秒** | 快速掃描 |

### 文字檔案準備技巧

1. **逐字稿最好**：完整記錄所有對話
2. **摘要也可以**：只記錄主要內容和情緒關鍵詞
3. **分段有幫助**：太長的文字會自動分段分析（每 400 字元）
4. **編碼要正確**：存成 UTF-8 編碼

### 結果解讀技巧

1. **看主要情緒分布**：圓餅圖了解整體基調
2. **觀察時間序列**：找出情緒高峰和低谷
3. **比較兩條線**：實線和虛線重疊 = 表裡一致
4. **相關性數字**：低相關可能代表隱藏情緒或說反話
5. **結合關鍵詞**：文字解釋會顯示為什麼是那種情緒


---

## 完整工作流程範例

### 情境：為期末報告分析 Vlog

```bash
# 1. 準備檔案
# - vlog.mp4（你的影片）
# - transcript.txt（手動輸入對話內容）

# 2. 執行分析（精細模式）
python quick_start.py vlog.mp4 transcript.txt 1

# 3. 同時儲存日誌
python quick_start.py vlog.mp4 transcript.txt 1 | Tee-Object -FilePath report_log.txt

# 4. 查看結果
# - emotion_analysis_visualization.png（放進報告）
# - visual_emotions.csv（用 Excel 做進一步分析）
# - report_log.txt（包含詳細解釋）

# 5. 如果需要調整，重新執行
python quick_start.py vlog.mp4 transcript_v2.txt 1
```

### 使用的模型和工具

- **DeepFace**: 臉部情緒識別
- **Librosa**: 音訊特徵提取
- **Transformers**: 文字情緒分析
- **MoviePy**: 影片處理
- **OpenCV**: 電腦視覺