# 多模態情緒分析系統

## 系統簡介

這個系統可以分析影片中的**視覺（臉部表情）**、**音訊**和**文字**三種模態的情緒，並比較它們之間的一致性。


## 輸出結果

所有結果保存在 `emotion_analysis_results/` 資料夾：

```
emotion_analysis_results/
├── analysis_result.txt                 # 綜合結果說明
├── analysis_summary.json               # 分析摘要（JSON格式）
├── emotion_analysis_visualization.png  # 綜合視覺化圖表
├── visual_emotions.csv                 # 視覺情緒詳細數據
├── audio_emotions.csv                  # 音訊情緒詳細數據
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

### 結果解讀說明

1. **看主要情緒分布**：圓餅圖了解整體基調
2. **觀察時間序列**：找出情緒高峰和低谷
3. **比較兩條線**：實線和虛線重疊 = 表裡一致
4. **相關性數字**：低相關可能代表隱藏情緒或說反話
5. **結合關鍵詞**：文字解釋會顯示為什麼是那種情緒


---

### 使用的模型和工具

- **DeepFace**: 臉部情緒識別
- **Librosa**: 音訊特徵提取
- **Transformers**: 文字情緒分析
- **MoviePy**: 影片處理
- **OpenCV**: 電腦視覺