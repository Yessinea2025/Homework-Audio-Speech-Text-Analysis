"""
快速啟動腳本 - 改進版
支援自訂文字檔案 + 只分析喜怒哀樂四種情緒
"""

from multimodal_emotion_analysis import MultiModalEmotionAnalyzer
import sys
from pathlib import Path

def quick_analysis(video_path, text_file=None, sample_rate=2, use_chinese=True):
    """
    快速分析影片情緒
    
    參數:
        video_path: 影片路徑
        text_file: 文字檔案路徑（選填）
        sample_rate: 取樣率（秒）- 預設每2秒分析一幀
        use_chinese: 是否使用中文情緒分析（預設 True）
    """
    model_type = "中文關鍵詞模型" if use_chinese else "英文 NLP 模型"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       多模態情緒分析系統 - 快速啟動                            ║
║       Multi-Modal Emotion Analysis - Quick Start             ║
╚══════════════════════════════════════════════════════════════╝

✨ 新功能:
   - 只分析喜怒哀樂四種基本情緒
   - 支援自訂文字檔案（不需要語音轉錄）
   - 支援中文情緒分析（更準確）

📁 影片檔案: {video_path}
📝 文字檔案: {text_file if text_file else '使用自動語音轉錄'}
⏱️  取樣率: 每 {sample_rate} 秒
🎭 情緒類別: 快樂、生氣、悲傷、中性
🧠 文字模型: {model_type}
    """)
    
    # 創建分析器
    analyzer = MultiModalEmotionAnalyzer(
        video_path, 
        text_file=text_file,
        use_chinese_model=use_chinese
    )
    
    # 執行分析
    analyzer.run_full_analysis(sample_rate=sample_rate)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ 分析完成！結果已保存在 emotion_analysis_results/ 資料夾   ║
╚══════════════════════════════════════════════════════════════╝

📊 查看結果:
   - emotion_analysis_visualization.png  (視覺化圖表)
   - visual_emotions.csv                 (視覺情緒數據)
   - audio_emotions.csv                  (音訊情緒數據)
   - analysis_summary.json               (摘要統計)
    """)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║                      使用方法                                  ║
╚════════════════════════════════════════════════════════════════╝

基本用法:
    python quick_start_improved.py <video_path>

進階用法:
    python quick_start_improved.py <video_path> <text_file> [sample_rate]

範例 1 - 基本分析（自動語音轉錄）:
    python quick_start_improved.py my_vlog.mp4

範例 2 - 使用自己的文字檔案:
    python quick_start_improved.py my_vlog.mp4 transcript.txt

範例 3 - 使用文字檔案 + 調整取樣率:
    python quick_start_improved.py my_vlog.mp4 transcript.txt 1

範例 4 - 只調整取樣率（不用文字檔案）:
    python quick_start_improved.py my_vlog.mp4 2

文字檔案格式:
    - 純文字檔案 (.txt)
    - UTF-8 編碼
    - 內容為影片中的對話或旁白

參數說明:
    video_path   : 影片路徑（必填）
    text_file    : 文字檔案路徑（選填，.txt 檔案）
    sample_rate  : 取樣率，每 N 秒分析一幀（選填，預設 2）
                   - 1 秒：較精細但較慢
                   - 2 秒：平衡速度和精度（推薦）
                   - 3 秒：較快但較粗略
        """)
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 判斷參數
    text_file = None
    sample_rate = 2.0
    
    if len(sys.argv) > 2:
        # 檢查第二個參數是文字檔案還是數字
        if sys.argv[2].endswith('.txt'):
            text_file = sys.argv[2]
            if len(sys.argv) > 3:
                sample_rate = float(sys.argv[3])
        else:
            sample_rate = float(sys.argv[2])
    
    # 驗證檔案存在
    if not Path(video_path).exists():
        print(f"❌ 錯誤：找不到影片檔案 '{video_path}'")
        sys.exit(1)
    
    if text_file and not Path(text_file).exists():
        print(f"❌ 錯誤：找不到文字檔案 '{text_file}'")
        sys.exit(1)
    
    # 執行分析
    quick_analysis(video_path, text_file, sample_rate)