"""
Multi-Modal Emotion Analysis for Video Files (改進版)
分析影片的語音、文字、和視覺情緒
- 支援自訂文字檔案輸入
- 只分析四種基本情緒：喜怒哀樂 (happy, angry, sad, neutral)
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Audio and speech processing
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# Deep learning for emotion recognition
try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not installed. Install with: pip install deepface")

try:
    from transformers import pipeline
except ImportError:
    print("Transformers not installed. Install with: pip install transformers")


class MultiModalEmotionAnalyzer:
    """多模態情緒分析器"""
    
    def __init__(self, video_path, output_dir="emotion_analysis_results", text_file=None, use_chinese_model=True):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.text_file = text_file  # 新增：自訂文字檔案路徑
        self.use_chinese_model = use_chinese_model  # 新增：是否使用中文模型
        
        # Store analysis results
        self.visual_emotions = []
        self.audio_emotions = []
        self.text_emotions = []
        self.timestamps = []
        
        # 改為只有四種基本情緒
        self.emotion_labels = ['happy', 'angry', 'sad', 'neutral']
        
        # 中文情緒關鍵詞字典（統一定義）
        self.emotion_keywords = {
            'happy': {
                'keywords': [
                    # 中文
                    '開心', '快樂', '高興', '喜歡', '愛', '棒', '好', '讚', '太好了', 
                    '成功', '完成', '達成', '滿意', '幸福', '享受', '美好', '興奮',
                    '哈哈', '呵呵', '笑', '^^', '😊', '開懷', '愉快', '歡樂', '欣喜', '喜悅',
                    '爽', '酷', '棒棒', '厲害', '牛', '贊', '讚讚', '耶',
                    '太棒', '完美', '優秀', '精彩', '很好', '不錯', '滿足',
                    # 英文
                    'happy', 'joy', 'good', 'great', 'love', 'like', 'awesome', 
                    'wonderful', 'excellent', 'amazing', 'fantastic', 'glad',
                    'haha', 'lol', 'smile', 'laugh', 'yay', 'hooray'
                ],
                'weight': 1.0
            },
            'angry': {
                'keywords': [
                    # 中文
                    '生氣', '憤怒', '氣', '討厭', '煩', '爛', '糟', '該死', '可惡',
                    '受不了', '忍不住', '火大', '惱火', '抓狂', '白痴', '智障',
                    '垃圾', '廢物', '去死', '混蛋', '靠', '幹', '操', '他媽',
                    '煩死', '煩人', '煩躁', '惱', '怒', '不爽', '不滿', '抱怨',
                    # 英文
                    'angry', 'mad', 'hate', 'annoying', 'terrible', 'awful',
                    'damn', 'shit', 'fuck', 'stupid', 'idiot', 'pissed'
                ],
                'weight': 1.2
            },
            'sad': {
                'keywords': [
                    # 中文
                    '難過', '悲傷', '傷心', '哭', '痛苦', '沮喪', '失望', '遺憾',
                    '可憐', '淚', '絕望', '憂鬱', '孤單', '寂寞', '無助', '心痛',
                    '難受', '不開心', '憂傷', '悲慘', '慘', '淒慘', '悽慘',
                    '傷感', '感傷', '哀傷', '悲哀', '慘淡', '低落', '消極',
                    # 英文
                    'sad', 'unhappy', 'cry', 'tears', 'depressed', 'disappointed',
                    'miserable', 'sorry', 'painful', 'hurt', 'lonely', 'upset'
                ],
                'weight': 1.1
            },
            'neutral': {
                'keywords': [
                    # 中文
                    '還好', '普通', '一般', '平常', '沒什麼', '正常', '可以',
                    '尚可', '平淡', '平凡', '平靜', '冷靜', '理性', '客觀',
                    # 英文
                    'okay', 'fine', 'normal', 'neutral', 'average', 'so-so'
                ],
                'weight': 0.8
            }
        }
        
        # 否定詞和程度副詞
        self.negation_words = ['不', '沒', '無', '非', '未', '別', '莫', '勿', '毋']
        self.intensifiers = {
            '非常': 1.5, '很': 1.3, '超': 1.4, '特別': 1.3, '極': 1.5,
            '相當': 1.2, '十分': 1.4, '格外': 1.3, '太': 1.4, '真': 1.2,
            '好': 1.2, '超級': 1.5, '巨': 1.4, '爆': 1.4
        }
        
        # DeepFace 到我們的情緒映射
        self.deepface_mapping = {
            'happy': 'happy',
            'angry': 'angry',
            'sad': 'sad',
            'neutral': 'neutral',
            'fear': 'sad',      # 恐懼歸類為悲傷
            'disgust': 'angry',  # 厭惡歸類為生氣
            'surprise': 'happy'  # 驚訝歸類為快樂
        }
        
    def extract_audio(self):
        """提取影片音訊"""
        print("📽️ 正在提取音訊...")
        try:
            video = VideoFileClip(self.video_path)
            audio_path = self.output_dir / "extracted_audio.wav"
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            video.close()
            print(f"✅ 音訊已保存到: {audio_path}")
            return str(audio_path)
        except Exception as e:
            print(f"❌ 音訊提取失敗: {e}")
            return None
    
    def _map_deepface_to_basic_emotions(self, deepface_emotions):
        """將 DeepFace 的 7 種情緒映射到 4 種基本情緒"""
        basic_emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        
        for deepface_emotion, score in deepface_emotions.items():
            basic_emotion = self.deepface_mapping.get(deepface_emotion.lower())
            if basic_emotion:
                basic_emotions[basic_emotion] += score
        
        # 正規化
        total = sum(basic_emotions.values())
        if total > 0:
            basic_emotions = {k: v/total for k, v in basic_emotions.items()}
        
        return basic_emotions
    
    def analyze_visual_emotions(self, sample_rate=1):
        """分析視覺情緒（臉部表情）- 只輸出 4 種情緒"""
        print("\n👤 正在分析視覺情緒...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frame_interval = int(fps * sample_rate)
        frame_idx = 0
        analyzed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], 
                                             enforce_detection=False, silent=True)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    # 將 7 種情緒映射到 4 種
                    deepface_emotions = result['emotion']
                    basic_emotions = self._map_deepface_to_basic_emotions(deepface_emotions)
                    
                    dominant_emotion = max(basic_emotions, key=basic_emotions.get)
                    
                    self.visual_emotions.append({
                        'timestamp': timestamp,
                        'dominant_emotion': dominant_emotion,
                        **basic_emotions
                    })
                    analyzed_count += 1
                    
                    if analyzed_count % 10 == 0:
                        print(f"  已分析 {analyzed_count} 幀 ({timestamp:.1f}s / {duration:.1f}s)")
                
                except Exception as e:
                    self.visual_emotions.append({
                        'timestamp': timestamp,
                        'dominant_emotion': 'neutral',
                        **{emotion: 0 for emotion in self.emotion_labels}
                    })
            
            frame_idx += 1
        
        cap.release()
        print(f"✅ 視覺分析完成！共分析 {analyzed_count} 幀")
        
        # 加入分析摘要和原因解釋
        if self.visual_emotions:
            self._explain_visual_analysis()
        
        return pd.DataFrame(self.visual_emotions)
    
    def _explain_visual_analysis(self):
        """解釋視覺分析結果"""
        df = pd.DataFrame(self.visual_emotions)
        
        print(f"\n   📊 視覺情緒分析摘要：")
        emotion_zh = {'happy': '快樂', 'angry': '生氣', 'sad': '悲傷', 'neutral': '中性'}
        
        # 統計主要情緒
        emotion_counts = df['dominant_emotion'].value_counts()
        total = len(df)
        
        print(f"   在 {total} 個時間點中：")
        for emotion, count in emotion_counts.items():
            percentage = count / total * 100
            bar_length = int(percentage / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"   {emotion_zh[emotion]:3s}: {bar} {count:3d} 次 ({percentage:.1f}%)")
        
        # 找出情緒變化最大的時刻
        print(f"\n   💡 分析要點：")
        
        # 1. 主要情緒
        dominant = emotion_counts.index[0]
        print(f"   1. 整體表情以「{emotion_zh[dominant]}」為主")
        
        # 2. 情緒變化
        emotion_changes = 0
        for i in range(1, len(df)):
            if df.iloc[i]['dominant_emotion'] != df.iloc[i-1]['dominant_emotion']:
                emotion_changes += 1
        
        if emotion_changes > len(df) * 0.3:
            print(f"   2. 表情變化頻繁（共 {emotion_changes} 次變化），情緒較不穩定")
        elif emotion_changes > 0:
            print(f"   3. 表情有 {emotion_changes} 次變化，情緒相對穩定")
        else:
            print(f"   2. 表情始終保持一致")
        
        # 3. 找出最強烈的情緒時刻
        for emotion in self.emotion_labels:
            if emotion in df.columns:
                max_idx = df[emotion].idxmax()
                max_value = df[emotion].max()
                if max_value > 0.7:  # 只顯示強烈的情緒
                    timestamp = df.iloc[max_idx]['timestamp']
                    print(f"   3. 在 {timestamp:.1f} 秒處，「{emotion_zh[emotion]}」情緒最強烈 ({max_value:.1%})")
                    break
    
    def analyze_audio_emotions(self, audio_path):
        """分析音訊情緒（基於聲學特徵）- 只輸出 4 種情緒"""
        print("\n🎵 正在分析音訊情緒...")
        
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            segment_length = sr
            num_segments = int(duration)
            
            for i in range(num_segments):
                start_sample = i * segment_length
                end_sample = min((i + 1) * segment_length, len(y))
                segment = y[start_sample:end_sample]
                
                features = self._extract_audio_features(segment, sr)
                emotion_scores = self._map_features_to_emotions(features)
                
                self.audio_emotions.append({
                    'timestamp': i,
                    'dominant_emotion': max(emotion_scores, key=emotion_scores.get),
                    **emotion_scores
                })
            
            print(f"✅ 音訊分析完成！共分析 {num_segments} 段")
            
            # 加入分析摘要和原因解釋
            if self.audio_emotions:
                self._explain_audio_analysis()
            
            return pd.DataFrame(self.audio_emotions)
        
        except Exception as e:
            print(f"❌ 音訊分析失敗: {e}")
            return pd.DataFrame()
    
    def _explain_audio_analysis(self):
        """解釋音訊分析結果"""
        df = pd.DataFrame(self.audio_emotions)
        
        print(f"\n   📊 音訊情緒分析摘要：")
        emotion_zh = {'happy': '快樂', 'angry': '生氣', 'sad': '悲傷', 'neutral': '中性'}
        
        # 統計主要情緒
        emotion_counts = df['dominant_emotion'].value_counts()
        total = len(df)
        
        print(f"   在 {total} 個時間段中：")
        for emotion, count in emotion_counts.items():
            percentage = count / total * 100
            bar_length = int(percentage / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"   {emotion_zh[emotion]:3s}: {bar} {count:3d} 次 ({percentage:.1f}%)")
        
        print(f"\n   💡 分析要點：")
        
        # 1. 主要情緒
        dominant = emotion_counts.index[0]
        print(f"   1. 聲音特徵主要顯示「{emotion_zh[dominant]}」情緒")
        
        # 2. 音訊特徵說明
        reasoning = {
            'happy': "聲音能量較高且音調明亮",
            'angry': "聲音能量高且音調變化大",
            'sad': "聲音能量低且音調較低沉",
            'neutral': "聲音特徵平穩，無明顯情緒波動"
        }
        print(f"   2. {reasoning[dominant]}")
        
        # 3. 情緒變化
        emotion_changes = 0
        for i in range(1, len(df)):
            if df.iloc[i]['dominant_emotion'] != df.iloc[i-1]['dominant_emotion']:
                emotion_changes += 1
        
        if emotion_changes > 0:
            print(f"   3. 語調有 {emotion_changes} 次明顯變化")
    
    def _extract_audio_features(self, audio_segment, sr):
        """提取音訊特徵"""
        features = {}
        
        features['energy'] = np.mean(librosa.feature.rms(y=audio_segment))
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
        
        pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        features['pitch_mean'] = np.mean(pitch_values) if pitch_values else 0
        features['pitch_std'] = np.std(pitch_values) if pitch_values else 0
        
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc)
        features['mfcc_std'] = np.std(mfcc)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        return features
    
    def _map_features_to_emotions(self, features):
        """將聲學特徵映射到 4 種基本情緒"""
        emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # 高能量 + 高音調 = happy
        if features['energy'] > 0.05 and features['pitch_mean'] > 200:
            emotions['happy'] = 0.6
        
        # 高能量 + 高變化 = angry
        if features['energy'] > 0.06 and features['pitch_std'] > 50:
            emotions['angry'] = 0.7
        
        # 低能量 + 低音調 = sad
        if features['energy'] < 0.03 and features['pitch_mean'] < 180:
            emotions['sad'] = 0.6
        
        # 正常化
        total = sum(emotions.values())
        if total == 0:
            emotions['neutral'] = 1.0
        else:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def load_text_from_file(self):
        """從檔案載入文字（新增功能）"""
        if not self.text_file:
            return None
        
        print(f"\n📄 正在從檔案載入文字: {self.text_file}")
        try:
            with open(self.text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            print(f"✅ 成功載入 {len(text)} 個字元")
            return text
        except Exception as e:
            print(f"❌ 載入文字檔案失敗: {e}")
            return None
    
    
    def _explain_emotion_reasoning(self, text, dominant_emotion, basic_emotions, emotion_details):
        """解釋情緒判斷的原因"""
        
        # 找出文字中的關鍵詞（使用類別屬性）
        found_keywords = {emotion: [] for emotion in self.emotion_labels}
        text_lower = text.lower()
        
        for emotion, config in self.emotion_keywords.items():
            for keyword in config['keywords']:
                if keyword.lower() in text_lower:
                    found_keywords[emotion].append(keyword)
        
        # 顯示主要情緒的原因
        emotion_zh = {'happy': '快樂', 'angry': '生氣', 'sad': '悲傷', 'neutral': '中性'}
        
        reasons = []
        
        # 原因 1: 關鍵詞
        if found_keywords[dominant_emotion]:
            keywords_str = '、'.join(found_keywords[dominant_emotion][:5])  # 最多顯示 5 個
            reasons.append(f"檢測到相關詞彙：{keywords_str}")
        
        # 原因 2: 分數差距
        sorted_emotions = sorted(basic_emotions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) > 1:
            second_emotion, second_score = sorted_emotions[1]
            dominant_score = basic_emotions[dominant_emotion]
            
            if dominant_score - second_score > 0.3:
                reasons.append(f"情緒強度明顯高於其他情緒（高出 {(dominant_score-second_score):.1%}）")
            elif dominant_score - second_score < 0.1:
                reasons.append(f"與「{emotion_zh[second_emotion]}」情緒相近，但略高一些")
        
        # 原因 3: 原始模型的情緒分佈（僅用於英文模型）
        if emotion_details and self.use_chinese_model == False:
            print(f"   原始模型檢測到的情緒：")
            for orig_emotion, scores in emotion_details.items():
                avg_score = sum(scores) / len(scores) if isinstance(scores, list) else scores
                if avg_score > 0.1:
                    print(f"      • {orig_emotion}: {avg_score:.1%}")
        
        # 原因 4: 情緒混合
        high_emotions = [e for e, s in basic_emotions.items() if s > 0.2]
        if len(high_emotions) > 1:
            emotions_str = '、'.join([emotion_zh[e] for e in high_emotions])
            reasons.append(f"文字包含多種情緒：{emotions_str}")
        
        # 原因 5: 文字長度
        if len(text) > 500:
            reasons.append(f"文字較長，包含多個情緒段落")
        
        # 顯示原因
        if reasons:
            for i, reason in enumerate(reasons, 1):
                print(f"      {i}. {reason}")
        else:
            print(f"      根據整體語境判斷為「{emotion_zh[dominant_emotion]}」")
        
        # 如果有其他明顯情緒，也說明
        other_emotions = [(e, s) for e, s in basic_emotions.items() 
                         if e != dominant_emotion and s > 0.15]
        if other_emotions:
            print(f"\n   ⚠️ 同時檢測到其他情緒：")
            for emotion, score in sorted(other_emotions, key=lambda x: x[1], reverse=True):
                keywords = found_keywords[emotion]
                if keywords:
                    keywords_str = '、'.join(keywords[:3])
                    print(f"      • {emotion_zh[emotion]} ({score:.1%})：可能因為「{keywords_str}」等詞彙")
                else:
                    print(f"      • {emotion_zh[emotion]} ({score:.1%})：根據整體語境")
    
    def transcribe_and_analyze_text(self, audio_path):
        """轉錄語音並分析文字情緒（改進版）"""
        
        # 優先使用自訂文字檔案
        if self.text_file:
            text = self.load_text_from_file()
            if text:
                return self._analyze_text_emotion(text)
        
        # 如果沒有自訂文字，才使用語音轉錄
        print("\n📝 正在轉錄語音...")
        
        try:
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            
            text_zh = ""
            text_en = ""
            
            try:
                text_zh = recognizer.recognize_google(audio_data, language='zh-TW')
                print(f"  中文轉錄: {text_zh}")
            except:
                print("  無法辨識中文語音")
            
            try:
                text_en = recognizer.recognize_google(audio_data, language='en-US')
                print(f"  英文轉錄: {text_en}")
            except:
                print("  無法辨識英文語音")
            
            text = text_zh if text_zh else text_en
            
            if text:
                return self._analyze_text_emotion(text)
            else:
                print("⚠️ 無法轉錄語音")
                return None, None
        
        except Exception as e:
            print(f"❌ 語音轉錄失敗: {e}")
            return None, None
    
    def _analyze_text_emotion(self, text):
        """分析文字情緒（內部方法）- 支援中英文模型"""
        print("\n💬 正在分析文字情緒...")
        
        try:
            # 處理長文字：如果太長，分段分析
            max_length = 400  # 安全長度，確保不超過 512 tokens
            
            if len(text) > max_length:
                print(f"   ⚠️ 文字較長 ({len(text)} 字元)，將分段分析...")
                # 分成多段
                segments = []
                for i in range(0, len(text), max_length):
                    segments.append(text[i:i+max_length])
                print(f"   分為 {len(segments)} 段進行分析")
            else:
                segments = [text]
            
            # 選擇模型
            if self.use_chinese_model:
                print("   使用中文情緒分析模型...")
                emotion_scores = self._analyze_chinese_emotion(segments)
            else:
                print("   使用英文情緒分析模型...")
                emotion_scores = self._analyze_english_emotion(segments)
            
            # 找出主要情緒
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # 儲存詳細資訊（用於解釋）
            self.text_emotions.append({
                'text': text,
                'dominant_emotion': dominant_emotion,
                'emotion_details': emotion_scores,
                **emotion_scores
            })
            
            print(f"✅ 文字分析完成！")
            print(f"   文字內容: {text[:100]}..." if len(text) > 100 else f"   文字內容: {text}")
            
            # 詳細情緒分析說明
            print(f"\n   📊 情緒分析結果：")
            emotion_zh = {'happy': '快樂', 'angry': '生氣', 'sad': '悲傷', 'neutral': '中性'}
            
            # 排序顯示（由高到低）
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions:
                bar_length = int(score * 20)  # 20 個字元的進度條
                bar = '█' * bar_length + '░' * (20 - bar_length)
                print(f"   {emotion_zh[emotion]:3s} ({emotion:7s}): {bar} {score:.1%}")
            
            # 解釋為什麼是這個情緒
            print(f"\n   💡 為什麼是「{emotion_zh[dominant_emotion]}」？")
            self._explain_emotion_reasoning(text, dominant_emotion, emotion_scores, {})
            
            return text, emotion_scores
        
        except Exception as e:
            print(f"❌ 文字情緒分析失敗: {e}")
            return None, None
    
    def _analyze_chinese_emotion(self, segments):
        """使用中文模型分析情緒"""
        try:
            # 方法 1: 嘗試使用 ckiplab 的中文 BERT 情緒模型
            try:
                from transformers import BertTokenizer, BertForSequenceClassification
                import torch
                
                print("   載入 ckiplab 中文情緒模型...")
                # 這個模型專門為中文情緒分析訓練
                model_name = "ckiplab/bert-base-chinese-ner"
                tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
                
                # 使用關鍵詞方法作為主要方式
                emotion_scores = self._keyword_based_emotion_analysis(segments)
                print("   使用關鍵詞方法分析")
                
                return emotion_scores
                
            except Exception as e:
                print(f"   ⚠️ 中文 BERT 模型載入失敗，使用關鍵詞方法")
                return self._keyword_based_emotion_analysis(segments)
        
        except Exception as e:
            print(f"   ⚠️ 分析失敗: {e}，使用關鍵詞方法")
            return self._keyword_based_emotion_analysis(segments)
    
    def _keyword_based_emotion_analysis(self, segments):
        """基於關鍵詞的情緒分析（使用類別屬性的關鍵詞字典）"""
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # 分析每個段落
        for segment in segments:
            segment_lower = segment.lower()
            
            # 檢查每個情緒的關鍵詞
            for emotion, config in self.emotion_keywords.items():
                for keyword in config['keywords']:
                    if keyword in segment:
                        score = config['weight']
                        
                        # 檢查前面是否有否定詞（往前看 2 個字）
                        keyword_pos = segment.find(keyword)
                        start_idx = max(0, keyword_pos - 2)
                        prefix = segment[start_idx:keyword_pos]
                        
                        has_negation = any(neg in prefix for neg in self.negation_words)
                        
                        # 檢查程度副詞
                        intensifier_multiplier = 1.0
                        for intensifier, multiplier in self.intensifiers.items():
                            if intensifier in prefix:
                                intensifier_multiplier = multiplier
                                break
                        
                        # 計算分數
                        if has_negation:
                            # 否定詞會降低這個情緒，增加 neutral
                            score *= -0.5
                            emotion_scores['neutral'] += 0.3
                        else:
                            score *= intensifier_multiplier
                        
                        emotion_scores[emotion] += score
        
        # 正規化分數
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            # 如果沒有檢測到任何情緒，設為 neutral
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores
    
    def _analyze_english_emotion(self, segments):
        """使用英文模型分析情緒（原本的方法）"""
        emotion_analyzer = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base",
                                   top_k=None,
                                   truncation=True,
                                   max_length=512)
        
        # 分析每一段並取平均
        all_emotions = []
        for idx, segment in enumerate(segments):
            if len(segments) > 1:
                print(f"   分析第 {idx+1}/{len(segments)} 段...")
            results = emotion_analyzer(segment)
            all_emotions.append(results[0])
        
        # 合併多段結果（取平均）
        if len(all_emotions) > 1:
            print(f"   合併 {len(all_emotions)} 段的分析結果...")
        
        results = [all_emotions[0]]  # 使用第一段的結果結構
        # 平均所有段的分數
        emotion_scores_sum = {}
        for segment_results in all_emotions:
            for item in segment_results:
                label = item['label']
                score = item['score']
                emotion_scores_sum[label] = emotion_scores_sum.get(label, 0) + score
        
        # 取平均並重建結果格式
        results = [[{'label': label, 'score': score / len(all_emotions)} 
                   for label, score in emotion_scores_sum.items()]]
        
        # 映射到 4 種基本情緒
        emotion_mapping = {
            'joy': 'happy',
            'happiness': 'happy',
            'anger': 'angry',
            'sadness': 'sad',
            'fear': 'sad',
            'disgust': 'angry',
            'surprise': 'happy',
            'neutral': 'neutral'
        }
        
        basic_emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        
        for result in results[0]:
            label = result['label'].lower()
            score = result['score']
            mapped_emotion = emotion_mapping.get(label, 'neutral')
            basic_emotions[mapped_emotion] += score
        
        # 正規化
        total = sum(basic_emotions.values())
        if total > 0:
            basic_emotions = {k: v/total for k, v in basic_emotions.items()}
        
        return basic_emotions
    
    def calculate_correlations(self, visual_df, audio_df):
        """計算不同模態之間的相關性
        
        說明：
        - 對每個情緒，計算視覺和音訊在時間序列上的皮爾森相關係數
        - 相關係數範圍：-1 到 1
        - 接近 1：兩者變化趨勢非常一致（同時高、同時低）
        - 接近 0：兩者變化無關
        - 接近 -1：兩者變化趨勢相反
        """
        print("\n📊 正在計算相關性...")
        print("   說明：計算視覺和音訊情緒強度的時間序列相關性")
        
        correlations = {}
        
        # 對齊時間戳
        common_timestamps = set(visual_df['timestamp']).intersection(set(audio_df['timestamp']))
        
        if len(common_timestamps) > 0:
            visual_aligned = visual_df[visual_df['timestamp'].isin(common_timestamps)].sort_values('timestamp')
            audio_aligned = audio_df[audio_df['timestamp'].isin(common_timestamps)].sort_values('timestamp')
            
            for emotion in self.emotion_labels:
                if emotion in visual_aligned.columns and emotion in audio_aligned.columns:
                    visual_values = visual_aligned[emotion].values
                    audio_values = audio_aligned[emotion].values
                    
                    # 計算皮爾森相關係數
                    corr = np.corrcoef(visual_values, audio_values)[0, 1]
                    correlations[emotion] = corr if not np.isnan(corr) else 0
                    
                    # 詳細說明
                    print(f"   {emotion}: {corr:.3f}", end="")
                    if corr > 0.5:
                        print(" (高度一致 ✅)")
                    elif corr > 0.3:
                        print(" (中度一致 ⚠️)")
                    else:
                        print(" (一致性較低 ❌)")
            
            print("✅ 相關性計算完成！")
        else:
            print("⚠️ 無法對齊時間戳")
        
        return correlations
    
    def visualize_results(self, visual_df, audio_df, text_emotion=None, correlations=None):
        """視覺化分析結果（改進版：只顯示 4 種情緒）"""
        print("\n📈 正在生成視覺化圖表...")
        
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 定義中文標籤
        emotion_labels_zh = {
            'happy': '快樂',
            'angry': '生氣',
            'sad': '悲傷',
            'neutral': '中性'
        }
        
        colors = {
            'happy': '#FFD700',    # 金黃色
            'angry': '#FF4444',    # 紅色
            'sad': '#4169E1',      # 藍色
            'neutral': '#808080'   # 灰色
        }
        
        # 1. 視覺情緒時間序列
        ax1 = fig.add_subplot(gs[0, :])
        for emotion in self.emotion_labels:
            if emotion in visual_df.columns:
                ax1.plot(visual_df['timestamp'], visual_df[emotion], 
                        label=f'{emotion_labels_zh[emotion]} ({emotion})', 
                        marker='o', markersize=4, linewidth=2.5,
                        color=colors[emotion])
        ax1.set_xlabel('時間 (秒)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('情緒強度', fontsize=12, fontweight='bold')
        ax1.set_title('視覺情緒分析 (Visual/Facial Emotion)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', ncol=4, fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # 2. 音訊情緒時間序列
        ax2 = fig.add_subplot(gs[1, :])
        for emotion in self.emotion_labels:
            if emotion in audio_df.columns:
                ax2.plot(audio_df['timestamp'], audio_df[emotion], 
                        label=f'{emotion_labels_zh[emotion]} ({emotion})', 
                        marker='s', markersize=4, linewidth=2.5,
                        color=colors[emotion])
        ax2.set_xlabel('時間 (秒)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('情緒強度', fontsize=12, fontweight='bold')
        ax2.set_title('音訊情緒分析 (Audio/Voice Emotion)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', ncol=4, fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # 3. 疊加比較圖
        ax3 = fig.add_subplot(gs[2, :])
        for emotion in self.emotion_labels:
            if emotion in visual_df.columns:
                ax3.plot(visual_df['timestamp'], visual_df[emotion], 
                        label=f'{emotion_labels_zh[emotion]} (視覺)', 
                        linestyle='-', linewidth=2.5, color=colors[emotion], alpha=0.8)
            if emotion in audio_df.columns:
                ax3.plot(audio_df['timestamp'], audio_df[emotion], 
                        label=f'{emotion_labels_zh[emotion]} (音訊)', 
                        linestyle='--', linewidth=2.5, color=colors[emotion], alpha=0.6)
        
        ax3.set_xlabel('時間 (秒)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('情緒強度', fontsize=12, fontweight='bold')
        ax3.set_title('視覺 vs 音訊情緒比較 (Visual vs Audio Comparison)', 
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', ncol=4, fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        
        # 4. 主要情緒分布 - 視覺
        ax4 = fig.add_subplot(gs[3, 0])
        visual_counts = visual_df['dominant_emotion'].value_counts()
        pie_colors = [colors[emotion] for emotion in visual_counts.index]
        pie_labels = [f"{emotion_labels_zh[e]}\n({e})" for e in visual_counts.index]
        ax4.pie(visual_counts.values, labels=pie_labels, autopct='%1.1f%%',
               colors=pie_colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax4.set_title('視覺主要情緒分布', fontsize=12, fontweight='bold')
        
        # 5. 主要情緒分布 - 音訊
        ax5 = fig.add_subplot(gs[3, 1])
        audio_counts = audio_df['dominant_emotion'].value_counts()
        pie_colors = [colors[emotion] for emotion in audio_counts.index]
        pie_labels = [f"{emotion_labels_zh[e]}\n({e})" for e in audio_counts.index]
        ax5.pie(audio_counts.values, labels=pie_labels, autopct='%1.1f%%',
               colors=pie_colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax5.set_title('音訊主要情緒分布', fontsize=12, fontweight='bold')
        
        # 6. 相關性熱圖
        if correlations:
            ax6 = fig.add_subplot(gs[3, 2])
            # 轉換為中文標籤
            corr_data_zh = {emotion_labels_zh[k]: v for k, v in correlations.items()}
            corr_df = pd.DataFrame([corr_data_zh])
            sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, vmin=-1, vmax=1, ax=ax6, 
                       cbar_kws={'label': '相關係數'})
            ax6.set_title('視覺-音訊相關性\n(Visual-Audio Correlation)', 
                         fontsize=12, fontweight='bold')
            ax6.set_yticklabels([])
        
        plt.suptitle('多模態情緒分析結果 (Multi-Modal Emotion Analysis)\n喜怒哀樂四種基本情緒', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / "emotion_analysis_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 圖表已保存到: {output_path}")
        
        return fig
    
    def save_results(self, visual_df, audio_df, text_emotion, correlations):
        """保存分析結果"""
        print("\n💾 正在保存結果...")
        
        visual_df.to_csv(self.output_dir / "visual_emotions.csv", index=False, encoding='utf-8-sig')
        audio_df.to_csv(self.output_dir / "audio_emotions.csv", index=False, encoding='utf-8-sig')
        
        summary = {
            'video_path': str(self.video_path),
            'text_file': str(self.text_file) if self.text_file else None,
            'emotion_labels': self.emotion_labels,
            'visual_emotion_stats': visual_df['dominant_emotion'].value_counts().to_dict(),
            'audio_emotion_stats': audio_df['dominant_emotion'].value_counts().to_dict(),
            'correlations': correlations if correlations else {},
            'text_emotion': text_emotion if text_emotion else {}
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果已保存到: {self.output_dir}")
    
    def run_full_analysis(self, sample_rate=1):
        """執行完整分析流程"""
        print("=" * 80)
        print("🎬 開始多模態情緒分析（改進版）")
        print("   - 情緒類別：喜怒哀樂四種基本情緒")
        if self.text_file:
            print(f"   - 使用自訂文字檔案：{self.text_file}")
        print("=" * 80)
        
        # 1. 提取音訊
        audio_path = self.extract_audio()
        
        # 2. 視覺分析
        visual_df = self.analyze_visual_emotions(sample_rate=sample_rate)
        
        # 3. 音訊分析
        audio_df = pd.DataFrame()
        if audio_path:
            audio_df = self.analyze_audio_emotions(audio_path)
        
        # 4. 文字分析（優先使用自訂文字）
        text_emotion = None
        if audio_path or self.text_file:
            text, text_emotion = self.transcribe_and_analyze_text(audio_path)
        
        # 5. 計算相關性
        correlations = None
        if not visual_df.empty and not audio_df.empty:
            correlations = self.calculate_correlations(visual_df, audio_df)
        
        # 6. 視覺化
        if not visual_df.empty and not audio_df.empty:
            self.visualize_results(visual_df, audio_df, text_emotion, correlations)
        
        # 7. 保存結果
        self.save_results(visual_df, audio_df, text_emotion, correlations)
        
        # 8. 打印摘要
        self.print_summary(visual_df, audio_df, correlations)
        
        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
    
    def print_summary(self, visual_df, audio_df, correlations):
        """打印分析摘要"""
        print("\n" + "=" * 80)
        print("📋 分析摘要（喜怒哀樂四種情緒）")
        print("=" * 80)
        
        emotion_zh = {'happy': '快樂', 'angry': '生氣', 'sad': '悲傷', 'neutral': '中性'}
        
        if not visual_df.empty:
            print("\n視覺情緒統計:")
            for emotion, count in visual_df['dominant_emotion'].value_counts().items():
                print(f"  {emotion_zh[emotion]} ({emotion}): {count} 次")
            print(f"\n最常見的視覺情緒: {emotion_zh[visual_df['dominant_emotion'].mode()[0]]} ({visual_df['dominant_emotion'].mode()[0]})")
        
        if not audio_df.empty:
            print("\n音訊情緒統計:")
            for emotion, count in audio_df['dominant_emotion'].value_counts().items():
                print(f"  {emotion_zh[emotion]} ({emotion}): {count} 次")
            print(f"\n最常見的音訊情緒: {emotion_zh[audio_df['dominant_emotion'].mode()[0]]} ({audio_df['dominant_emotion'].mode()[0]})")
        
        if correlations:
            print("\n視覺-音訊相關性:")
            for emotion, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                status = "✅" if corr > 0.5 else ("⚠️" if corr > 0.3 else "❌")
                print(f"  {emotion_zh[emotion]} ({emotion}): {corr:.3f} {status}")
            
            avg_corr = np.mean(list(correlations.values()))
            print(f"\n平均相關係數: {avg_corr:.3f}")
            
            if avg_corr > 0.5:
                print("✅ 視覺和音訊情緒高度一致")
            elif avg_corr > 0.3:
                print("⚠️ 視覺和音訊情緒中度一致")
            else:
                print("❌ 視覺和音訊情緒一致性較低")


def main():
    """主程式"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python multimodal_emotion_analysis.py <video_path> [text_file] [sample_rate]")
        print("\n範例:")
        print("  python multimodal_emotion_analysis.py my_vlog.mp4")
        print("  python multimodal_emotion_analysis.py my_vlog.mp4 transcript.txt")
        print("  python multimodal_emotion_analysis.py my_vlog.mp4 transcript.txt 2")
        print("\n參數說明:")
        print("  video_path: 影片路徑（必填）")
        print("  text_file: 文字檔案路徑（選填，如果有的話會直接使用而不進行語音轉錄）")
        print("  sample_rate: 每N秒分析一幀（選填，預設=1）")
        sys.exit(1)
    
    video_path = sys.argv[1]
    text_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith('.txt') else None
    sample_rate = float(sys.argv[-1]) if len(sys.argv) > 2 and not sys.argv[-1].endswith('.txt') else 1.0
    
    if not Path(video_path).exists():
        print(f"❌ 找不到影片檔案: {video_path}")
        sys.exit(1)
    
    if text_file and not Path(text_file).exists():
        print(f"❌ 找不到文字檔案: {text_file}")
        sys.exit(1)
    
    # 執行分析
    analyzer = MultiModalEmotionAnalyzer(video_path, text_file=text_file)
    analyzer.run_full_analysis(sample_rate=sample_rate)


if __name__ == "__main__":
    main()