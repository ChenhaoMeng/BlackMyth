"""
黑神話：悟空 - 中外輿論對比分析系統配置文件
支持計算社會科學研究需求
Updated: 增加目標抓取數量以獲取更多樣本
"""
import os
import torch

# 自動檢測設備
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"配置文件已加載，檢測到運行設備: {DEVICE}")

CONFIG = {
    # ---------------------------------------------------------
    # 本地數據設置
    # 如果 data 目錄下有此文件，優先讀取，不進行爬蟲
    # 如果你想強制重新爬取更多數據，請刪除 data/raw_reviews.csv 或修改此路徑
    # ---------------------------------------------------------
    'local_data_path': 'raw_reviews.xlsx', 
    
    # 基礎設置 (爬蟲參數)
    'app_id': '2358720',  
    # [修改] 增加抓取數量：每個語種抓取 10000 條 (總計 20000)
    # Steam API 可能会有限制，能抓多少抓多少
    'review_count': 10000, 
    'languages': {
        'chinese': 'schinese',
        'english': 'english'
    },
    
    # NLP 模型設置
    'nlp': {
        'device': DEVICE,
        'batch_size': 16, # GPU 批處理大小
        'sentiment_model': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'zero_shot_model': 'facebook/bart-large-mnli',
        'absa_aspects': [
            'Cultural Heritage (文化遺產)', 
            'Visuals & Art (美術表現)', 
            'Gameplay & Difficulty (玩法難度)', 
            'Narrative & Lore (敘事與世界觀)', 
            'National Image (國家形象)',
            'Technical Issues (技術優化)'
        ],
        'topic_range': (5, 30)
    },
    
    # 數據去重設置
    'deduplication': {
        'method': 'tfidf_cosine',
        'threshold': 0.85,
    },
    
    # 統計檢驗
    'stats': {
        'alpha': 0.05,
        'test_method': 'mann-whitney'
    },
    
    # 輸出路徑
    'output_paths': {
        'raw_data': 'data/raw_reviews.csv', 
        'result_xlsx': 'output/analysis_results.xlsx',
        'semantic_network': 'output/semantic_network.png',
        'topic_model': 'output/bertopic_model'
    }
}