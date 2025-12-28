
import os

CONFIG = {

    'app_id': '2358720', 
    'review_count': 5000,   
    'languages': {
        'chinese': 'schinese',
        'english': 'english'
    },
    

    'deduplication': {
        'method': 'tfidf_cosine',
        'threshold': 0.85, 
    },
    
    'nlp': {
        'device': 'cpu', 
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
    
 
    'stats': {
        'alpha': 0.05,
        'test_method': 'mann-whitney' 
    },

    
    'output_paths': {
        'raw_data': 'data/raw_reviews.csv',
        'cleaned_data': 'data/cleaned_reviews.csv',
        'result_xlsx': 'output/analysis_results.xlsx',
        'semantic_network': 'output/semantic_network.png',
        'topic_model': 'output/bertopic_model'
    }
}