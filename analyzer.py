"""
基於 BERT 與 BERTopic 的高級文本分析器
實現 ABSA 與語義網絡挖掘
"""
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ResearchTextAnalyzer:
    def __init__(self, config):
        self.config = config
        self.device = 0 if config['nlp']['device'] == 'cuda' else -1
        

        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model=config['nlp']['sentiment_model'],
            device=self.device
        )
        
        self.absa_pipe = pipeline(
            "zero-shot-classification",
            model=config['nlp']['zero_shot_model'],
            device=self.device
        )
        
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def deduplicate_reviews(self, df):
        """基於 TF-IDF + 余弦相似度過濾重複文本"""
        logger.info("正在執行文本去重...")
        if len(df) < 2: return df
        
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df['review'].fillna(''))
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if not mask[i]: continue

            similar_indices = np.where(sim_matrix[i] > self.config['deduplication']['threshold'])[0]
            for j in similar_indices:
                if i != j:
                    mask[j] = False
        
        cleaned_df = df[mask].copy()
        logger.info(f"去重完成：過濾了 {len(df) - len(cleaned_df)} 條高度相似評論")
        return cleaned_df

    def analyze_sentiment_and_aspects(self, df):
        """執行 BERT 情感分析與 ABSA 維度提取"""
        logger.info("開始執行細粒度分析 (BERT + ABSA)...")
        
        results = []
        aspects = self.config['nlp']['absa_aspects']
        
        for idx, row in df.iterrows():
            text = str(row['review'])[:512]
            
            sent = self.sentiment_pipe(text)[0]
            star = int(sent['label'].split()[0])
            sent_score = (star - 3) / 2
            
            absa = self.absa_pipe(text, candidate_labels=aspects)
            top_aspect = absa['labels'][0]
            
            results.append({
                'sentiment_score': sent_score,
                'main_aspect': top_aspect,
                'aspect_confidence': absa['scores'][0]
            })
            
        res_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), res_df], axis=1)

    def run_bertopic_analysis(self, df):
        """執行 BERTopic 主題建模"""
        logger.info("啟動 BERTopic 主題建模...")
        docs = df['review'].astype(str).tolist()
        
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics="auto",
            calculate_probabilities=False,
            verbose=True
        )
        
        topics, _ = topic_model.fit_transform(docs)
        df['topic'] = topics
        info = topic_model.get_topic_info()
        return topic_model, df, info

    def perform_statistical_test(self, df):
        cn_scores = df[df['language'] == 'schinese']['sentiment_score']
        en_scores = df[df['language'] == 'english']['sentiment_score']
        
        u_stat, p_val = stats.mannwhitneyu(cn_scores, en_scores)
        
        summary = {
            'cn_mean': cn_scores.mean(),
            'en_mean': en_scores.mean(),
            'p_value': p_val,
            'is_significant': p_val < self.config['stats']['alpha']
        }
        return summary

    def build_semantic_network(self, df, lang='english'):

        import networkx as nx
        from sklearn.feature_extraction.text import CountVectorizer
        
        texts = df[df['language'] == lang]['review'].astype(str)
        cv = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=30)
        words_matrix = cv.fit_transform(texts)
        
        adj_matrix = (words_matrix.T * words_matrix)
        adj_matrix.setdiag(0)
        
        G = nx.from_scipy_sparse_array(adj_matrix)
        mapping = {i: word for i, word in enumerate(cv.get_feature_names_out())}
        G = nx.relabel_nodes(G, mapping)
        
        return G