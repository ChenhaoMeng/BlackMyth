"""
基於 BERT 與 BERTopic 的高級文本分析器
實現 ABSA 與語義網絡挖掘
Updated: 實現 GPU 批處理推理 (Batch Inference) 以最大化效率
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
import logging
from tqdm import tqdm  # 引入進度條

logger = logging.getLogger(__name__)

class ResearchTextAnalyzer:
    def __init__(self, config):
        self.config = config
        self.device = 0 if config['nlp']['device'] == 'cuda' and torch.cuda.is_available() else -1
        
        logger.info(f"正在加載模型 (Device={self.device})...")
        
        # 1. 初始化情感分析 (BERT)
        try:
            self.sentiment_pipe = pipeline(
                "sentiment-analysis", 
                model=config['nlp']['sentiment_model'],
                device=self.device
            )
        except Exception as e:
            logger.error(f"情感模型加載失敗: {e}")
            raise e
        
        # 2. 初始化維度分類 (Zero-shot for ABSA)
        try:
            self.absa_pipe = pipeline(
                "zero-shot-classification",
                model=config['nlp']['zero_shot_model'],
                device=self.device
            )
        except Exception as e:
            logger.error(f"Zero-shot 模型加載失敗: {e}")
            raise e
        
        # 3. 初始化主題嵌入模型
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def deduplicate_reviews(self, df):
        """基於 TF-IDF + 余弦相似度過濾重複文本"""
        logger.info("正在執行文本去重...")
        if len(df) < 2: return df
        
        clean_docs = df['review'].fillna('').astype(str)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_docs)
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
        except Exception as e:
            logger.warning(f"去重過程出錯 (可能是數據過少): {e}, 跳過去重步驟")
            return df

    def analyze_sentiment_and_aspects(self, df):
        """執行 BERT 情感分析與 ABSA 維度提取 (Batch Processing 模式)"""
        logger.info("開始執行細粒度分析 (BERT + ABSA) - 批處理模式...")
        
        aspects = self.config['nlp']['absa_aspects']
        batch_size = self.config['nlp'].get('batch_size', 16)
        
        # 1. 預處理數據：確保索引對齊且無髒數據
        df_clean = df.copy()
        df_clean['review'] = df_clean['review'].fillna('').astype(str)
        # 過濾空文本和純空白符
        df_clean = df_clean[df_clean['review'].str.strip().str.len() > 0].reset_index(drop=True)
        
        if df_clean.empty:
            logger.warning("沒有有效數據進行分析")
            return df
        
        # 準備輸入列表 (同時截斷以符合 BERT 限制)
        # 注意：雖然 pipeline 有 truncation 參數，但手動截斷可減少顯存佔用
        texts = [t[:512] for t in df_clean['review'].tolist()]
        total = len(texts)
        
        logger.info(f"準備對 {total} 條評論進行推理 (Batch Size={batch_size})...")

        # 容器
        sent_scores = []
        main_aspects = []
        aspect_confs = []

        # 2. 批量執行情感分析
        logger.info("Step 1/2: 正在執行情感分析 (Sentiment Analysis)...")
        try:
            # 這裡我們傳入列表，pipeline 會自動處理為 Dataset 進行 batching
            for out in tqdm(self.sentiment_pipe(texts, batch_size=batch_size, truncation=True), total=total):
                # out: {'label': '5 stars', 'score': 0.9}
                star = int(out['label'].split()[0])
                score = (star - 3) / 2
                sent_scores.append(score)
        except Exception as e:
            logger.error(f"情感分析推理失敗: {e}")
            raise e

        # 3. 批量執行 ABSA (Zero-shot)
        logger.info("Step 2/2: 正在執行維度分類 (Zero-Shot Classification)...")
        try:
            # Zero-shot pipeline 接受 candidate_labels 參數
            for out in tqdm(self.absa_pipe(texts, candidate_labels=aspects, batch_size=batch_size, truncation=True), total=total):
                # out: {'sequence': '...', 'labels': ['Visuals', ...], 'scores': [0.99, ...]}
                main_aspects.append(out['labels'][0])
                aspect_confs.append(out['scores'][0])
        except Exception as e:
            logger.error(f"維度分類推理失敗: {e}")
            raise e

        # 4. 整合結果
        # 由於我們重置了索引且順序處理，可以直接賦值
        df_clean['sentiment_score'] = sent_scores
        df_clean['main_aspect'] = main_aspects
        df_clean['aspect_confidence'] = aspect_confs
        
        return df_clean

    def run_bertopic_analysis(self, df):
        """執行 BERTopic 主題建模"""
        logger.info("啟動 BERTopic 主題建模...")
        
        df = df[df['review'].str.strip().astype(bool)].copy()
        docs = df['review'].astype(str).tolist()
        
        if not docs:
            logger.warning("沒有足夠的文本進行主題建模")
            return None, df, None

        try:
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
        except Exception as e:
            logger.error(f"BERTopic 建模失敗: {e}")
            return None, df, None

    def perform_statistical_test(self, df):
        """執行統計顯著性檢驗"""
        try:
            cn_scores = df[df['language'] == 'schinese']['sentiment_score'].dropna()
            en_scores = df[df['language'] == 'english']['sentiment_score'].dropna()
            
            if len(cn_scores) < 5 or len(en_scores) < 5:
                return {"error": "樣本量不足以進行統計檢驗"}

            u_stat, p_val = stats.mannwhitneyu(cn_scores, en_scores)
            
            summary = {
                'cn_mean': cn_scores.mean(),
                'en_mean': en_scores.mean(),
                'p_value': p_val,
                'is_significant': p_val < self.config['stats']['alpha']
            }
            return summary
        except Exception as e:
            logger.error(f"統計檢驗失敗: {e}")
            return {"error": str(e)}

    def build_semantic_network(self, df, lang='english'):
        """構建語義共現網絡"""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            texts = df[df['language'] == lang]['review'].astype(str)
            if len(texts) < 10:
                return None

            cv = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=30)
            words_matrix = cv.fit_transform(texts)
            
            adj_matrix = (words_matrix.T * words_matrix)
            adj_matrix.setdiag(0)
            
            G = nx.from_scipy_sparse_array(adj_matrix)
            mapping = {i: word for i, word in enumerate(cv.get_feature_names_out())}
            G = nx.relabel_nodes(G, mapping)
            
            return G
        except Exception as e:
            logger.error(f"構建語義網絡失敗: {e}")
            return None