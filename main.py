"""
中外輿論對比分析系統 - 主程序
集成異步爬蟲、深度學習分析與學術統計
Updated: 增加日誌文件保存 (output/runtime.log)
"""
import asyncio
import httpx
import pandas as pd
import logging
from config import CONFIG
from analyzer import ResearchTextAnalyzer
import os

# ---------------------------------------------------------
# [修改] 提前創建目錄並配置日誌保存到文件
# ---------------------------------------------------------
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

# 配置日誌：同時輸出到控制台和文件
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/runtime.log", encoding='utf-8', mode='w'), # 寫入文件
        logging.StreamHandler() # 輸出到屏幕
    ]
)
logger = logging.getLogger("MainSystem")

class AsyncSteamScraper:
    """異步 Steam 評論抓取器"""
    def __init__(self):
        self.base_url = "https://store.steampowered.com/appreviews/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.client = httpx.AsyncClient(timeout=30.0, headers=self.headers)

    async def fetch_page(self, app_id, lang, cursor="*"):
        params = {
            'json': 1, 'filter': 'all', 'language': lang, 'cursor': cursor,
            'review_type': 'all', 'purchase_type': 'all', 'num_per_page': 100
        }
        try:
            resp = await self.client.get(f"{self.base_url}{app_id}", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"請求失敗 ({lang}): {e}")
            return None

    async def get_all_reviews(self, app_id, lang, target_count=1000):
        reviews = []
        cursor = "*"
        retry_count = 0
        while len(reviews) < target_count:
            data = await self.fetch_page(app_id, lang, cursor)
            if not data or not data.get('reviews'):
                if retry_count < 3:
                    retry_count += 1
                    await asyncio.sleep(2)
                    continue
                else:
                    break
            retry_count = 0
            for r in data['reviews']:
                reviews.append({
                    'language': lang,
                    'review': r['review'],
                    'voted_up': r['voted_up'],
                    'playtime': r['author'].get('playtime_forever', 0)
                })
            cursor = data.get('cursor', '*')
            logger.info(f"已獲取 {lang} 評論: {len(reviews)}/{target_count}")
            if len(data['reviews']) < 100: break
            await asyncio.sleep(0.5)
        return reviews

    async def close(self):
        await self.client.aclose()

def normalize_columns(df):
    """標準化列名"""
    col_mapping = {
        'content': 'review', 'text': 'review', 'comment': 'review', '評論': 'review', '内容': 'review',
        'lang': 'language', 'language_code': 'language', '语种': 'language', '语言': 'language'
    }
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    df = df.rename(columns=col_mapping)
    
    if 'review' not in df.columns:
        logger.warning("未找到標準 'review' 列，嘗試自動識別文本列...")
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.info(f"將列 '{col}' 視為評論文本列")
                df = df.rename(columns={col: 'review'})
                break
    
    if 'language' not in df.columns:
        logger.warning("未找到 'language' 列，將默認設為 'unknown'")
        df['language'] = 'unknown'
        
    return df

async def main():
    df = None
    local_path = CONFIG.get('local_data_path', '')
    raw_csv_path = CONFIG['output_paths']['raw_data']
    
    # ---------------------------------------------------------
    # 1. 數據加載模塊
    # ---------------------------------------------------------
    if local_path and os.path.exists(local_path):
        logger.info(f"檢測到用戶指定的本地數據: {local_path}")
        try:
            if local_path.endswith('.xlsx') or local_path.endswith('.xls'):
                df = pd.read_excel(local_path)
            elif local_path.endswith('.csv'):
                df = pd.read_csv(local_path)
            
            if df is not None and not df.empty:
                df = normalize_columns(df)
                logger.info(f"成功加載本地數據: {len(df)} 條")
            else:
                logger.warning("本地數據文件為空或讀取失敗")
                df = None
        except Exception as e:
            logger.error(f"加載本地文件失敗: {e}")
            df = None

    if df is None and os.path.exists(raw_csv_path):
        logger.info(f"檢測到爬蟲備份數據: {raw_csv_path}")
        try:
            df = pd.read_csv(raw_csv_path)
            if df.empty:
                logger.warning(f"本地 CSV 為空，將重新抓取。")
                df = None
            else:
                if 'review' not in df.columns: df = normalize_columns(df)
                df['review'] = df['review'].fillna('').astype(str)
                logger.info(f"從 CSV 加載 {len(df)} 條數據")
        except Exception:
            df = None

    if df is None:
        logger.info("未找到有效本地數據，開始執行異步爬蟲...")
        scraper = AsyncSteamScraper()
        all_data = []
        try:
            for lang_key, lang_code in CONFIG['languages'].items():
                logger.info(f"正在開始抓取語種: {lang_key}")
                lang_reviews = await scraper.get_all_reviews(CONFIG['app_id'], lang_code, CONFIG['review_count'])
                all_data.extend(lang_reviews)
            
            df = pd.DataFrame(all_data)
            df.to_csv(raw_csv_path, index=False)
            logger.info(f"抓取完成，數據已保存")
        finally:
            await scraper.close()

    if df is None or df.empty:
        logger.error("沒有數據可供分析，程序退出。")
        return

    # ---------------------------------------------------------
    # 1.5 數據概覽與詳細預處理統計
    # ---------------------------------------------------------
    logger.info("="*50)
    logger.info("DATA HEALTH CHECK (數據概覽)")
    logger.info("="*50)
    
    total_raw = len(df)
    logger.info(f"原始數據總條數: {total_raw}")
    
    # 1. 處理空值
    df['review'] = df['review'].fillna('').astype(str)
    
    # 2. 統計空評論
    empty_mask = df['review'].str.strip() == ""
    empty_count = empty_mask.sum()
    logger.info(f"發現空評論 (Empty/NaN): {empty_count} 條")
    
    # 3. 過濾無效評論
    df_clean = df[~empty_mask].copy()
    
    # 4. 統計語種分佈
    if 'language' in df_clean.columns:
        logger.info("\n--- 語種分佈 (Top 5) ---")
        lang_stats = df_clean['language'].value_counts().head(5)
        for lang, count in lang_stats.items():
            logger.info(f"{lang}: {count}")
    else:
        logger.warning("警告: 數據中未找到 'language' 列，無法統計語種。")

    # 5. 統計評論長度與進一步篩選
    df_clean['length'] = df_clean['review'].str.len()
    avg_len = df_clean['length'].mean()
    logger.info(f"\n平均評論長度: {avg_len:.2f} 字符")
    
    min_length = 2
    short_reviews = df_clean[df_clean['length'] <= min_length]
    logger.info(f"過短評論 (<= {min_length} 字): {len(short_reviews)} 條 (將被過濾)")
    
    # 應用過濾
    df = df_clean[df_clean['length'] > min_length].drop(columns=['length'])
    final_count = len(df)
    
    logger.info("-" * 30)
    logger.info(f"有效數據總數: {final_count} (過濾率: {((total_raw - final_count)/total_raw)*100:.2f}%)")
    logger.info("="*50 + "\n")

    if df.empty:
        logger.error("過濾後無有效數據，程序退出。")
        return

    # ---------------------------------------------------------
    # 3. 深度分析
    # ---------------------------------------------------------
    analyzer = ResearchTextAnalyzer(CONFIG)
    
    # 去重
    df = analyzer.deduplicate_reviews(df)

    logger.info(f"正在對 {len(df)} 條數據進行深度分析...")
    target_df = df.copy()

    # 執行 Batch 分析
    analyzed_df = analyzer.analyze_sentiment_and_aspects(target_df)

    # Checkpoint
    try:
        analyzed_df.to_excel("output/checkpoint_sentiment_absa.xlsx", index=False)
        logger.info("中間結果已保存至 checkpoint")
    except Exception as e:
        logger.warning(f"Checkpoint 保存失敗: {e}")

    # ---------------------------------------------------------
    # 4. 主題建模
    # ---------------------------------------------------------
    topic_model, final_df, topic_info = analyzer.run_bertopic_analysis(analyzed_df)
    
    if topic_info is not None:
        logger.info(f"BERTopic 分析完成，發現主題數量: {len(topic_info)}")
        try:
            fig = topic_model.visualize_topics()
            fig.write_html("output/topics_visualization.html")
        except Exception:
            pass

    # ---------------------------------------------------------
    # 5. 統計與保存
    # ---------------------------------------------------------
    if 'language' in final_df.columns:
        stats_res = analyzer.perform_statistical_test(final_df)
        logger.info(f"統計檢驗結果: {stats_res}")
    
    final_df.to_excel(CONFIG['output_paths']['result_xlsx'], index=False)
    logger.info(f"分析完成，結果已保存至 {CONFIG['output_paths']['result_xlsx']}")
    logger.info(f"完整運行日誌已保存至 output/runtime.log")

if __name__ == "__main__":
    asyncio.run(main())