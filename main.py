
import asyncio
import httpx
import pandas as pd
import logging
from config import CONFIG
from analyzer import ResearchTextAnalyzer
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainSystem")

class AsyncSteamScraper:

    def __init__(self):
        self.base_url = "https://store.steampowered.com/appreviews/"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def fetch_page(self, app_id, lang, cursor="*"):
        params = {
            'json': 1,
            'filter': 'all',
            'language': lang,
            'cursor': cursor,
            'review_type': 'all',
            'purchase_type': 'all',
            'num_per_page': 100
        }
        resp = await self.client.get(f"{self.base_url}{app_id}", params=params)
        return resp.json()

    async def get_all_reviews(self, app_id, lang, target_count=1000):
        reviews = []
        cursor = "*"
        while len(reviews) < target_count:
            data = await self.fetch_page(app_id, lang, cursor)
            if not data.get('reviews'): break
            
            for r in data['reviews']:
                reviews.append({
                    'language': lang,
                    'review': r['review'],
                    'voted_up': r['voted_up'],
                    'playtime': r['author'].get('playtime_forever', 0)
                })
            
            cursor = data['cursor']
            logger.info(f"已獲取 {lang} 評論: {len(reviews)}/{target_count}")
            if len(data['reviews']) < 100: break
            await asyncio.sleep(0.5) 
            
        return reviews

async def main():

    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)


    scraper = AsyncSteamScraper()
    all_data = []
    for lang_key, lang_code in CONFIG['languages'].items():
        logger.info(f"正在開始抓取語種: {lang_key}")
        lang_reviews = await scraper.get_all_reviews(CONFIG['app_id'], lang_code, CONFIG['review_count'])
        all_data.extend(lang_reviews)
    
    df = pd.DataFrame(all_data)
    df.to_csv(CONFIG['output_paths']['raw_data'], index=False)


    analyzer = ResearchTextAnalyzer(CONFIG)

    df = analyzer.deduplicate_reviews(df)


    sample_df = df.sample(min(len(df), 500)) 
    analyzed_df = analyzer.analyze_sentiment_and_aspects(sample_df)

    topic_model, final_df, topic_info = analyzer.run_bertopic_analysis(analyzed_df)
    logger.info(f"發現主題數量: {len(topic_info)}")

    stats_res = analyzer.perform_statistical_test(final_df)
    logger.info(f"統計檢驗結果: {stats_res}")

    final_df.to_excel(CONFIG['output_paths']['result_xlsx'], index=False)
    logger.info("分析流程全部完成，結果已保存。")

if __name__ == "__main__":
    asyncio.run(main())