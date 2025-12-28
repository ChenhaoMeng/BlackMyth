"""
Create the required files for the Black Myth: Wukong analysis project
"""

import pandas as pd
import numpy as np

# Create a simple mock dataset
data = {
    'review_id': [f'cn_{i}' for i in range(5)] + [f'en_{i}' for i in range(5)],
    'language': ['Chinese'] * 5 + ['English'] * 5,
    'review_text': [
        '游戏画面非常精美，中国风浓厚，很棒！',
        '画面太棒了，剧情也很吸引人',
        '优化有点问题，但整体很棒',
        '美术风格真的太美了',
        '游戏性不错，但有些地方卡顿',
        'Amazing graphics and unique art style, really impressed!',
        'Great game with beautiful visuals but needs optimization.',
        'The story is engaging and the world is well designed.',
        'Solid gameplay, though a bit challenging at times.',
        'Beautiful art style inspired by Chinese mythology.'
    ],
    'voted_up': [True, True, True, False, True, True, False, True, True, True],
    'timestamp_created': [1690000000 + i * 1000 for i in range(10)],
    'playtime_forever': [45, 32, 28, 15, 60, 55, 12, 40, 35, 50],
    'cleaned_text': [
        '游戏画面非常精美 中国风浓厚 很棒',
        '画面太棒了 剧情也很吸引人',
        '优化有点问题 但整体很棒',
        '美术风格真的太美了',
        '游戏性不错 但有些地方卡顿',
        'Amazing graphics and unique art style really impressed',
        'Great game with beautiful visuals but needs optimization',
        'The story is engaging and the world is well designed',
        'Solid gameplay though a bit challenging at times',
        'Beautiful art style inspired by Chinese mythology'
    ],
    'tokens': [
        '游戏 画面 非常 精美 中国风 浓厚 很棒',
        '画面 太棒 剧情 吸引人',
        '优化 问题 整体 很棒',
        '美术 风格 真的 美',
        '游戏性 不错 地方 卡顿',
        'amazing graphic unique art style really impress',
        'great game beautiful visual need optimization',
        'story engaging world well design',
        'solid gameplay though bit challenging time',
        'beautiful art style inspire chinese mythology'
    ],
    'sentiment_score': [0.8, 0.7, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.5, 0.6]
}

df = pd.DataFrame(data)

# Save the mock dataset
df.to_excel('data/raw_reviews.xlsx', index=False)
df.to_excel('result_data.xlsx', index=False)

print("Files created successfully:")
print("- data/raw_reviews.xlsx")
print("- result_data.xlsx")