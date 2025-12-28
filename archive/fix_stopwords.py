"""
Script to fix the stopwords file issue in the Black Myth: Wukong analysis script
"""

import os

def create_stopwords_file():
    """
    Create the stopwords file if it doesn't exist
    """
    stopwords_path = "data/stopwords.txt"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(stopwords_path), exist_ok=True)
    
    # Check if the file already exists
    if not os.path.exists(stopwords_path):
        # Create a basic Chinese stopwords list
        chinese_stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还是', '为了', '都', '把', '这个', '什么',
            '非常', '可以', '这个', '还是', '然后', '但是', '因为', '所以', '如果', '虽然',
            '真的', '很', '太', '了', '吧', '啊', '呢', '吗', '哈', '哦', '嗯', '额', '嘿'
        ]
        
        with open(stopwords_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chinese_stopwords))
        
        print(f"Created stopwords file at {stopwords_path}")
    else:
        print(f"Stopwords file already exists at {stopwords_path}")

if __name__ == "__main__":
    create_stopwords_file()
    print("Stopwords file issue fixed!")