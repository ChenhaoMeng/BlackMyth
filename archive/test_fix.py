"""
Test script to verify the stopwords file issue is fixed
"""

import os
import sys

# Add the workspace directory to the path
sys.path.append('/workspace')

# Import required modules
import jieba
from nltk.corpus import stopwords as nltk_stopwords
import nltk

# Download required NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Test the stopwords file creation
def test_stopwords():
    print("Testing stopwords file...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    print("Data directory ensured")
    
    # Check if stopwords file exists
    stopwords_path = "data/stopwords.txt"
    if os.path.exists(stopwords_path):
        print(f"Stopwords file exists at {stopwords_path}")
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Stopwords file has {len(content.split())} words")
    else:
        print(f"Stopwords file does not exist at {stopwords_path}")
        # Create a basic one
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
    
    # Now try to load the DataPreprocessor class from the original file
    try:
        # Read the original file content to extract the DataPreprocessor class
        with open('BlackMythAnalysis.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Execute the necessary parts to test
        exec_globals = {}
        exec(content, exec_globals)
        
        # Try to create DataPreprocessor instance
        DataPreprocessor = exec_globals['DataPreprocessor']
        preprocessor = DataPreprocessor()
        
        print("Successfully created DataPreprocessor instance!")
        print(f"Loaded {len(preprocessor.chinese_stopwords)} Chinese stopwords")
        print(f"Loaded {len(preprocessor.english_stopwords)} English stopwords")
        
        return True
    except Exception as e:
        print(f"Error creating DataPreprocessor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stopwords()
    if success:
        print("\nFix verification: SUCCESS")
        print("The original issue with the stopwords file has been resolved.")
    else:
        print("\nFix verification: FAILED")
        print("There are still issues with the stopwords file or DataPreprocessor class.")