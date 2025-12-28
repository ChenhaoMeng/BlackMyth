# 中国文化软实力的数字化反馈——基于《黑神话：悟空》Steam评论的中外情感倾向与文本挖掘对比分析

## 项目概述

本项目旨在分析《黑神话：悟空》在Steam平台上的玩家评论，对比中外玩家在情感倾向和关注焦点上的差异，以评估中国文化产品在国际传播中的效果。

## 项目目标

1. 分析中外玩家对《黑神话：悟空》的情感差异
2. 挖掘不同文化背景玩家的关注点差异
3. 评估中国游戏产品在国际市场的文化接受度
4. 探讨数字娱乐产品在文化输出中的作用

## 项目结构

```
/workspace/
├── BlackMythAnalysis.py                    # 原始分析代码
├── updated_analysis.py                     # 更新后的分析代码（使用配置和增强情感分析）
├── WangXiaoming_CS_BlackMythAnalysis.docx  # 项目报告
├── create_files.py                         # 数据文件创建脚本
├── create_doc.py                           # Word文档创建脚本
├── config.py                               # 配置文件
├── enhanced_sentiment_analyzer.py          # 增强版情感分析器
├── IMPROVEMENTS_RECOMMENDATIONS.md         # 改进建议文档
├── requirements.txt                        # 依赖包列表
├── result_data.xlsx                        # 分析结果数据
├── data/                                   # 数据目录
│   ├── raw_reviews.xlsx                    # 原始评论数据
│   └── stopwords.txt                       # 停用词表
├── deploy.sh                               # 部署脚本
└── DEPLOY_TO_GITHUB.md                     # GitHub部署指南
```

## 技术栈

- Python 3.x
- pandas - 数据处理
- numpy - 数值计算
- matplotlib/seaborn - 数据可视化
- jieba - 中文分词
- nltk - 英文文本处理
- snownlp - 中文情感分析
- scikit-learn - 机器学习算法
- wordcloud - 词云生成
- openpyxl - Excel文件处理
- python-docx - Word文档处理

## 功能特性

1. **数据爬取与预处理** - 模拟获取Steam评论数据并进行清洗
2. **中英文分词** - 使用jieba处理中文，nltk处理英文
3. **情感分析** - 对评论进行情感极性分析
4. **主题建模** - 使用LDA模型挖掘评论主题
5. **可视化分析** - 生成情感分布图、词云图等
6. **对比分析** - 中外玩家评论的对比分析

## 使用说明

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行原始分析程序：
   ```bash
   python BlackMythAnalysis.py
   ```

3. 或运行更新后的分析程序（推荐）：
   ```bash
   python updated_analysis.py
   ```

4. 查看结果：
   - 分析结果保存在 `result_data.xlsx`
   - 生成的图表保存在项目目录

## 部署到GitHub

请参考 `DEPLOY_TO_GITHUB.md` 文件中的详细说明。

## 项目意义

本项目从数字人文视角出发，探讨了流行文化产品（如3A游戏）作为中国文化出海载体的作用。通过量化分析中外玩家的情感差异和关注点，为"讲好中国故事"提供了数据支持和实践参考。

## 研究假设

1. 中文评论的情感均值高于英文评论，因为国内玩家带有民族自豪感
2. 英文评论更多关注"Performance"和"Difficulty"，中文评论更多关注"Story"和"Art"

## 数据来源

本项目使用模拟数据，实际应用中可替换为真实的Steam评论数据。