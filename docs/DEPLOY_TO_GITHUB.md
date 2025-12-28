# 部署到GitHub指南

要将此项目部署到GitHub，请按照以下步骤操作：

## 方法1：使用命令行（推荐）

### 1. 创建GitHub仓库
1. 登录GitHub账户
2. 点击"New repository"按钮
3. 输入仓库名称，例如：`chinese-culture-digital-analysis`
4. 选择"Public"或"Private"
5. 不要初始化README、.gitignore或license文件
6. 点击"Create repository"

### 2. 获取Personal Access Token（如果需要）
1. 进入GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. 点击"Generate new token"
3. 选择适当的权限（至少需要repo权限）
4. 复制生成的token

### 3. 在本地执行以下命令
```bash
cd /workspace

# 添加远程仓库（替换<your-username>和<your-repo-name>）
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 设置用户名和邮箱
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"

# 推送到GitHub
git branch -M main
git push -u origin main
```

## 方法2：使用GitHub CLI（如果已安装）

```bash
# 在/workspace目录中
gh repo create chinese-culture-digital-analysis --public --push
```

## 项目文件说明

本项目包含以下重要文件：

1. **主要代码文件：**
   - `WangXiaoming_CS_BlackMythAnalysis.py` - 完整的分析代码实现
   - `create_files.py` - 创建数据文件的脚本
   - `create_doc.py` - 创建Word文档的脚本

2. **数据文件：**
   - `/data/raw_reviews.xlsx` - 原始评论数据
   - `/data/stopwords.txt` - 中英文停用词表
   - `result_data.xlsx` - 分析结果数据

3. **文档文件：**
   - `WangXiaoming_CS_BlackMythAnalysis.docx` - 项目分析报告

4. **其他文件：**
   - `requirements.txt` - Python依赖包列表
   - `main.py` - 主程序入口文件

## 项目结构

```
/workspace/
├── WangXiaoming_CS_BlackMythAnalysis.docx
├── WangXiaoming_CS_BlackMythAnalysis.py
├── WangXiaoming_CS_BlackMythAnalysis_Final.py
├── WangXiaoming_CS_BlackMythAnalysis_Minimal.py
├── WangXiaoming_CS_BlackMythAnalysis_Mock.py
├── create_doc.py
├── create_files.py
├── main.py
├── requirements.txt
├── result_data.xlsx
├── data/
│   ├── raw_reviews.xlsx
│   └── stopwords.txt
└── DEPLOY_TO_GITHUB.md
```

## 项目描述

这是一个关于中国文化软实力数字化反馈的研究项目，基于《黑神话：悟空》Steam评论的中外情感倾向与文本挖掘对比分析。项目实现了完整的文本挖掘流程，包括数据预处理、中英文分词、情感分析、结果可视化等功能。

## 依赖项

项目依赖以下Python包：
- pandas
- numpy
- matplotlib
- seaborn
- jieba
- nltk
- snownlp
- wordcloud
- scikit-learn
- openpyxl
- python-docx

可以通过运行以下命令安装：
```bash
pip install -r requirements.txt
```