#!/bin/bash

# 中国文化软实力数字化反馈项目 - 部署脚本

echo "中国文化软实力数字化反馈项目部署脚本"
echo "====================================="

# 检查是否在正确的目录
if [ ! -f "WangXiaoming_CS_BlackMythAnalysis.py" ]; then
    echo "错误: 未在项目根目录中找到主要文件"
    echo "请确保在 /workspace 目录下运行此脚本"
    exit 1
fi

echo "项目文件检查通过"
echo

# 检查Git是否已初始化
if [ ! -d ".git" ]; then
    echo "初始化Git仓库..."
    git init
    if [ $? -ne 0 ]; then
        echo "错误: 无法初始化Git仓库"
        exit 1
    fi
fi

echo "添加所有文件到Git..."
git add .

# 检查是否有待提交的更改
if git diff --cached --quiet; then
    echo "没有文件需要提交"
else
    echo "提交更改..."
    git config --global user.email "example@example.com"
    git config --global user.name "Example User"
    git commit -m "Initial commit: 中国文化软实力的数字化反馈分析项目"
    if [ $? -eq 0 ]; then
        echo "提交成功"
    else
        echo "提交失败"
        exit 1
    fi
fi

echo
echo "项目已准备就绪，接下来请按以下步骤操作："
echo
echo "1. 登录GitHub账户"
echo "2. 创建一个新的仓库（例如：chinese-culture-digital-analysis）"
echo "3. 不要初始化README、.gitignore或license文件"
echo
echo "4. 然后运行以下命令（替换您的仓库URL）："
echo "   git remote add origin https://github.com/<your-username>/<your-repo-name>.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "或者，如果您使用SSH密钥："
echo "   git remote add origin git@github.com:<your-username>/<your-repo-name>.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "注意：如果您想使用Personal Access Token进行身份验证，"
echo "请使用以下格式的URL："
echo "   https://<token>@github.com/<your-username>/<your-repo-name>.git"
echo