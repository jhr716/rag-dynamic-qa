📝 README.md 完整内容（可直接复制）
markdown
# 📚 动态RAG问答系统

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/ollama-deepseek--r1-orange.svg)](https://ollama.ai/)

## 📖 项目简介

这是一个基于本地大模型的智能问答系统，能根据问题类型自动调整检索策略。用户上传知识库文档后，系统可以基于文档内容回答问题，所有回答均可溯源验证。

### ✨ 核心特性

- **动态策略**：根据问题类型自动切换参数
  - 定义型问题（如"什么是X"）：chunk_overlap=50，精准定位
  - 对比型问题（如"X和Y的区别"）：chunk_overlap=75, k=4，覆盖更广
- **完全本地**：基于Ollama部署，数据不外传，保护隐私
- **可解释性**：展示答案来源，支持溯源验证
- **双交互模式**：命令行调试 + Web界面展示

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.11 | 核心开发语言 |
| 大模型 | Ollama (deepseek-r1:7b) | 本地部署的LLM |
| 嵌入模型 | nomic-embed-text | 文本向量化 |
| RAG框架 | LangChain + ChromaDB | 检索增强生成 |
| 前端 | Gradio | Web交互界面 |

## 📊 实验优化

### chunk_size对比实验
| size | 定义型 | 对比型 | 原理型 | 结论 |
|------|--------|--------|--------|------|
| 200 | 准但碎片 | ❌ 失败 | 部分正确 | 不稳定 |
| 500 | ✅ 准确 | ❌ 失败 | ✅ 完整 | 最平衡 |
| 800 | ✅ 准确 | ✅ 成功 | ❌ 错误 | 有风险 |

**关键发现**：800虽然平均分高，但在ResNet问题上给出错误答案，而500诚实地回答"无法回答"。这说明RAG系统的第一原则是"不胡编"。

### chunk_overlap调优
| overlap | 定义型 | 对比型 | 适用场景 |
|---------|--------|--------|----------|
| 50 | ✅ | ❌ | 精准定位 |
| 75 | ❌ | ✅ | 跨段对比 |

**创新点**：根据问题类型动态调整参数，平衡准确性和完整性。

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Ollama（安装并运行）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/jhr716/rag-dynamic-qa.git
cd rag-dynamic-qa

# 2. 创建虚拟环境
conda create -n rag python=3.11
conda activate rag

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
运行
命令行模式
bash
python rag_dynamic.py
Web界面模式
bash
python app.py
访问 http://127.0.0.1:7860

📝 使用示例
bash
📝 问题: 什么是深度学习？
💡 答案: 深度学习是机器学习的一个分支，通过构建多层神经网络来学习数据的多层次特征表示...

📝 问题: LSTM和GRU有什么区别？
💡 答案: LSTM有三个门（输入门、遗忘门、输出门），GRU只有两个门（更新门、重置门）...
📁 项目结构
text
rag-dynamic-qa/
├── rag_dynamic.py      # 核心RAG系统
├── app.py              # Web界面
├── summary.txt         # 知识库文档
├── requirements.txt    # 依赖包
├── .gitignore          # Git忽略文件
└── README.md           # 项目说明
🔧 常见问题
Q: 第一次运行很慢怎么办？
A: 第一次需要下载模型和创建向量数据库，后续启动会很快。

Q: 如何更换自己的知识库？
A: 把 summary.txt 替换成你自己的文档，重新运行即可。

Q: Web界面无法访问？
A: 确保先安装Gradio：pip install gradio

📌 待优化方向
支持多文件上传（PDF/Word/TXT）

添加对话历史功能

用小型分类模型替代关键词匹配

混合检索策略（同时用多种参数）

📄 许可证
MIT License © 2024 JHR716