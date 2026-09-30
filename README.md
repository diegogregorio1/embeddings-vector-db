# Embeddings 与向量数据库

本项目是 **Embeddings（嵌入）与向量数据库（Vector Database）** 的学习代码集合，涵盖从传统词向量到现代语义嵌入、从经典相似度计算到向量检索的完整演进路线。

## 项目结构

```
embeddings-vector-db/
├── word2vec/                    # 模块 1：传统词向量（Word2Vec）
│   ├── word_seg.py              #   中文分词（西游记语料）
│   ├── word_similarity.py       #   Word2Vec 训练与词相似度计算
│   ├── requirements.txt
│   ├── utils/                   #   工具函数库
│   │   ├── segment.py           #     分词、去停用词、padding
│   │   ├── files_processing.py  #     文件读取、标签编码、批量划分
│   │   ├── create_word2vec.py   #     词向量转索引矩阵
│   │   └── create_batch_data.py #     批量数据生成器
│   ├── journey_to_the_west/     #   西游记语料（分词前后）
│   └── three_kingdoms/          #   三国演义语料
│
├── hotel_recommendation/        # 模块 2：基于 TF-IDF 的推荐
│   ├── hotel_rec.py             #   酒店推荐（TF-IDF + 余弦相似度）
│   ├── Seattle_Hotels.csv       #   西雅图酒店数据集
│   └── requirements.txt
│
├── CASE-向量数据库/              # 模块 3：FAISS 向量数据库入门
│   ├── 1-embedding计算.py       #   调用百炼 API 生成文本嵌入
│   ├── 2-embedding-faiss-元数据.py # FAISS 索引构建 + 向量检索 + 元数据
│   └── requirements.txt
│
├── Case-ChatPDF-Faiss/          # 模块 4：ChatPDF RAG 应用
│   ├── chatpdf-faiss.py         #   LangChain + FAISS + 百炼 API 文档问答
│   └── requirements.txt
│
└── .gitignore
```

## 模块详解

### 模块 1：Word2Vec 词向量

**核心思路**：通过无监督学习将词语映射到低维稠密向量空间，语义相近的词在向量空间中距离更近。

| 文件 | 功能 |
|------|------|
| `word_seg.py` | 使用 jieba 对西游记原著进行中文分词，输出到 `segment/` 目录 |
| `word_similarity.py` | 用 gensim 训练 Word2Vec 模型，演示词向量、词相似度、most_similar 运算 |
| `utils/segment.py` | 通用分词工具，支持 jieba 分词、去停用词、句子 padding |
| `utils/files_processing.py` | 文件遍历、标签编码、训练/验证集划分、批量读取 |
| `utils/create_word2vec.py` | 将词向量转为索引矩阵，支持训练好的模型加载与转换 |

**运行顺序**：先执行 `word_seg.py` 分词，再执行 `word_similarity.py` 训练模型。

**关键代码示例**（词相似度）：

```python
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)
print(model.wv.similarity('孙悟空', '猪八戒'))   # 两个词的余弦相似度
print(model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))  # 词类比
```

---

### 模块 2：TF-IDF 酒店推荐

**核心思路**：不依赖预训练模型，直接从文本提取 n-gram 特征，计算 TF-IDF 权重矩阵，再用余弦相似度做内容推荐。

| 文件 | 功能 |
|------|------|
| `hotel_rec.py` | 完整推荐流程：数据清洗 -> TF-IDF 特征提取 -> 相似度计算 -> Top-K 推荐 |
| `Seattle_Hotels.csv` | 西雅图酒店数据集（含酒店名称、描述等） |

**技术栈**：`scikit-learn` 的 `TfidfVectorizer` + `linear_kernel`

**推荐逻辑**：

```python
# 计算酒店描述的 TF-IDF 矩阵
tfidf_matrix = tf.fit_transform(df['desc_clean'])
# 计算酒店两两之间的余弦相似度
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
# 给定酒店名，返回相似度最高的 Top10
recommendations('Hilton Seattle Airport & Conference Center')
```

---

### 模块 3：FAISS 向量数据库入门

**核心思路**：调用阿里云百炼（DashScope）的 Embedding API 生成语义向量，用 FAISS 构建索引，实现带元数据的向量检索。

| 文件 | 功能 |
|------|------|
| `1-embedding计算.py` | 最小化示例：单条文本 -> Embedding API -> 1024 维向量 |
| `2-embedding-faiss-元数据.py` | 完整流程：文档集合 -> 批量向量化 -> FAISS IndexIDMap 索引构建 -> 相似度搜索 -> 元数据回显 |

**关键概念**：

| 概念 | 说明 |
|------|------|
| FAISS IndexFlatL2 | 暴力 L2 距离索引，精确但慢 |
| IndexIDMap | 包装器，支持自定义 ID（向量索引）-> 元数据映射 |
| text-embedding-v4 | 百炼的嵌入模型，支持 `dimensions` 参数自定义向量维度 |

**FAISS + 元数据关联原理**：

```
metadata_store = [doc1, doc2, doc3, doc4]      # 元数据列表，索引即 ID
index = faiss.IndexIDMap(faiss.IndexFlatL2(1024))
index.add_with_ids(vectors_np, ids_np)          # 向量 + 自定义 ID 入索引

distances, retrieved_ids = index.search(query_vec, k=3)
# retrieved_ids 返回的是我们自定义的 ID，可以直接用作 metadata_store 的下标
```

---

### 模块 4：ChatPDF RAG 应用

**核心思路**：经典 RAG（检索增强生成）流程 —— PDF 文本提取 -> 分块 -> Embedding -> 向量库存储 -> 相似度检索 -> LLM 问答。

| 文件 | 功能 |
|------|------|
| `chatpdf-faiss.py` | 完整 RAG 链路：PDF 读取 -> RecursiveCharacterTextSplitter 分块 -> DashScopeEmbeddings 嵌入 -> FAISS 存储 -> Tongyi/DeepSeek 问答链 |

**亮点特性**：

- **页码追踪**：每个文本块记录对应的 PDF 页码，问答后展示信息来源
- **持久化**：FAISS 向量库和页码信息可保存到磁盘，下次直接加载
- **成本统计**：用 `get_openai_callback` 跟踪 LLM API 调用费用

**RAG 流程**：

```
用户 PDF -> PyPDF2 提取文本 + 页码
    -> RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    -> DashScopeEmbeddings (text-embedding-v1)
    -> FAISS.from_texts -> 向量库

用户问题 -> 嵌入 -> 向量库相似度搜索 (k=2)
    -> load_qa_chain (Tongyi/DeepSeek, chain_type="stuff")
    -> 生成答案 + 来源页码
```

## 演进对比

| 维度 | Word2Vec | TF-IDF | Embedding API |
|------|----------|--------|---------------|
| 语义理解 | 词级别 | 词频统计，无语义 | 句/段落级别语义 |
| 依赖数据 | 需要语料训练 | 无模型 | 直接调用 API |
| 向量质量 | 较低 | 无向量（稀疏） | 高质量稠密向量 |
| 适用场景 | 词相似度 | 关键词匹配推荐 | 语义检索、RAG |
| 本项目应用 | 模块 1 | 模块 2 | 模块 3、4 |

## 环境要求

- Python >= 3.9
- 操作系统：Windows / macOS / Linux

## 安装依赖

各模块独立，按需安装：

```bash
# 模块 1：Word2Vec
cd word2vec
pip install -r requirements.txt

# 模块 2：酒店推荐
cd hotel_recommendation
pip install -r requirements.txt

# 模块 3：FAISS 向量数据库
cd "CASE-向量数据库"
pip install -r requirements.txt

# 模块 4：ChatPDF RAG
cd "Case-ChatPDF-Faiss"
pip install -r requirements.txt
```

## 配置环境变量

模块 3 和模块 4 需要阿里云百炼（DashScope）API Key：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"

# macOS / Linux
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

API Key 获取地址：https://bailian.console.aliyun.com/

## 运行示例

```bash
# 1. 词向量：西游记分词 + 训练 Word2Vec
cd word2vec
python word_seg.py       # 分词
python word_similarity.py # 训练并演示相似度

# 2. 酒店推荐（无外部依赖，直接运行）
cd hotel_recommendation
python hotel_rec.py

# 3. FAISS 入门（需 DashScope API Key）
cd "CASE-向量数据库"
python 1-embedding计算.py
python "2-embedding-faiss-元数据.py"

# 4. ChatPDF RAG（需 DashScope API Key）
cd "Case-ChatPDF-Faiss"
python chatpdf-faiss.py
```

## 依赖总览

| 模块 | 主要依赖 |
|------|----------|
| Word2Vec | `gensim`, `jieba`, `scikit-learn`, `pandas` |
| 酒店推荐 | `scikit-learn`, `pandas`, `matplotlib`, `nltk` |
| FAISS 入门 | `faiss_cpu`, `numpy`, `openai` |
| ChatPDF | `langchain`, `langchain_community`, `PyPDF2` |

## 说明

- PDF 文件未纳入版本控制（见 `.gitignore`），运行模块 4 时请自行提供 PDF
- `__pycache__`、`.ipynb_checkpoints`、模型文件等已自动排除
- 代码注释均为中文，使用 UTF-8 编码
