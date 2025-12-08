# ChartMirage

[中文](#chartmirage-中文) | [English](#chartmirage-english)

---

## ChartMirage (中文)

ChartMirage 是一个专注于评估多模态大语言模型（MLLMs）在图表理解方面鲁棒性的项目。本项目采用先进的**多模态 RAG (Retrieval-Augmented Generation)** 架构，旨在探索模型在面对对抗性或复杂图表数据时的检索与推理能力。

### 核心架构 (RAG Framework)

本项目基于 **LlamaIndex** 构建多模态 RAG 流程，底层原理分为三个核心阶段：**索引（Indexing）**、**检索（Retrieval）** 和 **生成（Generation）**。

| 组件 | 实现技术/模型 | 作用 |
| :--- | :--- | :--- |
| **框架 (Framework)** | `LlamaIndex` | 负责编排整个 RAG 流程（数据加载、索引构建、检索、提示词组装）。 |
| **索引结构 (Index)** | `MultiModalVectorStoreIndex` | 专门用于存储和管理多模态数据（图片+文本）的向量索引。 |
| **嵌入模型 (Embedding)** | `ClipEmbedding` (ViT-L/14) | **检索核心**。将图片和文本映射到同一个高维向量空间，通过计算 Query 向量与图片向量的余弦相似度来实现语义检索。 |
| **生成模型 (LLM)** | `Qwen3-VL-Plus` | 多模态大模型。它不仅能处理文本，还能直接“看”懂检索到的图片（Visual Token），并结合上下文回答用户问题。 |

#### 工作流原理

1.  **索引构建**: 使用 `SimpleDirectoryReader` 加载数据集中的图表，通过 **CLIP** 视觉编码器将图片转换为向量并存储。
2.  **语义检索**: 当用户提出问题时，系统将文本 Query 转换为向量，在库中检索出语义最相关的图表（Top-K）。
3.  **多模态生成**: 将用户的文本问题与检索到的图表一起输入给 **Qwen-VL**，模型结合视觉信息与文本指令生成最终回答。

### 快速开始 (Quick Start)

#### 1. 环境准备

克隆代码仓库并安装依赖。建议使用 Conda 管理环境。

```bash
# 克隆项目
git clone https://github.com/your-username/ChartMirage.git
cd ChartMirage

# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate ChartMirage
```

#### 2. 配置 API Key

项目依赖 OpenAI 格式的 API 接口（如 Qwen-VL 通过 DashScope 提供的兼容接口）。请确保在环境变量中设置了 `OPENAI_API_KEY`。

#### 3. 运行 RAG 演示

使用提供的测试脚本运行多模态 RAG 流程：

```bash
python test.py
```

该脚本将演示如何加载图表数据、构建多模态索引，并针对特定问题（如销售趋势）进行检索和问答。

---

## ChartMirage (English)

ChartMirage is a project dedicated to evaluating the robustness of Multi-modal Large Language Models (MLLMs) in chart understanding. It utilizes an advanced **Multi-modal RAG (Retrieval-Augmented Generation)** architecture to explore model performance in retrieval and reasoning tasks when facing adversarial or complex chart data.

### Architecture (RAG Framework)

The project is built upon **LlamaIndex** to orchestrate the Multi-modal RAG pipeline, consisting of three core stages: **Indexing**, **Retrieval**, and **Generation**.

| Component | Implementation/Model | Function |
| :--- | :--- | :--- |
| **Framework** | `LlamaIndex` | Orchestrates the RAG pipeline (data loading, indexing, retrieval, prompt assembly). |
| **Index Structure** | `MultiModalVectorStoreIndex` | Vector index designed for storing and managing multi-modal data (images + text). |
| **Embedding** | `ClipEmbedding` (ViT-L/14) | **Retrieval Core**. Maps images and text to the same high-dimensional vector space, enabling semantic retrieval via cosine similarity between Query and Image vectors. |
| **Generation (LLM)** | `Qwen3-VL-Plus` | Multi-modal LLM. It processes both text and retrieved images (Visual Tokens) directly to generate context-aware answers. |

#### Workflow

1.  **Indexing**: Loads charts from the dataset using `SimpleDirectoryReader`, converts images into vectors using the **CLIP** visual encoder, and stores them.
2.  **Semantic Retrieval**: Converts the user's text query into a vector and retrieves the most semantically relevant charts (Top-K) from the index.
3.  **Multi-modal Generation**: Feeds both the user query and the retrieved charts into **Qwen-VL**, which generates the final answer by combining visual information with text instructions.

### Quick Start

#### 1. Prerequisites

Clone the repository and install dependencies. Using Conda is recommended.

```bash
# Clone the repository
git clone https://github.com/your-username/ChartMirage.git
cd ChartMirage

# Create and activate Conda environment
conda env create -f environment.yml
conda activate ChartMirage
```

#### 2. Configure API Key

The project relies on OpenAI-compatible APIs (e.g., Qwen-VL via DashScope). Ensure `OPENAI_API_KEY` is set in your environment variables.

#### 3. Run RAG Demo

Run the test script to demonstrate the multi-modal RAG pipeline:

```bash
python test.py
```

This script demonstrates how to load chart data, build a multi-modal index, and perform retrieval and QA on specific queries (e.g., sales trends).
