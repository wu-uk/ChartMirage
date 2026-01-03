# ChartMirage

[中文](#chartmirage-中文) | [English](#chartmirage-english)

---

## ChartMirage (中文)

ChartMirage 是一个专注于评估和提升多模态大语言模型（MLLMs）在图表理解方面鲁棒性的研究项目。本项目通过构建一个带有**多级防御机制的增强型多模态 RAG (Retrieval-Augmented Generation)** 架构，旨在识别并拦截针对图表数据的对抗性攻击。

### 核心特性

- **多级防御管道 (Defensive RAG Pipeline)**:
    1.  **完整性校验 (Hash Check)**: 毫秒级拦截被非法篡改的图表。
    2.  **信号一致性检测 (Signal Consistency)**: 通过特征空间分析识别潜在的对抗样本。
    3.  **多模态语义审计 (VLM Audit)**: 利用 VLM 的跨模态推理能力对检索结果进行深度审计。
- **标准化路径管理**: 全局动态路径解析，支持跨环境部署，所有实验结果统一输出至 `outputs/`。
- **模块化架构**: 核心逻辑、基准测试、实验脚本和数据集清晰分离。

### 项目结构

- [core/](file:///home/ASC26team2/wuyukai/project/ChartMirage/core/): 包含防御管道的核心实现 `defensive_rag_pipeline.py` 及关键实验脚本。
- [benchmarks/](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/): 性能评估和对比测试脚本。
- [dataset/](file:///home/ASC26team2/wuyukai/project/ChartMirage/dataset/): 包含图表图片、问答对、哈希注册表及 OCR 缓存。
- [tests/](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/): 各种功能测试和分析工具。
- [data_charts_gen/](file:///home/ASC26team2/wuyukai/project/ChartMirage/data_charts_gen/): 图表素材与生成相关配置。

### 快速开始

#### 1. 环境准备

建议使用 Conda 管理环境：

```bash
conda env create -f environment.yml
conda activate ChartMirage
```

#### 2. 配置 API Key

在根目录下创建 `.env` 文件（或修改模板），配置您的 API 密钥：

```env
OPENAI_API_KEY="your_api_key_here"
OPENAI_API_BASE="https://your_api_endpoint/v1"
```

#### 3. 运行验证

测试防御性 RAG 管道：

```bash
python benchmarks/verify_defensive_pipeline.py
```

---

## ChartMirage (English)

ChartMirage is a research project dedicated to evaluating and enhancing the robustness of Multi-modal Large Language Models (MLLMs) in chart understanding. It features an **Enhanced Multi-modal RAG (Retrieval-Augmented Generation)** architecture with multi-level defense mechanisms designed to detect and intercept adversarial attacks on chart data.

### Key Features

- **Multi-level Defensive Pipeline**:
    1.  **Integrity Verification (Hash Check)**: Millisecond-level interception of tampered charts.
    2.  **Signal Consistency Detection**: Identifies potential adversarial samples via feature space analysis.
    3.  **Semantic Audit (VLM Audit)**: Deeply audits retrieval results using cross-modal reasoning of VLMs.
- **Standardized Path Management**: Global dynamic path resolution supporting cross-environment deployment, with all outputs centralized in `outputs/`.
- **Modular Architecture**: Clean separation of core logic, benchmarks, experiments, and datasets.

### Project Structure

- [core/](file:///home/ASC26team2/wuyukai/project/ChartMirage/core/): Core implementation of `defensive_rag_pipeline.py` and key experiments.
- [benchmarks/](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/): Performance evaluation and comparison scripts.
- [dataset/](file:///home/ASC26team2/wuyukai/project/ChartMirage/dataset/): Chart images, QA pairs, hash registry, and OCR cache.
- [tests/](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/): Functional tests and analysis tools.
- [data_charts_gen/](file:///home/ASC26team2/wuyukai/project/ChartMirage/data_charts_gen/): Chart assets and generation configurations.

### Quick Start

#### 1. Prerequisites

Conda environment is recommended:

```bash
conda env create -f environment.yml
conda activate ChartMirage
```

#### 2. Configure API Key

Create a `.env` file in the root directory and set your API key:

```env
OPENAI_API_KEY="your_api_key_here"
OPENAI_API_BASE="https://your_api_endpoint/v1"
```

#### 3. Run Verification

Test the defensive RAG pipeline:

```bash
python benchmarks/verify_defensive_pipeline.py
```
