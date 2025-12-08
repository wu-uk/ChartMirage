# ChartMirage

[中文](#chartmirage-中文) | [English](#chartmirage-english)

---

## ChartMirage (中文)

ChartMirage 是一个专注于生成和评估对抗性（投毒）图表的项目，旨在测试多模态大语言模型（MLLMs）在图表理解和事实核查方面的鲁棒性。

该项目通过生成成对的“干净”图表和“投毒”图表（数据或趋势被篡改），并结合针对性的问答对，来模拟图表数据被操纵或误导的场景，从而评估模型是否能够正确识别视觉信息中的差异。

### 核心架构

项目主要包含以下几个部分：

1.  **数据生成 (Data Generation)**:
    -   核心脚本 `generate_charts.py` 使用 Matplotlib 生成合成图表。
    -   它会生成两组图表：
        -   **Clean (干净版)**: 展示真实、正确的数据趋势。
        -   **Poisoned (投毒版)**: 对数据、趋势或标签进行特定篡改（例如反转趋势、修改极值），旨在误导模型。

2.  **数据集 (Dataset)**:
    -   `dataset/` 目录包含用于评估的图像和元数据。
    -   `final_qa_merged_unified.json` 包含了图像路径、原始描述 (caption)、篡改后的描述 (fake_caption)、查询问题 (query) 以及对应的真实答案和误导性答案。

3.  **评估 (Evaluation)**:
    -   项目包含利用 LlamaIndex 等工具对接多模态模型（如 Qwen-VL）的测试脚本（如 `tmp/test_llm.py`），用于验证模型对生成图表的理解能力。

### 快速开始 (Quick Start)

#### 1. 环境准备

首先，克隆代码仓库并安装依赖。建议使用 Conda 管理环境。

```bash
# 克隆项目
git clone https://github.com/your-username/ChartMirage.git
cd ChartMirage

# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate ChartMirage
```

## ChartMirage (English)

ChartMirage is a project dedicated to generating and evaluating adversarial (poisoned) charts to test the robustness and fact-checking capabilities of Multi-modal Large Language Models (MLLMs).

By generating pairs of "clean" and "poisoned" charts (where data or trends are manipulated) along with targeted QA pairs, the project simulates scenarios where chart data is manipulated or misleading, evaluating whether models can correctly identify discrepancies in visual information.

### Architecture

The project consists of the following main components:

1.  **Data Generation**:
    -   The core script `generate_charts.py` uses Matplotlib to generate synthetic charts.
    -   It produces pairs of charts:
        -   **Clean**: Displays correct data trends.
        -   **Poisoned**: Specific manipulations to data, trends, or labels (e.g., reversing trends, modifying extremes) designed to mislead models.

2.  **Dataset**:
    -   The `dataset/` directory contains images and metadata for evaluation.
    -   `final_qa_merged_unified.json` links image paths, original captions, fake captions, queries, and corresponding true/fake answers.

3.  **Evaluation**:
    -   Includes scripts (e.g., `tmp/test_llm.py`) utilizing tools like LlamaIndex to interface with multi-modal models (e.g., Qwen-VL) for validating model understanding of the generated charts.

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
