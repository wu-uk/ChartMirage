# Core Module

本项目核心模块，包含防御性 RAG 管道的实现及相关实验脚本。

## 目录结构

- [defensive_rag_pipeline.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/core/defensive_rag_pipeline.py): 核心防御性 RAG 管道实现，包含三层防御机制：
  1. **Hash Check**: 基于图像哈希的完整性校验。
  2. **Signal Consistency**: 基于模型预测信号的一致性检测。
  3. **VLM Audit**: 基于视觉语言模型（VLM）的语义审计。
- [generate_hash_registry.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/core/generate_hash_registry.py): 用于生成图像哈希注册表的工具脚本。
- `experiments/`: 包含各种实验脚本，用于验证不同防御策略的有效性。
  - `consistency_check/`: 信号一致性检测的相关模型和训练脚本。
  - `standard_rag_pipeline.py`: 标准 RAG 流程的基准实现。
  - `plot_results.py`: 实验结果可视化工具。

## 关键功能

### DefensiveRAGPipeline
该类集成了多级防御机制，能够有效识别并拦截对抗性图表攻击（如图像篡改、错误标注等）。它使用 LlamaIndex 框架构建，并结合了 CLIP 嵌入和 Qwen-VL 等先进模型。

### 哈希校验
通过预先计算合法图表的哈希值并存储在 `hash_registry.json` 中，在检索阶段实时比对，确保加载的图表未被篡改。

### 环境适配建议

在某些环境下，PaddleOCR 可能无法正确加载 CUDA 相关的动态库（如 `libnvrtc.so`）。[defensive_rag_pipeline.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/core/defensive_rag_pipeline.py) 中包含了一段硬编码的 `nvidia_lib_path` 用于修复此问题：

```python
nvidia_lib_path = '/home/ASC26team2/miniconda3/envs/ChartMirage/lib/python3.12/site-packages/nvidia/cu13/lib'
```

**注意**：如果您在其他机器或环境下运行，请根据实际的 Conda 环境路径修改此变量，或者确保系统 `LD_LIBRARY_PATH` 已包含正确的 NVIDIA 库路径。
