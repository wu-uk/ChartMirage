# Benchmarks Module

该模块包含用于评估 ChartMirage 系统性能的基准测试脚本。

## 目录结构

- [run_baseline_b.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/run_baseline_b.py): 运行基准测试的主脚本，用于生成对比数据。
- [benchmark_vs_standard.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/benchmark_vs_standard.py): 将防御性管道与标准 RAG 管道进行性能对比。
- [verify_defensive_pipeline.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/verify_defensive_pipeline.py): 验证防御性管道在特定攻击场景下的拦截率。
- [analyze_partial.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/analyze_partial.py): 对部分实验结果进行深入分析的工具。
- [debug_baseline_b.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/benchmarks/debug_baseline_b.py): 基准测试调试脚本。

## 评估指标

主要评估指标包括：
- **准确率 (Accuracy)**: 模型回答用户问题的准确程度。
- **拦截率 (Detection Rate)**: 防御管道成功拦截恶意攻击的比例。
- **检索精度 (Retrieval Precision)**: 检索到的 Top-K 图表中包含正确信息的比例。
- **推理延迟 (Latency)**: 整个 RAG 流程的端到端处理时间。

## 使用说明

运行基准测试前，请确保 `dataset/` 目录下的数据完整，并已配置好 `.env` 中的 API 密钥。

```bash
python benchmarks/benchmark_vs_standard.py
```
