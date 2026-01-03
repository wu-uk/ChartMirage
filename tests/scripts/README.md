# Test Scripts Module

该目录包含用于测试项目各个组件功能和鲁棒性的工具脚本。

## 脚本列表

- [test_new_pipeline.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/test_new_pipeline.py): 测试最新的防御性 RAG 管道流程。
- [test_consistency.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/test_consistency.py): 专门测试信号一致性检测模块。
- [test_baseline.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/test_baseline.py): 运行基础的 RAG 测试，不含防御机制。
- [run_noise_experiments.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/run_noise_experiments.py): 运行针对噪声干扰的鲁棒性实验。
- [augment_images.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/augment_images.py): 图像增强工具，用于生成更多测试用例。
- [analyze_noise_impact.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/analyze_noise_impact.py): 分析不同级别的噪声对模型性能的影响。
- [test_llm.py](file:///home/ASC26team2/wuyukai/project/ChartMirage/tests/scripts/test_llm.py): 测试底层多模态大模型的连接和基本能力。

## 使用建议

在进行重大代码修改后，建议运行 `test_new_pipeline.py` 以确保防御机制依然有效。

```bash
python tests/scripts/test_new_pipeline.py
```
