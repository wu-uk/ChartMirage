# Dataset Module

本项目使用的数据集目录，包含原始图表、篡改图表以及对应的问答对。

## 目录结构

- `images_merged/`: 包含合法的、未经过篡改的原始图表图片。
- `images_merged_fake/`: 包含经过篡改或带有对抗性信息的虚假图表图片。
- `images_noise_fake/`: 包含带有噪声或其他扰动的对抗性图表（由于体积巨大，未完全上传至 Git）。
- [final_qa_merged_unified.json](file:///home/ASC26team2/wuyukai/project/ChartMirage/dataset/final_qa_merged_unified.json): 统一的问答数据集，包含问题、对应图片路径及标准答案。
- [hash_registry.json](file:///home/ASC26team2/wuyukai/project/ChartMirage/dataset/hash_registry.json): 存储合法图片哈希值的注册表，用于完整性校验。
- `ablation_inject_*.json`: 用于消融实验的不同比例注入数据集。
- [ocr_cache.json](file:///home/ASC26team2/wuyukai/project/ChartMirage/dataset/ocr_cache.json): 缓存的 OCR 识别结果，用于加速处理。

## 数据说明

数据集涵盖了多种图表类型（折线图、柱状图、饼图等）以及多种攻击场景：
1. **内容篡改**: 修改图表中的数值或趋势。
2. **错误标注**: 图表内容与标题或坐标轴说明不符。
3. **噪声攻击**: 在图像中添加人眼难以察觉但可能干扰模型理解的噪声。

## 维护建议

如果添加了新的合法图表，请务必运行 `core/generate_hash_registry.py` 更新哈希注册表，否则防御管道会将其识别为篡改图片。
