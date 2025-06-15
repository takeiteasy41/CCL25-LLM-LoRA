本项目为 CCL25-Eval 任务10：细粒度中文仇恨识别评测 比赛项目，包含结果文件及相关代码。

项目流程为：train.json -> 数据预处理模块 (构建Few-shot Prompt) -> LoRA微调模块 (冻结的DeepSeek-Qwen-7B + 可训练的LoRA层) -> 推理模块 (输入测试集文本，使用Beam Search生成) -> submission.txt (初步结果) -> 后处理脚本 -> submission_fixed.txt (最终提交)
deepseek-r1-distill-qwen-7b模型文件总共约15GB，没有放在项目内，需要自行下载，文件夹与ccl25-hate-speech文件夹并列放置。
感谢 CCL25-Eval 组委会提供平台和STATE ToxiCN 论文 开源高质量细粒度中文仇恨识别数据集。
