import json
from datasets import Dataset, DatasetDict
from rich.console import Console

console = Console()

# --- 配置 ---
TRAIN_FILE = "train.json"
OUTPUT_DIR = "processed_data_qwen" # 使用一个新目录名，防止与旧数据混淆
TEST_SIZE = 0.1
SEED = 42

# --- 加载数据 ---
console.log(f"Loading data from [cyan]{TRAIN_FILE}[/cyan]...")
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 【核心修改】只取一小部分数据用于快速实验
#SUBSET_SIZE = 500
#raw_data = raw_data[:SUBSET_SIZE]
#console.log(f"[yellow]Using a subset of {SUBSET_SIZE} samples for quick iteration.[/yellow]")

dataset = Dataset.from_list(raw_data)

# --- 划分数据集 ---
dataset_dict = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
console.log(f"Dataset split into training ({len(dataset_dict['train'])}) and validation ({len(dataset_dict['test'])}) sets.")

# 【核心修改】为基础模型构建更严格的 Few-shot Prompt 和目标
def create_fewshot_prompt(examples):
    prompts = []
    # 示例部分保持不变
    few_shot_examples = (
        "从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
        "输入：你可真是头蠢驴，这都做不好。\n"
        "输出：你 | 蠢驴 | non-hate | non-hate [END]\n\n"
        "从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
        "输入：老黑我是真的讨厌，媚黑的还倒贴。\n"
        "输出：老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]\n\n"
    )

    for content, output in zip(examples["content"], examples["output"]):
        # 1. 规范化目标文本中的标签
        # 这是为了让模型学习到更一致的格式
        parts = output.split(" [END]")[0].split(" [SEP] ")
        new_parts = []
        for part in parts:
            elements = part.split(" | ")
            if len(elements) == 4:
                # 对目标群体标签按字母排序
                groups = sorted([g.strip() for g in elements[2].split(',')])
                elements[2] = ", ".join(groups)
                new_parts.append(" | ".join(elements))
        
        # 重新拼接规范化后的 output
        # 注意：这里我们只规范化，不改变原始数据内容，因为我们是自监督学习
        # 这个规范化步骤主要用于更复杂的方案，这里我们先注释掉，保留思路
        # normalized_output = " [SEP] ".join(new_parts) + " [END]"
        
        # 2. 构建最终的训练文本
        text = few_shot_examples
        text += f"从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
        text += f"输入：{content}\n"
        # 训练时，模型学习的目标是补完 "输出：" 之后的所有内容，包括答案和停止符
        text += f"输出：{output}</s>" 
        prompts.append(text)
        
    return {"text": prompts}

console.log("Applying few-shot prompt template to datasets...")
processed_datasets = dataset_dict.map(create_fewshot_prompt, batched=True, remove_columns=dataset.column_names)

# --- 保存 ---
console.log(f"Saving processed datasets to [cyan]{OUTPUT_DIR}[/cyan]...")
processed_datasets.save_to_disk(OUTPUT_DIR)
console.log("[bold green]Data preparation for Qwen model complete![/bold green]")