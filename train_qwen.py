import os
import torch
from rich.console import Console
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

console = Console()

# --- 1. 配置 (保持不变) ---
MODEL_PATH = "/root/autodl-tmp/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
PROCESSED_DATA_DIR = "processed_data_qwen"
OUTPUT_DIR = "qwen-lora-finetuned-compat" # 新的输出目录

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 

# 训练参数
EPOCHS = 5 # 5个epoch在完整数据上是比较合适的起点
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4 # 这个学习率可以保持

# 【恢复】将steps恢复到适合大数据集的值
LOGGING_STEPS = 50
SAVE_STEPS = 200 # 在完整数据上，可以不用那么频繁地保存
# EVAL_STEPS = 200 # 如果你加回了评估，也需要调整

# --- 2. 加载模型和分词器 (无量化版本) ---
console.log("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
console.log(f"Tokenizer loaded. Pad token set to: {tokenizer.pad_token}")

# 【核心修改】移除 device_map="auto"
# 模型会默认加载到CPU，之后由Trainer移动到GPU
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # device_map="auto", # <-- 注释或删除这一行
    local_files_only=True
)

model.resize_token_embeddings(len(tokenizer))

# --- 3. LoRA配置 (保持不变) ---
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
console.log("LoRA configured. Trainable parameters:")
model.print_trainable_parameters()

# --- 4. 加载数据集 (保持不变) ---
console.log("Loading and tokenizing datasets...")
tokenized_datasets = load_from_disk(PROCESSED_DATA_DIR).map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=1024),
    batched=True,
    remove_columns=["text"]
)

# --- 5. 训练 (使用最兼容的参数) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    
    # 【核心修改】移除所有可能引起冲突的 strategy 参数
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    
    # 【核心修改】由于没有 evaluation_strategy，我们暂时无法在训练中进行评估
    # 我们将在训练结束后手动评估或直接用最终模型进行预测
    # 相关的评估参数也需要注释掉
    # evaluation_strategy="steps", # 移除
    # eval_steps=EVAL_STEPS,       # 移除
    # load_best_model_at_end=True, # 移除
    # metric_for_best_model="eval_loss", # 移除
    # greater_is_better=False,      # 移除
    
    save_total_limit=3,
    bf16=True, # 假设你当前版本支持 bf16，如果不行就改成 fp16=True
    report_to="none",
)

# 【核心修改】由于无法在训练中评估，我们暂时不传入 eval_dataset 和 compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["test"], # 暂时移除
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # compute_metrics=compute_metrics # 暂时移除
)

console.log("Starting Qwen-based model training (compatibility mode)...")
trainer.train()

console.log("Saving the final model...")
# 训练结束后，我们直接保存最终的模型
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
console.log("[bold green]Training complete![/bold green]")