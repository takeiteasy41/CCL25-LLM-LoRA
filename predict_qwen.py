import json
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from peft import PeftModel
from tqdm import tqdm
from rich.console import Console
import gc

console = Console()

# --- 1. 配置 ---
BASE_MODEL_PATH = "/root/autodl-tmp/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
PEFT_MODEL_PATH = "./qwen-lora-finetuned-compat/final_model"
TEST_FILE_PATH = "test1.json"
SUBMISSION_FILE_PATH = "submission.txt" 
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")

    # --- 2. 加载模型和分词器 (无量化，优化显存) ---
    console.log(f"Loading tokenizer from [cyan]{BASE_MODEL_PATH}[/cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.log(f"Tokenizer loaded. Pad token set to: {tokenizer.pad_token}")
    
    console.log(f"Loading base model from [cyan]{BASE_MODEL_PATH}[/cyan]...")
    base_model = Qwen2ForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    base_model.resize_token_embeddings(len(tokenizer))
    
    console.log(f"Loading LoRA adapter from [cyan]{PEFT_MODEL_PATH}[/cyan]...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
    model.eval()
    
    # 强制进行垃圾回收，释放不必要的内存
    gc.collect()
    torch.cuda.empty_cache()
    
    console.log("Model with LoRA adapter loaded successfully.")

    # --- 3. 加载测试数据 ---
    console.log(f"Loading test data from [cyan]{TEST_FILE_PATH}[/cyan]...")
    try:
        with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        console.log(f"[bold red]Error: Test file not found at '{TEST_FILE_PATH}'. Please check the path.[/bold red]")
        return

    input_texts = [item['content'] for item in test_data]
    console.log(f"Loaded {len(input_texts)} samples from {TEST_FILE_PATH} in original order.")

    # --- 4. 批量推理 ---
    few_shot_prompt = (
        "从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
        "输入：你可真是头蠢驴，这都做不好。\n"
        "输出：你 | 蠢驴 | non-hate | non-hate [END]\n\n"
        "从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
        "输入：老黑我是真的讨厌，媚黑的还倒贴。\n"
        "输出：老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]\n\n"
    )
    batch_size = 8
    all_predictions = []

    console.log("Starting prediction loop...")
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Predicting"):
        prompts = []
        for content in input_texts[i:i+batch_size]:
            text = few_shot_prompt
            text += f"从以下文本中抽取出所有（评论对象 | 论点 | 目标群体 | 是否仇恨）的四元组。\n"
            text += f"输入：{content}\n"
            text += f"输出："
            prompts.append(text)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                # 【优化】调整生成参数
                max_new_tokens=128,      # 限制最大生成长度，防止失控
                num_beams=5,             # 使用Beam Search提升质量
                do_sample=False,         # 关闭随机采样，保证结果稳定
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        decoded_outputs = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        all_predictions.extend([output.strip() for output in decoded_outputs])

    # --- 5. 后处理和保存 ---
    console.log(f"Saving predictions to [cyan]{SUBMISSION_FILE_PATH}[/cyan]...")
    final_outputs = []
    for pred in all_predictions:
        # 【核心修复】在拼接前，对字符串进行 .strip() 清理
        if "[END]" in pred:
            # 1. 截断
            main_part = pred.split("[END]")[0]
            # 2. 清理尾部空格
            clean_part = main_part.strip()
            # 3. 拼接
            clean_pred = clean_part + " [END]"
        else:
            clean_pred = pred.strip() + " [END]"
        
        # 移除非预期的换行符，保证每条预测只有一行
        clean_pred = clean_pred.replace('\n', ' ').replace('\r', '')
        
        final_outputs.append(clean_pred)

    with open(SUBMISSION_FILE_PATH, 'w', encoding='utf-8') as f:
        for pred in final_outputs:
            f.write(pred + '\n') # 写入文件时，write会自动加换行符，所以pred本身不要带\n

    console.log("[bold green]Submission file created successfully![/bold green]")
    console.log(f"--- First 5 predictions in {SUBMISSION_FILE_PATH} ---")
    for i in range(min(5, len(final_outputs))):
        console.log(final_outputs[i])


if __name__ == "__main__":
    main()