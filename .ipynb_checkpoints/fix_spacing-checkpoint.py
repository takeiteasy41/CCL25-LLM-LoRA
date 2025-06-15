import os
from rich.console import Console

console = Console()

# --- 配置 ---
# 输入文件名（你当前有问题的提交文件）
input_filename = "submission.txt"
# 输出文件名（修复后的新文件）
output_filename = "submission_fixed.txt"

def fix_submission_spacing(input_file, output_file):
    """
    读取输入文件，修复[END]前的多余空格，并写入输出文件。
    """
    if not os.path.exists(input_file):
        console.log(f"[bold red]错误：输入文件 '{input_file}' 不存在。请检查文件名和路径。[/bold red]")
        return

    lines_processed = 0
    lines_changed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            lines_processed += 1
            # 核心替换逻辑：将两个空格加[END] 替换为一个空格加[END]
            # 我们用replace进行多次替换，确保即时有更多空格也能处理
            fixed_line = line.replace("  [END]", " [END]")
            
            # 为了更稳妥，处理可能存在的三个或更多空格的情况
            while "  [END]" in fixed_line:
                fixed_line = fixed_line.replace("  [END]", " [END]")

            if line != fixed_line:
                lines_changed += 1

            # 写入修复后的行，注意要保留原始的换行符
            f_out.write(fixed_line)

    console.log(f"[bold green]处理完成！[/bold green]")
    console.log(f"总共处理了 [cyan]{lines_processed}[/cyan] 行。")
    console.log(f"成功修复了 [cyan]{lines_changed}[/cyan] 行。")
    console.log(f"修复后的文件已保存为: [bold magenta]{output_file}[/bold magenta]")
    console.log("你现在可以提交这个新文件了。")

# --- 主程序入口 ---
if __name__ == "__main__":
    fix_submission_spacing(input_filename, output_filename)