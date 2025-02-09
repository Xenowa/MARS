import json
import csv

# 输入和输出文件路径
input_jsonl_file = 'lsat-ar.jsonl'
output_csv_file = 'lsat-ar_format.csv'

# 打开JSONL文件并读取内容
with open(input_jsonl_file, 'r', encoding='utf-8') as jsonl_file, \
     open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    
    # 创建CSV写入器
    csv_writer = csv.writer(csv_file)
    
    # 写入CSV文件的表头
    csv_writer.writerow(['question', 'answer'])
    
    # 逐行读取JSONL文件
    for line in jsonl_file:
        # 解析JSON行
        data = json.loads(line.strip())
        
        # 组合passage、question和options
        options_text = "\n".join(data['options'])  # 将options列表转换为文本，每项占一行
        question = f"{data['passage']}\nquestions:\n{data['question']}\nOptions:\n{options_text}"
        
        # 获取label并加上括号
        answer = f"({data['label']})"
        
        # 写入CSV文件
        csv_writer.writerow([question, answer])

print(f"转换完成，结果已保存到 {output_csv_file}")