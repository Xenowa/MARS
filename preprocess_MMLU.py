import csv

# 输入文件和输出文件的路径
input_file = '/data1/zj/wanzhangqi/dataset/MMLU/test/human_aging_test.csv'  # 替换为你的输入文件路径
output_file = '/data1/zj/wanzhangqi/dataset/MMLU/format/human_aging_test.csv'  # 替换为你的输出文件路径

# 打开输入文件并读取内容
with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)  # 读取所有行

# 处理每一行数据
formatted_data = []
for row in rows:
    # 提取问题和选项
    question = row[0]
    option_a = f"(A){row[1]}"
    option_b = f"(B){row[2]}"
    option_c = f"(C){row[3]}"
    option_d = f"(D){row[4]}"
    
    # 组合问题和选项
    formatted_question = f"{question}\n\nOptions:\n{option_a}\n{option_b}\n{option_c}\n{option_d}\n"
    
    # 提取答案并格式化为 target:(答案)
    answer = row[5]
    formatted_answer = f"({answer})"
    
    # 将格式化后的数据添加到列表中
    formatted_data.append([formatted_question, formatted_answer])

# 将格式化后的数据写入输出文件
with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    # 写入表头
    writer.writerow(['question', 'answer'])
    # 写入数据
    writer.writerows(formatted_data)

print(f"格式化后的数据已保存到 {output_file}")