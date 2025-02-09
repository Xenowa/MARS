import pandas as pd
import re
import csv

# 加载训练集和测试集
# train_dataset  = pd.read_parquet("/data1/zj/wanzhangqi/dataset/GSM8K/train-00000-of-00001.parquet")
# test_dataset  = pd.read_parquet("/data1/zj/wanzhangqi/dataset/GSM8K/test-00000-of-00001.parquet")

# 查看数据
# print(test_dataset.iloc[2]['answer'])  # 获取第一行的"answer"列数据
# print(test_dataset.head())  # 获取第一行的"answer"列数据


## 读取Parquet文件
test_dataset = pd.read_parquet("/data1/zj/wanzhangqi/dataset/GSM8K/test-00000-of-00001.parquet")

# print(test_dataset.iloc[2]['answer'])  # 获取第一行的"answer"列数据

# 提取####后面的内容
def extract_answer(answer):
    match = re.search(r'####\s*(\S+)', answer)
    if match:
        return match.group(1)
    return None

# 生成新的answer列
test_dataset['new_answer'] = test_dataset['answer'].apply(extract_answer)

# 创建新的DataFrame
new_dataset = test_dataset[['question', 'new_answer']]
new_dataset.columns = ['question', 'answer']

# 保存为CSV文件，强制所有字段用引号包围
new_dataset.to_csv("gsm8k_test.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

print("文件已保存为 gsm8k_test.csv")