import pandas as pd

# 读取旧的CSV文件
df = pd.read_csv('/data1/zj/wanzhangqi/dataset/C-EVAL/val/clinical_medicine_val.csv')

# 格式化question列
df['question'] = df['question'] + '\nOptions:\n' + \
                 '(A) ' + df['A'].astype(str) + '\n' + \
                 '(B) ' + df['B'].astype(str) + '\n' + \
                 '(C) ' + df['C'].astype(str) + '\n' + \
                 '(D) ' + df['D'].astype(str)

# 格式化answer列
df['answer'] = '(' + df['answer'].astype(str) + ')'

# 选择需要的列
new_df = df[['question', 'answer']]

# 保存为新的CSV文件
new_df.to_csv('/data1/zj/wanzhangqi/dataset/C-EVAL/format/clinical_medicine_val.csv', index=False)

print("转换完成，新的CSV文件已保存为 'clinical_medicine_val.csv'")