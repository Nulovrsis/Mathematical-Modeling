import pandas as pd
import os

# 读取Excel文件
excel_file = os.path.join('..', '附件1.xlsx')
df = pd.read_excel(excel_file)

# 显示前10行数据
print("前10行数据：")
print(df.head(10))

# 显示数据信息
print("\n数据信息：")
print(df.info()) 