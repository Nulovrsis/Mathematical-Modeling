import numpy as np

# 参数注释字典
param_comments = {
    'A': '主周期正弦振幅',
    'phi': '主周期正弦相位',
    'mu': '温度均值',
    'A2': '次周期正弦振幅',
    'phi2': '次周期正弦相位',
    'k': '线性趋势系数'
}

data = np.load('outputs/温度模型参数.npz')

with open('outputs/温度模型参数.txt', 'w', encoding='utf-8') as f:
    for key in data.files:
        value = data[key]
        comment = param_comments.get(key, '')
        if value.size == 1:
            f.write(f"{key} = {value.item()}  # {comment}\n")
        else:
            f.write(f"{key} = {value.tolist()}  # {comment}\n")

print("已导出为 outputs/温度模型参数.txt，并添加注释。")