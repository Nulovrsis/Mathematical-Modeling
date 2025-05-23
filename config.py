"""
项目配置文件 - 集中管理所有数据文件路径和共享参数
"""
import os

# 获取项目根目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件路径（使用相对路径）
DATA_PATHS = {
    'temperature_data': '附件1.xlsx',  # 温度数据
    'bacterial_data': '附件2.xlsx',    # 细菌生长实验数据
    'best_model': os.path.join('第一题第二问', 'best_growth_model.py')  # 最佳生长模型
}

# 输出目录
OUTPUT_DIR = 'outputs'
# 如果输出目录不存在，则创建
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        print(f"警告: 无法创建输出目录: {e}")

# 可视化输出路径
VISUALIZATION_PATHS = {
    'temperature_time_plot': os.path.join(OUTPUT_DIR, 'Temperature-Time Plot.png'),
    'temperature_distribution': os.path.join(OUTPUT_DIR, 'Temperature Distribution Histogram.png'),
    'temperature_sine_fit': os.path.join(OUTPUT_DIR, 'Temperature Sine Fit.png'),
    'growth_rate_plot': os.path.join(OUTPUT_DIR, '温度与增殖速率关系图.png'),
    'growth_curves': os.path.join(OUTPUT_DIR, '不同温度下病原细菌的生长曲线.png'),
    'bacterial_count': os.path.join(OUTPUT_DIR, '细菌数量与温度变化.png'),
    'bacterial_ln': os.path.join(OUTPUT_DIR, '细菌lnN与温度变化.png'),
    'model_comparison': os.path.join(OUTPUT_DIR, '不同种群增长模型比较.png'),
    'growth_rate_change': os.path.join(OUTPUT_DIR, '细菌增长率变化.png')
}

# 各模块的结果输出路径
OUTPUT_PATHS = {
    'temperature_model_params': os.path.join(OUTPUT_DIR, '温度模型参数.npz'),
    'simulation_results': os.path.join(OUTPUT_DIR, '细菌种群增长模拟结果.xlsx')
}

# 模型参数
MODEL_PARAMS = {
    'cardinal': {
        'r_opt': 0.5523,  # 最大增殖速率 h^-1
        'T_min': 10.45,   # 最低生长温度 °C
        'T_opt': 28.13,   # 最适生长温度 °C
        'T_max': 38.76    # 最高生长温度 °C
    }
} 