import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy import stats
import matplotlib.ticker as mtick
from matplotlib import font_manager
import matplotlib as mpl
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入配置
from config import DATA_PATHS, VISUALIZATION_PATHS, MODEL_PARAMS

# 设置更优美的风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'     # 使用STIX字体渲染数学符号
plt.rcParams['font.size'] = 12

print("加载实验数据...")
# 加载数据 - 附件2中的数据
try:
    # 尝试直接从Excel文件读取
    data = pd.read_excel(DATA_PATHS['bacterial_data'], header=1)  # 假设表头在第二行
    print("成功直接从Excel文件读取数据")
except Exception as e:
    print(f"从Excel读取失败: {e}")
    print("使用手动输入的数据")
    # 实验数据
temperatures = [20, 25, 28, 30]
# 实验观察到的不同温度下细菌数量
bacteria_counts = {
    20: [0, 2030, 13100, 182000, 1100000, 4930000, 60800000, 107000000, 492000000, 750000000, 772000000],
    25: [0, 3130, 16900, 144000, 431000, 14500000, 139000000, 320000000, 736000000, 821000000, 853000000],
    28: [0, 3130, 13900, 124000, 764000, 5380000, 77500000, 176000000, 672000000, 837000000, 824000000],
    30: [0, 1860, 5890, 12500, 268000, 938000, 23800000, 53800000, 307000000, 647000000, 747000000]
}

# 观察时间点
time_points = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# 为每个温度计算r值（指数增长模型）
print("计算各温度下的增殖速率...")

def calculate_growth_rate(temps, counts_dict, times):
    """计算各温度下的增殖速率与拟合优度"""
    growth_rates = []
    r_squared_values = []
    
    for temp in temps:
        # 获取当前温度的细菌计数
        counts = counts_dict[temp]
        # 过滤掉零值，计算自然对数
        valid_indices = [i for i, count in enumerate(counts) if count > 0]
        log_counts = np.log(np.array([counts[i] for i in valid_indices]))
        valid_times = np.array([times[i] for i in valid_indices])
        
        # 进行线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_times, log_counts)
        
        growth_rates.append(slope)
        r_squared_values.append(r_value**2)
        
        print(f"温度: {temp}°C, 增殖速率: {slope:.4f} h⁻¹, R²: {r_value**2:.4f}")
    
    return growth_rates, r_squared_values

temps = list(bacteria_counts.keys())
growth_rates, r_squared_values = calculate_growth_rate(temps, bacteria_counts, time_points)

# 1. 二次多项式模型
def quadratic_model(T, a, b, c):
    """二次函数模型: r(T) = aT² + bT + c"""
    return a * T**2 + b * T + c

# 2. Cardinal温度模型
def cardinal_model(T, r_opt, T_min, T_opt, T_max):
    """Cardinal温度模型，符合微生物生理学特性
    
    r(T) = r_opt * ((T-T_min)/(T_opt-T_min)) * ((T_max-T)/(T_max-T_opt))^((T_max-T_opt)/(T_opt-T_min))
    
    参数:
    - r_opt: 最适温度下的最大增殖速率
    - T_min: 最低生长温度
    - T_opt: 最适生长温度
    - T_max: 最高生长温度
    """
    if isinstance(T, (list, np.ndarray)):
        result = np.zeros_like(T, dtype=float)
        # 在有效温度范围内计算
        valid_idx = (T >= T_min) & (T <= T_max)
        if not np.any(valid_idx):
            return result
        
        T_valid = T[valid_idx]
        # 计算有效温度下的增殖速率
        numerator = (T_valid - T_min) * (T_max - T_valid)
        denominator = (T_opt - T_min) * (T_max - T_opt)
        exponent = (T_max - T_opt) / (T_opt - T_min)
        result[valid_idx] = r_opt * numerator / denominator * ((T_max - T_valid) / (T_max - T_opt)) ** exponent
        return result
    else:
        # 单个温度值的情况
        if T < T_min or T > T_max:
            return 0.0
        numerator = (T - T_min) * (T_max - T)
        denominator = (T_opt - T_min) * (T_max - T_opt)
        exponent = (T_max - T_opt) / (T_opt - T_min)
        return r_opt * numerator / denominator * ((T_max - T) / (T_max - T_opt)) ** exponent

# 3. 改进的Ratkowsky模型（平方根模型）
def ratkowsky_model(T, b, T_min, T_max):
    """Ratkowsky模型: √r = b(T-T_min)(1-exp(c(T-T_max)))"""
    if isinstance(T, (list, np.ndarray)):
        result = np.zeros_like(T, dtype=float)
        valid_idx = (T > T_min) & (T < T_max)
        T_valid = T[valid_idx]
        result[valid_idx] = (b * (T_valid - T_min) * (1 - np.exp(0.1 * (T_valid - T_max))))**2
        return result
    else:
        if T <= T_min or T >= T_max:
            return 0.0
        return (b * (T - T_min) * (1 - np.exp(0.1 * (T - T_max))))**2

print("\n开始拟合多种模型...")

# 拟合二次多项式模型
quadratic_params, quadratic_cov = optimize.curve_fit(quadratic_model, temps, growth_rates)
a, b, c = quadratic_params
print(f"二次模型拟合结果: r(T) = {a:.6f}T² + {b:.6f}T + {c:.6f}")

# 计算二次模型拟合优度
quadratic_pred = quadratic_model(np.array(temps), *quadratic_params)
quadratic_r2 = 1.0 - (np.sum((np.array(growth_rates) - quadratic_pred)**2) / 
                      np.sum((np.array(growth_rates) - np.mean(growth_rates))**2))
print(f"二次模型拟合优度 R²: {quadratic_r2:.4f}")

# 拟合Cardinal温度模型，需要设置更合理的初始参数和约束
try:
    # 根据微生物学经验，设置合理的参数约束
    cardinal_bounds = ([0.3, 5, 25, 32], [0.7, 15, 30, 45])  # r_opt, T_min, T_opt, T_max
    cardinal_params, cardinal_cov = optimize.curve_fit(
        cardinal_model, temps, growth_rates, 
        p0=[0.6, 10, 28, 35],  # 合理的初始猜测
        bounds=cardinal_bounds,
        maxfev=10000
    )
    r_opt, T_min, T_opt, T_max = cardinal_params
    print(f"Cardinal模型拟合结果:")
    print(f"- 最大增殖速率 r_opt: {r_opt:.4f} h⁻¹")
    print(f"- 最低生长温度 T_min: {T_min:.2f}°C")
    print(f"- 最适生长温度 T_opt: {T_opt:.2f}°C")
    print(f"- 最高生长温度 T_max: {T_max:.2f}°C")
    
    # 计算Cardinal模型拟合优度
    cardinal_pred = cardinal_model(np.array(temps), *cardinal_params)
    cardinal_r2 = 1.0 - (np.sum((np.array(growth_rates) - cardinal_pred)**2) / 
                         np.sum((np.array(growth_rates) - np.mean(growth_rates))**2))
    print(f"Cardinal模型拟合优度 R²: {cardinal_r2:.4f}")
except Exception as e:
    print(f"Cardinal模型拟合失败: {e}")
    # 使用配置文件中的默认参数值
    r_opt = MODEL_PARAMS['cardinal']['r_opt']
    T_min = MODEL_PARAMS['cardinal']['T_min']
    T_opt = MODEL_PARAMS['cardinal']['T_opt']
    T_max = MODEL_PARAMS['cardinal']['T_max']
    cardinal_r2 = 0
    print(f"使用配置文件中的Cardinal模型参数：r_opt={r_opt}, T_min={T_min}, T_opt={T_opt}, T_max={T_max}")

# 拟合Ratkowsky模型
try:
    ratkowsky_bounds = ([0, 5, 32], [0.1, 15, 45])  # b, T_min, T_max
    ratkowsky_params, ratkowsky_cov = optimize.curve_fit(
        ratkowsky_model, temps, growth_rates,
        p0=[0.05, 10, 35],
        bounds=ratkowsky_bounds,
        maxfev=10000
    )
    b_rat, T_min_rat, T_max_rat = ratkowsky_params
    print(f"Ratkowsky模型拟合结果:")
    print(f"- 参数 b: {b_rat:.6f}")
    print(f"- 最低生长温度 T_min: {T_min_rat:.2f}°C")
    print(f"- 最高生长温度 T_max: {T_max_rat:.2f}°C")
    
    # 计算Ratkowsky模型拟合优度
    ratkowsky_pred = ratkowsky_model(np.array(temps), *ratkowsky_params)
    ratkowsky_r2 = 1.0 - (np.sum((np.array(growth_rates) - ratkowsky_pred)**2) / 
                          np.sum((np.array(growth_rates) - np.mean(growth_rates))**2))
    print(f"Ratkowsky模型拟合优度 R²: {ratkowsky_r2:.4f}")
except Exception as e:
    print(f"Ratkowsky模型拟合失败: {e}")
    # 设置一些默认值
    b_rat, T_min_rat, T_max_rat = 0.05, 10, 35
    ratkowsky_r2 = 0

# 选择最佳模型
r2_values = {
    "二次多项式模型": quadratic_r2,
    "Cardinal温度模型": cardinal_r2,
    "Ratkowsky模型": ratkowsky_r2
}
best_model = max(r2_values.items(), key=lambda x: x[1])[0]
print(f"\n最佳拟合模型是: {best_model}，R² = {r2_values[best_model]:.4f}")

# 保存最佳模型函数到文件
def save_best_model():
    """保存最佳模型函数定义和参数到Python文件，方便其他程序导入使用"""
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_growth_model.py')
    with open(best_model_path, 'w', encoding='utf-8') as f:
        f.write("import numpy as np\n\n")
        
        # 写入所有模型函数定义
        f.write("# 二次多项式模型\n")
        f.write("def quadratic_model(T, a, b, c):\n")
        f.write("    return a * T**2 + b * T + c\n\n")
        
        f.write("# Cardinal温度模型\n")
        f.write("def cardinal_model(T, r_opt, T_min, T_opt, T_max):\n")
        f.write("    if isinstance(T, (list, np.ndarray)):\n")
        f.write("        result = np.zeros_like(T, dtype=float)\n")
        f.write("        valid_idx = (T >= T_min) & (T <= T_max)\n")
        f.write("        if not np.any(valid_idx):\n")
        f.write("            return result\n")
        f.write("        T_valid = T[valid_idx]\n")
        f.write("        numerator = (T_valid - T_min) * (T_max - T_valid)\n")
        f.write("        denominator = (T_opt - T_min) * (T_max - T_opt)\n")
        f.write("        exponent = (T_max - T_opt) / (T_opt - T_min)\n")
        f.write("        result[valid_idx] = r_opt * numerator / denominator * ((T_max - T_valid) / (T_max - T_opt)) ** exponent\n")
        f.write("        return result\n")
        f.write("    else:\n")
        f.write("        if T < T_min or T > T_max:\n")
        f.write("            return 0.0\n")
        f.write("        numerator = (T - T_min) * (T_max - T)\n")
        f.write("        denominator = (T_opt - T_min) * (T_max - T_opt)\n")
        f.write("        exponent = (T_max - T_opt) / (T_opt - T_min)\n")
        f.write("        return r_opt * numerator / denominator * ((T_max - T) / (T_max - T_opt)) ** exponent\n\n")
        
        f.write("# Ratkowsky模型\n")
        f.write("def ratkowsky_model(T, b, T_min, T_max):\n")
        f.write("    if isinstance(T, (list, np.ndarray)):\n")
        f.write("        result = np.zeros_like(T, dtype=float)\n")
        f.write("        valid_idx = (T > T_min) & (T < T_max)\n")
        f.write("        T_valid = T[valid_idx]\n")
        f.write("        result[valid_idx] = (b * (T_valid - T_min) * (1 - np.exp(0.1 * (T_valid - T_max))))**2\n")
        f.write("        return result\n")
        f.write("    else:\n")
        f.write("        if T <= T_min or T >= T_max:\n")
        f.write("            return 0.0\n")
        f.write("        return (b * (T - T_min) * (1 - np.exp(0.1 * (T - T_max))))**2\n\n")
        
        # 写入参数值
        f.write("# 模型参数值\n")
        f.write("quadratic_params = {}\n")
        f.write(f"quadratic_params['a'] = {a}\n")
        f.write(f"quadratic_params['b'] = {b}\n")
        f.write(f"quadratic_params['c'] = {c}\n\n")
        
        f.write("cardinal_params = {}\n")
        f.write(f"cardinal_params['r_opt'] = {r_opt}\n")
        f.write(f"cardinal_params['T_min'] = {T_min}\n")
        f.write(f"cardinal_params['T_opt'] = {T_opt}\n")
        f.write(f"cardinal_params['T_max'] = {T_max}\n\n")
        
        f.write("ratkowsky_params = {}\n")
        f.write(f"ratkowsky_params['b'] = {b_rat}\n")
        f.write(f"ratkowsky_params['T_min'] = {T_min_rat}\n")
        f.write(f"ratkowsky_params['T_max'] = {T_max_rat}\n\n")
        
        # 写入最佳模型信息
        f.write("# 最佳模型信息\n")
        f.write(f"best_model_name = '{best_model}'\n")
        
        # 定义通用的生长速率计算函数
        f.write("\n# 通用生长速率计算函数\n")
        f.write("def calculate_growth_rate(T):\n")
        f.write("    \"\"\"根据温度T计算细菌增殖速率r\"\"\"\n")
        if best_model == "二次多项式模型":
            f.write("    return quadratic_model(T, quadratic_params['a'], quadratic_params['b'], quadratic_params['c'])\n")
        elif best_model == "Cardinal温度模型":
            f.write("    return cardinal_model(T, cardinal_params['r_opt'], cardinal_params['T_min'], cardinal_params['T_opt'], cardinal_params['T_max'])\n")
        else:  # Ratkowsky模型
            f.write("    return ratkowsky_model(T, ratkowsky_params['b'], ratkowsky_params['T_min'], ratkowsky_params['T_max'])\n")
    
    print(f"最佳模型已保存到 {best_model_path}")

save_best_model()

print("\n生成可视化图表...")
# 创建更精细的温度范围用于绘制平滑曲线
fine_temps = np.linspace(5, 40, 500)

# 计算各模型在精细温度范围的预测值
quadratic_rates = quadratic_model(fine_temps, *quadratic_params)
cardinal_rates = cardinal_model(fine_temps, *cardinal_params)
ratkowsky_rates = ratkowsky_model(fine_temps, *ratkowsky_params)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制数据点
ax.scatter(temps, growth_rates, color='black', s=100, marker='o', 
          label='实验数据点', zorder=10)

# 绘制模型曲线
ax.plot(fine_temps, quadratic_rates, 'r-', linewidth=2, 
       label=f'二次多项式模型 (R²={quadratic_r2:.4f})')
ax.plot(fine_temps, cardinal_rates, 'g-', linewidth=2, 
       label=f'Cardinal温度模型 (R²={cardinal_r2:.4f})')
ax.plot(fine_temps, ratkowsky_rates, 'b-', linewidth=2, 
       label=f'Ratkowsky模型 (R²={ratkowsky_r2:.4f})')

# 突出显示最适生长温度
try:
    # 找出Cardinal模型的最大值点
    max_idx = np.argmax(cardinal_rates)
    max_temp = fine_temps[max_idx]
    max_rate = cardinal_rates[max_idx]
    ax.scatter([max_temp], [max_rate], color='green', s=150, marker='*', 
              edgecolor='black', linewidth=1.5, zorder=11,
              label=f'最适生长温度 ({max_temp:.1f}°C)')
    ax.annotate(f'T_opt = {max_temp:.1f}°C\nr_max = {max_rate:.4f} h⁻¹', 
               xy=(max_temp, max_rate), xytext=(max_temp+3, max_rate),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
except:
    print("无法确定Cardinal模型的最大值点")

# 标记有效温度范围
if best_model == "Cardinal温度模型" or best_model == "Ratkowsky模型":
    min_T = T_min if best_model == "Cardinal温度模型" else T_min_rat
    max_T = T_max if best_model == "Cardinal温度模型" else T_max_rat
    ax.axvspan(min_T, max_T, alpha=0.2, color='yellow', label=f'有效温度范围 ({min_T:.1f}°C - {max_T:.1f}°C)')
    
    # 添加垂直线标记
    ax.axvline(x=min_T, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=max_T, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=T_opt if best_model == "Cardinal温度模型" else max_temp, color='green', linestyle='--', alpha=0.7)

# 设置坐标轴标签和标题
ax.set_xlabel('温度 (°C)', fontsize=14)
ax.set_ylabel('增殖速率 r (h⁻¹)', fontsize=14)
ax.set_title('温度(T)与病原细菌增殖速率(r)的关系模型比较', fontsize=16, fontweight='bold')

# 设置坐标轴范围和网格
ax.set_xlim(5, 40)
ax.set_ylim(0, max(max(quadratic_rates), max(cardinal_rates), max(ratkowsky_rates))*1.1)
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(fontsize=12, loc='upper right')

# 微调坐标轴刻度
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

# 最佳模型公式
if best_model == "二次多项式模型":
    equation = f'r(T) = {a:.6f}·T² + {b:.6f}·T + {c:.6f}'
    ax.text(0.05, 0.05, f"最佳模型: {best_model}\n{equation}", 
           transform=ax.transAxes, fontsize=12, 
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.5'))
elif best_model == "Cardinal温度模型":
    equation = f'r(T) = 特征温度模型\nT_min = {T_min:.2f}°C, T_opt = {T_opt:.2f}°C, T_max = {T_max:.2f}°C'
    ax.text(0.05, 0.05, f"最佳模型: {best_model}\n{equation}", 
           transform=ax.transAxes, fontsize=12, 
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.5'))

# 优化布局
plt.tight_layout()
plt.savefig(VISUALIZATION_PATHS['growth_rate_plot'], dpi=300, bbox_inches='tight')

# 绘制不同温度下的细菌生长曲线
plt.figure(figsize=(14, 10))

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
markers = ['o', 's', '^', 'd']

# 绘制实验数据点和拟合曲线
for i, temp in enumerate(temps):
    # 获取当前温度的细菌计数
    counts = bacteria_counts[temp]
    # 绘制实验数据点
    plt.semilogy(time_points, counts, marker=markers[i], markersize=8, color=colors[i], 
                linestyle='-', linewidth=2, label=f'{temp}°C 实验数据')
    
    # 根据最佳模型计算理论增长曲线
    if best_model == "二次多项式模型":
        r = quadratic_model(temp, *quadratic_params)
    elif best_model == "Cardinal温度模型":
        r = cardinal_model(temp, *cardinal_params)
    else:  # Ratkowsky模型
        r = ratkowsky_model(temp, *ratkowsky_params)
    
    # 使用微分方程解求解理论生长曲线
    # N(t) = N0 * e^(r*t)，这里N0为初始菌量
    N0 = counts[1]  # 使用t=3h时的菌量作为初始值，避免0值问题
    t0 = time_points[1]  # 对应的初始时间
    
    # 生成理论曲线时间点
    t_fine = np.linspace(t0, time_points[-1], 100)
    # 计算理论菌数
    N_theory = N0 * np.exp(r * (t_fine - t0))
    
    # 绘制理论曲线
    plt.semilogy(t_fine, N_theory, '--', linewidth=1.5, color=colors[i],
                label=f'{temp}°C 理论曲线 (r={r:.4f})')

# 添加标签和标题
plt.xlabel('时间 (h)', fontsize=14)
plt.ylabel('细菌数量 (CFU/mL)', fontsize=14)
plt.title('不同温度下病原细菌的生长曲线与模型拟合', fontsize=16, fontweight='bold')

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 优化图例
plt.legend(fontsize=12, loc='upper left')

# 优化布局
plt.tight_layout()

# 保存图像
plt.savefig(VISUALIZATION_PATHS['growth_curves'], dpi=300, bbox_inches='tight')

print("图表已生成完成！")