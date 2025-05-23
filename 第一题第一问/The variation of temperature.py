import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import signal, stats
import sys
import os
import matplotlib

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入配置
from config import DATA_PATHS, VISUALIZATION_PATHS, OUTPUT_PATHS

# 动态加载SIMHEI.TTF字体
font_path = os.path.join(os.path.dirname(__file__), 'SIMHEI.TTF')
if os.path.exists(font_path):
    from matplotlib import font_manager as fm
    my_font = fm.FontProperties(fname=font_path)
    font_name = my_font.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.sans-serif'] = [font_name]
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("已强制使用SIMHEI.TTF字体，当前字体:", font_name)
else:
    print("警告：未找到SIMHEI.TTF，中文可能无法正常显示！")

print(matplotlib.rcParams['font.sans-serif'])

# 让matplotlib支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.size'] = 12               # 设置默认字体大小
plt.style.use('ggplot')                      # 使用ggplot风格美化图表

# 1. 数据读取与预处理
print("开始数据预处理...")
# 使用配置中的温度数据路径
temperature_file = DATA_PATHS['temperature_data']
encodings = ['utf-8', 'gbk', 'utf-8-sig', 'latin1']

try:
    # 尝试从Excel文件读取数据
    df = pd.read_excel(temperature_file, header=1)
    print(f"成功读取Excel文件: {temperature_file}")
except Exception as e:
    print(f"尝试读取Excel文件失败: {e}")
    print("尝试以CSV格式读取文件...")
    # 如果Excel读取失败，尝试以CSV格式读取
    for enc in encodings:
        try:
            df = pd.read_csv(temperature_file, encoding=enc)
            print(f"成功使用编码 {enc} 读取CSV文件。")
            break
        except Exception as e:
            print(f"尝试编码 {enc} 失败：{e}")
    else:
        raise ValueError(f"无法读取文件 {temperature_file}，请检查文件格式或内容。")

# 读取数据后，输出实际列名
print("实际读取到的列名：", df.columns.tolist())
# 自动查找并重命名温度列
for col in df.columns:
    if '温度' in col:
        df = df.rename(columns={col: '温度'})
        break

# 时间标准化：转为分钟数
def time_to_minutes(tstr):
    if pd.isna(tstr) or not isinstance(tstr, str):
        return np.nan
    try:
        t = datetime.strptime(tstr.strip(), '%H:%M:%S')
        return t.hour * 60 + t.minute + t.second / 60
    except ValueError:
        print(f"时间格式错误: '{tstr}'")
        return np.nan

# 应用转换并移除无效时间
df['分钟'] = df['时间'].apply(time_to_minutes)
df = df.dropna(subset=['分钟']).reset_index(drop=True)

# 2. 异常值检测与处理
print("进行异常值检测与处理...")
# Z-score方法检测异常值
z_scores = stats.zscore(df['温度'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)  # 标准Z分数阈值为3

# 标记异常值
df['异常温度'] = ~filtered_entries

# 使用多种平滑方法处理数据
# 移动平均平滑
df['温度_移动平均'] = df['温度'].rolling(window=5, center=True).mean().fillna(df['温度'])

# Savitzky-Golay滤波平滑（更适合保留峰值特征）
df['温度_SG平滑'] = signal.savgol_filter(df['温度'], window_length=11, polyorder=3)

# 3. 基础统计特征
print("计算基础统计特征...")
T = df['温度_SG平滑']  # 使用SG平滑后的温度
T_mean = T.mean()
T_median = T.median()
T_max = T.max()
T_min = T.min()
T_max_time = df.loc[T.idxmax(), '时间']
T_min_time = df.loc[T.idxmin(), '时间']
T_range = T_max - T_min
T_std = T.std()
T_skew = stats.skew(T)  # 偏度
T_kurtosis = stats.kurtosis(T)  # 峰度

print(f'均值: {T_mean:.2f}℃, 中位数: {T_median:.2f}℃')
print(f'最大值: {T_max:.2f}℃ ({T_max_time}), 最小值: {T_min:.2f}℃ ({T_min_time}), 极差: {T_range:.2f}℃')
print(f'标准差: {T_std:.2f}℃, 偏度: {T_skew:.2f}, 峰度: {T_kurtosis:.2f}')

# 4. 昼夜分段统计与分析
print("进行昼夜分段统计分析...")
# 按照日出日落时间划分白天和夜晚（可根据实际情况调整）
df['时段'] = df['分钟'].apply(lambda x: '白天' if 360 <= x < 1080 else '夜间')  # 6:00-18:00为白天
day_T = df[df['时段']=='白天']['温度_SG平滑']
night_T = df[df['时段']=='夜间']['温度_SG平滑']

# 昼夜统计比较
print(f'白天均值: {day_T.mean():.2f}℃, 夜间均值: {night_T.mean():.2f}℃, 昼夜温差: {day_T.mean()-night_T.mean():.2f}℃')
print(f'白天标准差: {day_T.std():.2f}℃, 夜间标准差: {night_T.std():.2f}℃')
print(f'白天最高温度: {day_T.max():.2f}℃, 夜间最低温度: {night_T.min():.2f}℃')

# 计算升温/降温速率
try:
    # 寻找最接近特定时间点的数据行
    def find_nearest_time(minutes_val):
        idx = (df['分钟'] - minutes_val).abs().idxmin()
        return df.loc[idx]
    
    T_6 = find_nearest_time(360)['温度_SG平滑']
    T_12 = find_nearest_time(720)['温度_SG平滑']
    T_14 = find_nearest_time(840)['温度_SG平滑']
    T_24 = find_nearest_time(1440)['温度_SG平滑']
    
    k_rise = (T_12 - T_6) / 6
    k_fall = (T_24 - T_14) / 10
    print(f'升温速率: {k_rise:.2f}℃/小时, 降温速率: {k_fall:.2f}℃/小时')
    
    # 计算温度变化率（导数）
    df['温度变化率'] = df['温度_SG平滑'].diff() / df['分钟'].diff() * 60  # 每小时温度变化
    print(f'平均升温率: {df[df["温度变化率"] > 0]["温度变化率"].mean():.2f}℃/小时')
    print(f'平均降温率: {df[df["温度变化率"] < 0]["温度变化率"].mean():.2f}℃/小时')
except Exception as e:
    print("计算升降温率时出错：", e)

# 5. 温度波动分析
print("分析温度波动特性...")
df['温度波动'] = df['温度_SG平滑'].diff().abs()
avg_fluctuation = df['温度波动'].mean()
max_fluctuation = df['温度波动'].max()
print(f'平均温度波动: {avg_fluctuation:.3f}℃, 最大温度波动: {max_fluctuation:.3f}℃')

# 分析温度周期性
# 傅里叶变换分析周期
if len(df) > 50:  # 确保有足够的数据点进行傅里叶分析
    temp_data = df['温度_SG平滑'].values
    # 去除趋势
    detrended = signal.detrend(temp_data)
    # 执行FFT
    fft_data = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(len(detrended), d=5/60)  # 采样间隔为5分钟
    # 找出主要频率
    pos_mask = freqs > 0
    peak_idx = np.argmax(np.abs(fft_data[pos_mask]))
    peak_freq = freqs[pos_mask][peak_idx]
    period = 1/peak_freq if peak_freq != 0 else 0
    print(f'主要温度周期: {period:.2f} 小时')

# 6. 温度模型拟合
print("拟合温度变化模型...")
# 定义更复杂的温度变化模型函数（带有日变化和其他扰动）
def temp_model(x, A, phi, mu, A2, phi2, k):
    """复合正弦模型，考虑主周期、次周期和线性趋势"""
    return A * np.sin(2 * np.pi * x / 24 + phi) + A2 * np.sin(2 * np.pi * x / 12 + phi2) + k * x + mu

try:
    # 使用更强大的全局优化方法拟合参数
    bounds = ([0, -np.pi, 10, 0, -np.pi, -1], [10, np.pi, 30, 5, np.pi, 1])  # 参数范围
    popt, pcov = curve_fit(temp_model, df['分钟']/60, df['温度_SG平滑'], 
                          p0=[5, 0, T_mean, 1, 0, 0], bounds=bounds, maxfev=5000)
    
    A, phi, mu, A2, phi2, k = popt
    print(f'温度模型拟合参数：')
    print(f'- 主周期振幅A={A:.2f}, 相位phi={phi:.2f}, 均值mu={mu:.2f}')
    print(f'- 次周期振幅A2={A2:.2f}, 相位phi2={phi2:.2f}, 线性趋势k={k:.4f}')
    
    # 计算拟合优度
    y_pred = temp_model(df['分钟']/60, *popt)
    residuals = df['温度_SG平滑'] - y_pred
    ss_tot = np.sum((df['温度_SG平滑'] - df['温度_SG平滑'].mean())**2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'拟合优度 R²: {r_squared:.4f}')
    
    # 计算残差标准误
    rmse = np.sqrt(np.mean(residuals**2))
    print(f'残差标准误: {rmse:.3f}℃')
    
    # 简单的模型函数（用于后续的预测）
    def predict_temperature(hour):
        """预测指定小时的温度"""
        return temp_model(hour, A, phi, mu, A2, phi2, k)
    
    # 保存模型参数到文件，方便后续使用
    np.savez(OUTPUT_PATHS['temperature_model_params'], A=A, phi=phi, mu=mu, A2=A2, phi2=phi2, k=k)
    print(f"温度模型参数已保存到 '{OUTPUT_PATHS['temperature_model_params']}'")
    
except Exception as e:
    print(f"温度模型拟合失败: {e}")
    # 使用简单正弦模型作为备选
    try:
        simple_model = lambda x, A, phi, mu: A * np.sin(2 * np.pi * x / 24 + phi) + mu
        popt, _ = curve_fit(simple_model, df['分钟']/60, df['温度_SG平滑'], p0=[5, 0, T_mean])
        print(f'简单正弦拟合参数：振幅A={popt[0]:.2f}, 相位phi={popt[1]:.2f}, 均值mu={popt[2]:.2f}')
        
        # 定义简单预测函数
        def predict_temperature(hour):
            """使用简单模型预测温度"""
            return simple_model(hour, *popt)
        
        A, phi, mu = popt
        A2, phi2, k = 0, 0, 0  # 设置额外参数为0
        # 保存备选模型参数
        np.savez(OUTPUT_PATHS['temperature_model_params'], A=A, phi=phi, mu=mu, A2=0, phi2=0, k=0)
    except:
        print("备选模型拟合也失败，无法建立温度预测模型")

# 7. 数据可视化
print("生成可视化图表...")
# 设置更美观的风格
plt.style.use('seaborn-v0_8-whitegrid')

# 7.1 温度时序图（含原始、平滑、异常值标记和拟合模型）
plt.figure(figsize=(14, 8))
plt.plot(df['分钟']/60, df['温度'], 'o', markersize=3, alpha=0.3, label='原始温度数据')
plt.plot(df['分钟']/60, df['温度_SG平滑'], '-', linewidth=2, label='Savitzky-Golay平滑温度')
plt.plot(df['分钟']/60, df['温度_移动平均'], '--', alpha=0.7, label='移动平均温度')

# 标记异常值
if '异常温度' in df.columns and any(df['异常温度']):
    plt.scatter(df[df['异常温度']]['分钟']/60, df[df['异常温度']]['温度'], 
               color='red', marker='x', s=80, label='异常值')

# 加入拟合模型曲线
try:
    x_smooth = np.linspace(0, 24, 1000)
    plt.plot(x_smooth, temp_model(x_smooth, A, phi, mu, A2, phi2, k), 
             'r-', linewidth=2.5, label='温度拟合模型')
except:
    print("无法绘制拟合模型曲线")

# 标注昼夜时段
plt.axvspan(0, 6, color='gray', alpha=0.2, label='夜间')
plt.axvspan(6, 18, color='yellow', alpha=0.1, label='白天')
plt.axvspan(18, 24, color='gray', alpha=0.2)

# 标注最高和最低温度点
plt.scatter([df.loc[T.idxmax(), '分钟']/60], [T_max], color='red', s=100, 
           marker='*', label=f'最高温度 ({T_max:.1f}℃)')
plt.scatter([df.loc[T.idxmin(), '分钟']/60], [T_min], color='blue', s=100,
           marker='*', label=f'最低温度 ({T_min:.1f}℃)')

# 添加注释
plt.annotate(f'{T_max:.1f}℃', (df.loc[T.idxmax(), '分钟']/60, T_max),
           xytext=(10, 10), textcoords='offset points', fontsize=12)
plt.annotate(f'{T_min:.1f}℃', (df.loc[T.idxmin(), '分钟']/60, T_min),
           xytext=(10, -15), textcoords='offset points', fontsize=12)

# 优化图表布局和标签
plt.xlabel('时间（小时）', fontsize=14)
plt.ylabel('温度（℃）', fontsize=14)
plt.title('温室24小时温度变化详细分析', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=10)
plt.xticks(np.arange(0, 25, 3))
plt.xlim(0, 24)
plt.tight_layout()
plt.savefig(VISUALIZATION_PATHS['temperature_time_plot'], dpi=300, bbox_inches='tight')

# 7.2 温度分布直方图（含正态分布拟合）
plt.figure(figsize=(10, 6))
# 计算直方图数据
hist, bins, _ = plt.hist(T, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# 拟合正态分布
mu, std = stats.norm.fit(T)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'正态分布拟合 (μ={mu:.2f}, σ={std:.2f})')

# 执行正态分布检验
stat, p_norm = stats.shapiro(T)
norm_test_result = "符合" if p_norm > 0.05 else "不符合"
plt.text(0.05, 0.9, f'Shapiro-Wilk检验: {norm_test_result}正态分布 (p={p_norm:.4f})',
        transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('温度（℃）', fontsize=14)
plt.ylabel('频率密度', fontsize=14)
plt.title('温度分布直方图与正态分布拟合', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(VISUALIZATION_PATHS['temperature_distribution'], dpi=300, bbox_inches='tight')

# 7.3 温度拟合模型图
plt.figure(figsize=(12, 6))
plt.scatter(df['分钟']/60, df['温度_SG平滑'], s=30, alpha=0.6, label='平滑温度数据')

try:
    # 绘制拟合曲线
    x_fit = np.linspace(0, 24, 1000)
    plt.plot(x_fit, temp_model(x_fit, A, phi, mu, A2, phi2, k), 
             'r-', linewidth=3, label='复合正弦拟合模型')
    
    # 分解模型组件
    main_component = A * np.sin(2 * np.pi * x_fit / 24 + phi) + mu
    secondary_component = A2 * np.sin(2 * np.pi * x_fit / 12 + phi2)
    trend_component = k * x_fit
    
    # 绘制主要组件
    plt.plot(x_fit, main_component, '--', color='green', alpha=0.7, linewidth=2,
             label='主周期成分')
    
    # 如果次要组件显著，则绘制
    if A2 > 0.2:
        plt.plot(x_fit, secondary_component + mu, '--', color='purple', alpha=0.7,
                linewidth=2, label='次周期成分')
    
    # 如果存在明显趋势，则绘制
    if abs(k) > 0.01:
        plt.plot(x_fit, trend_component + mu, '--', color='blue', alpha=0.7,
                linewidth=2, label='线性趋势')
    
    # 添加拟合公式
    equation = f'T(t) = {A:.2f}·sin(2πt/24 + {phi:.2f}) + {A2:.2f}·sin(2πt/12 + {phi2:.2f}) + {k:.4f}·t + {mu:.2f}'
    plt.annotate(equation, xy=(0.5, 0.03), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='center', fontsize=12)
    
    # 添加拟合优度
    plt.annotate(f'R² = {r_squared:.4f}, RMSE = {rmse:.3f}℃', xy=(0.5, 0.1), 
                xycoords='axes fraction', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
except Exception as e:
    print(f"绘制拟合模型图时出错: {e}")

plt.xlabel('时间（小时）', fontsize=14)
plt.ylabel('温度（℃）', fontsize=14)
plt.title('温度变化拟合模型及其组成成分', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.xlim(0, 24)
plt.tight_layout()
plt.savefig(VISUALIZATION_PATHS['temperature_sine_fit'], dpi=300, bbox_inches='tight')

print("温度分析完成！")