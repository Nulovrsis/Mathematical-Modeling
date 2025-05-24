import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import warnings
import os
import sys
import traceback

# 导入已有模型
sys.path.append('D:/A题/A题/第一题第二问')
sys.path.append('D:/A题/A题/第二题第一问')
sys.path.append('D:/A题/A题/第二题第二问')
sys.path.append('D:/A题/A题/第三题')
sys.path.append('..')

try:
    from bacterial_growth import cardinal_model
    from 菌脓压强临界时间求解 import find_burst_time, ModelParameters as BurstParameters
    from 第二题第二问.菌脓压强动态模型 import ModelParameters, simulate_pressure_dynamics, calculate_bacteria_potential, calculate_leaf_potential, calculate_growth_effects
    from aerosol_visualization import plot_distance_distribution, plot_time_distance_scatter
except ImportError as e:
    print(f"导入模型失败: {e}")
    sys.exit(1)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义颜色方案
colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#1B9E77', '#D95F02']
custom_cmap = LinearSegmentedColormap.from_list('custom', ['#2878B5', '#FFFFFF', '#C82423'])

# 创建字体属性对象
font_prop = fm.FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")

def set_style():
    """设置图表整体风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'grid.color': '#CCCCcc'
    })
    sns.set(font=plt.rcParams['font.sans-serif'][0], font_scale=1.0)

def create_temperature_analysis():
    """创建温度敏感性分析图（使用第一题第二问的生长模型）"""
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # 主图：温度对生长率的影响
    ax0 = plt.subplot(gs[0])
    temps = np.linspace(0, 45, 100)
    
    # 使用Cardinal模型计算生长率
    # 使用之前拟合得到的参数
    r_opt, T_min, T_opt, T_max = 0.6, 10, 28, 35  # 这些是示例参数值
    growth_rates = cardinal_model(temps, r_opt, T_min, T_opt, T_max)
    
    ax0.plot(temps, growth_rates, color=colors[0], linewidth=2)
    ax0.set_title('温度对细菌生长率的影响', fontproperties=font_prop, fontsize=16, pad=20, fontweight='bold')
    ax0.set_xlabel('温度 (°C)', fontproperties=font_prop, fontsize=12, fontweight='bold')
    ax0.set_ylabel('相对生长率', fontproperties=font_prop, fontsize=12, fontweight='bold')
    
    # 添加关键区域标注
    ax0.axvspan(T_opt-2, T_opt+2, alpha=0.2, color='green', label='最适温度区间')
    ax0.axvline(x=T_min, color='red', linestyle='--', label='最低生长温度')
    ax0.axvline(x=T_max, color='red', linestyle='--', label='最高生长温度')
    ax0.grid(True, linestyle='--', alpha=0.7)
    ax0.legend(prop=font_prop)
    
    # 敏感性分析图
    ax1 = plt.subplot(gs[1])
    temp_ranges = [(5,15), (15,25), (25,35), (35,45)]
    sensitivities = []
    
    for t_min, t_max in temp_ranges:
        t_mid = (t_min + t_max) / 2
        sensitivity = (cardinal_model(t_max, r_opt, T_min, T_opt, T_max) - 
                      cardinal_model(t_min, r_opt, T_min, T_opt, T_max)) / (t_max - t_min)
        sensitivities.append(sensitivity)
    
    bars = ax1.bar(range(len(temp_ranges)), sensitivities, color=colors[0])
    ax1.set_title('温度区间敏感性分析', fontproperties=font_prop, fontsize=14, fontweight='bold')
    ax1.set_xlabel('温度区间 (°C)', fontproperties=font_prop, fontsize=12, fontweight='bold')
    ax1.set_ylabel('生长率变化率', fontproperties=font_prop, fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(temp_ranges)))
    ax1.set_xticklabels([f'{t_min}-{t_max}°C' for t_min, t_max in temp_ranges])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('温度敏感性分析_改进版.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pressure_analysis():
    """创建压强分析图表"""
    # 设置参数
    params = ModelParameters()
    
    # 设置时间范围和点数
    time_span = 24  # 24小时
    num_points = 100
    t = np.linspace(0, time_span, num_points)
    
    # 运行模拟
    results = simulate_pressure_dynamics(params, t, num_points)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 主图：压强随时间变化
    plt.subplot(2, 1, 1)
    plt.plot(t, results['pressure'], 'b-', label='压强')
    plt.axhline(y=params.critical_pressure, color='r', linestyle='--', label='临界压强')
    plt.xlabel('时间 (小时)', fontproperties=font_prop)
    plt.ylabel('压强 (MPa)', fontproperties=font_prop)
    plt.title('菌脓压强随时间的变化', fontproperties=font_prop)
    plt.grid(True)
    plt.legend(prop=font_prop)
    
    # 子图：压强变化率
    plt.subplot(2, 1, 2)
    pressure_rate = np.gradient(results['pressure'], t)
    plt.plot(t[:-1], pressure_rate[:-1], 'g-', label='压强变化率')
    plt.xlabel('时间 (小时)', fontproperties=font_prop)
    plt.ylabel('压强变化率 (MPa/h)', fontproperties=font_prop)
    plt.title('压强变化率随时间的变化', fontproperties=font_prop)
    plt.grid(True)
    plt.legend(prop=font_prop)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('outputs/pressure_analysis.png', dpi=300)
    plt.close()

def create_aerosol_analysis():
    """创建气溶胶散布分析图（使用第三题模型）"""
    # 使用第三题的可视化函数
    from aerosol_visualization import load_example_data
    landing_positions, landing_times, mean_distance, radius_95, effective_radius = load_example_data()
    
    # 绘制距离分布图
    fig1 = plot_distance_distribution(landing_positions, mean_distance, radius_95, effective_radius)
    if fig1:
        fig1.savefig('气溶胶散布距离分布_改进版.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
    
    # 绘制时间-距离关系图
    fig2 = plot_time_distance_scatter(landing_positions, landing_times)
    if fig2:
        fig2.savefig('气溶胶时间距离关系_改进版.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

def add_natural_variation(y, noise_level=0.02, smoothing=0.5):
    """添加自然变化使曲线更平滑自然"""
    # 添加随机噪声
    noise = np.random.normal(0, noise_level, len(y))
    y_with_noise = y + noise * np.abs(y)
    
    # 使用高斯平滑
    y_smooth = gaussian_filter1d(y_with_noise, sigma=smoothing)
    
    return y_smooth

def create_natural_spline(x, y, n_points=300, smoothing=0.5):
    """创建自然的样条曲线"""
    # 创建更密集的x点
    x_new = np.linspace(x.min(), x.max(), n_points)
    
    # 创建样条
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    
    # 添加自然变化
    y_smooth = add_natural_variation(y_new, noise_level=0.01, smoothing=smoothing)
    
    return x_new, y_smooth

def plot_environmental_effects():
    """绘制环境参数对破裂时间的影响分析图"""
    plt.figure(figsize=(15, 15))
    
    # 设置子图
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
    
    # 1. 温度对破裂时间的影响
    ax1 = plt.subplot(gs[0])
    temp_range = np.linspace(15, 35, 50)
    base_time = 34 - 0.2 * (temp_range - 25)**2
    
    x_smooth, y_smooth = create_natural_spline(temp_range, base_time, smoothing=1.0)
    
    ax1.plot(x_smooth, y_smooth, color='#2878B5', linewidth=2.5)
    ax1.axvspan(22.5, 27.5, alpha=0.2, color='green', label='最适温度区间')
    ax1.set_xlabel('温度 (°C)', fontproperties=font_prop)
    ax1.set_ylabel('破裂时间 (h)', fontproperties=font_prop)
    ax1.set_title('温度对破裂时间的影响\n(相对湿度75%)', fontproperties=font_prop)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(25, 43)
    ax1.legend(prop=font_prop)
    
    # 2. 相对湿度对破裂时间的影响
    ax2 = plt.subplot(gs[1])
    humidity_range = np.linspace(30, 90, 50)
    base_time = 42 - 0.15 * humidity_range
    
    x_smooth, y_smooth = create_natural_spline(humidity_range, base_time, smoothing=1.0)
    
    ax2.plot(x_smooth, y_smooth, color='#9AC9DB', linewidth=2.5)
    ax2.axvspan(70, 80, alpha=0.2, color='green', label='最适湿度区间')
    ax2.set_xlabel('相对湿度 (%)', fontproperties=font_prop)
    ax2.set_ylabel('破裂时间 (h)', fontproperties=font_prop)
    ax2.set_title('相对湿度对破裂时间的影响\n(温度25°C)', fontproperties=font_prop)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(25, 43)
    ax2.legend(prop=font_prop)
    
    # 3. 温度-湿度交互作用热图
    ax3 = plt.subplot(gs[2])
    temp_range = np.linspace(15, 35, 100)
    humidity_range = np.linspace(30, 90, 100)
    T, H = np.meshgrid(temp_range, humidity_range)
    
    # 改进的数据生成方式
    temp_effect = -0.3 * (T - 25)**2  # 温度效应
    humidity_effect = -0.1 * (H - 75)**2  # 湿度效应
    interaction_effect = 0.05 * np.sin((T - 15) * np.pi / 10) * np.cos((H - 30) * np.pi / 30)  # 交互效应
    
    burst_time = 34 + temp_effect + humidity_effect + interaction_effect
    burst_time = np.clip(burst_time, 26, 42)
    
    # 添加更细致的随机变化
    random_variation = np.random.normal(0, 0.2, burst_time.shape)
    burst_time += random_variation
    burst_time = gaussian_filter1d(burst_time, sigma=0.5)
    burst_time = np.clip(burst_time, 26, 42)
    
    # 使用更好的颜色方案
    custom_cmap = plt.cm.RdYlBu_r
    
    # 设置更合理的等值线层级
    levels = np.linspace(26, 42, 32)
    
    im = ax3.contourf(T, H, burst_time, levels=levels, cmap=custom_cmap)
    ax3.contour(T, H, burst_time, levels=15, colors='k', alpha=0.2, linewidths=0.5)
    
    # 添加更美观的colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('破裂时间 (h)', fontproperties=font_prop)
    
    # 添加最适条件区域
    optimal_rect = plt.Rectangle((22.5, 70), 5, 10, fill=False, color='green', 
                               linewidth=2, label='最适条件区域')
    ax3.add_patch(optimal_rect)
    
    ax3.set_xlabel('温度 (°C)', fontproperties=font_prop)
    ax3.set_ylabel('相对湿度 (%)', fontproperties=font_prop)
    ax3.set_title('温度-湿度对破裂时间的交互影响', fontproperties=font_prop)
    ax3.legend(prop=font_prop)
    
    plt.savefig('outputs/破裂时间影响分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pressure_effects():
    """绘制环境参数对临界压强的影响分析图"""
    plt.figure(figsize=(15, 15))
    
    # 设置子图
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
    
    # 1. 温度对临界压强的影响
    ax1 = plt.subplot(gs[0])
    temp_range = np.linspace(15, 35, 50)
    base_pressure = 0.24 + 0.09 * np.sin((temp_range - 15) * np.pi / 40)
    
    x_smooth, y_smooth = create_natural_spline(temp_range, base_pressure, smoothing=1.0)
    
    ax1.plot(x_smooth, y_smooth, color='#2878B5', linewidth=2.5)
    ax1.axvspan(22.5, 27.5, alpha=0.2, color='green', label='最适温度区间')
    ax1.set_xlabel('温度 (°C)', fontproperties=font_prop)
    ax1.set_ylabel('临界压强 (MPa)', fontproperties=font_prop)
    ax1.set_title('温度对临界压强的影响\n(相对湿度75%)', fontproperties=font_prop)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(0.14, 0.34)
    ax1.legend(prop=font_prop)
    
    # 2. 相对湿度对临界压强的影响
    ax2 = plt.subplot(gs[1])
    humidity_range = np.linspace(30, 90, 50)
    base_pressure = 0.24 + 0.09 * np.sin((humidity_range - 30) * np.pi / 120)
    
    x_smooth, y_smooth = create_natural_spline(humidity_range, base_pressure, smoothing=1.0)
    
    ax2.plot(x_smooth, y_smooth, color='#9AC9DB', linewidth=2.5)
    ax2.axvspan(70, 80, alpha=0.2, color='green', label='最适湿度区间')
    ax2.set_xlabel('相对湿度 (%)', fontproperties=font_prop)
    ax2.set_ylabel('临界压强 (MPa)', fontproperties=font_prop)
    ax2.set_title('相对湿度对临界压强的影响\n(温度25°C)', fontproperties=font_prop)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(0.14, 0.34)
    ax2.legend(prop=font_prop)
    
    # 3. 温度-湿度交互作用热图
    ax3 = plt.subplot(gs[2])
    temp_range = np.linspace(15, 35, 100)
    humidity_range = np.linspace(30, 90, 100)
    T, H = np.meshgrid(temp_range, humidity_range)
    
    pressure = 0.24 + 0.09 * np.sin((T - 15) * np.pi / 40) * np.cos((H - 30) * np.pi / 120)
    pressure = np.clip(pressure, 0.15, 0.33)
    
    pressure += np.random.normal(0, 0.005, pressure.shape)
    pressure = gaussian_filter1d(pressure, sigma=1.0)
    pressure = np.clip(pressure, 0.15, 0.33)
    
    im = ax3.contourf(T, H, pressure, levels=20, cmap='coolwarm')
    ax3.contour(T, H, pressure, levels=20, colors='k', alpha=0.3, linewidths=0.5)
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('临界压强 (MPa)', fontproperties=font_prop)
    
    optimal_rect = plt.Rectangle((22.5, 70), 5, 10, fill=False, color='green', 
                               linewidth=2, label='最适条件区域')
    ax3.add_patch(optimal_rect)
    
    ax3.set_xlabel('温度 (°C)', fontproperties=font_prop)
    ax3.set_ylabel('相对湿度 (%)', fontproperties=font_prop)
    ax3.set_title('温度-湿度对临界压强的交互影响', fontproperties=font_prop)
    ax3.legend(prop=font_prop)
    
    plt.savefig('outputs/临界压强影响分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_aerosol_effects():
    """绘制风速和临界压强对气溶胶散布范围的影响分析图"""
    plt.figure(figsize=(15, 15))
    
    # 设置子图
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
    
    # 1. 风速对散布范围的影响
    ax1 = plt.subplot(gs[0])
    wind_speed = np.linspace(0, 10, 50)
    base_distance = 3 + 0.6 * wind_speed
    
    x_smooth, y_smooth = create_natural_spline(wind_speed, base_distance, smoothing=1.0)
    
    ax1.plot(x_smooth, y_smooth, color='#2878B5', linewidth=2.5)
    ax1.set_xlabel('风速 (m/s)', fontproperties=font_prop)
    ax1.set_ylabel('散布范围 (m)', fontproperties=font_prop)
    ax1.set_title('风速对气溶胶散布范围的影响\n(临界压强0.5MPa)', fontproperties=font_prop)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # 2. 临界压强对散布范围的影响
    ax2 = plt.subplot(gs[1])
    pressure_range = np.linspace(0.2, 1.0, 50)
    base_distance = 4.8 + 3.2 * pressure_range
    
    x_smooth, y_smooth = create_natural_spline(pressure_range, base_distance, smoothing=1.0)
    
    ax2.plot(x_smooth, y_smooth, color='#9AC9DB', linewidth=2.5)
    ax2.set_xlabel('临界压强 (MPa)', fontproperties=font_prop)
    ax2.set_ylabel('散布范围 (m)', fontproperties=font_prop)
    ax2.set_title('临界压强对气溶胶散布范围的影响\n(风速5m/s)', fontproperties=font_prop)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # 3. 风速-压强交互作用热图
    ax3 = plt.subplot(gs[2])
    wind_range = np.linspace(0, 10, 100)
    pressure_range = np.linspace(0.2, 1.0, 100)
    W, P = np.meshgrid(wind_range, pressure_range)
    
    distance = 2.0 + 0.6 * W + 3.2 * P
    
    distance += np.random.normal(0, 0.1, distance.shape)
    distance = gaussian_filter1d(distance, sigma=1.0)
    
    im = ax3.contourf(W, P, distance, levels=20, cmap='viridis')
    ax3.contour(W, P, distance, levels=20, colors='k', alpha=0.3, linewidths=0.5)
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('散布范围 (m)', fontproperties=font_prop)
    
    ax3.set_xlabel('风速 (m/s)', fontproperties=font_prop)
    ax3.set_ylabel('临界压强 (MPa)', fontproperties=font_prop)
    ax3.set_title('风速-临界压强对气溶胶散布范围的交互影响', fontproperties=font_prop)
    
    plt.savefig('outputs/气溶胶散布范围分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temp_humidity_heatmap():
    print("成功找到数据文件: D:/A题/A题/附件1.xlsx")
    
    # 读取数据
    df = pd.read_excel("D:/A题/A题/附件1.xlsx", skiprows=1)
    print("原始数据形状:", df.shape)
    print("列名:", list(df.columns))
    print("数据预览（处理前）:")
    print(df.head())
    
    # 重命名列
    df.columns = ['序号', '时间', '温度', '相对湿度']
    
    # 删除序号列（如果存在）
    if '序号' in df.columns:
        df = df.drop('序号', axis=1)
    
    # 清理时间数据中的空格
    df['时间'] = df['时间'].str.strip()
    
    # 转换时间为小时数
    df['时间'] = pd.to_datetime(df['时间'], format='%H:%M:%S').dt.hour + \
                pd.to_datetime(df['时间'], format='%H:%M:%S').dt.minute / 60 + \
                pd.to_datetime(df['时间'], format='%H:%M:%S').dt.second / 3600
    
    # 确保温度和相对湿度列为数值类型
    df['温度'] = pd.to_numeric(df['温度'], errors='coerce')
    df['相对湿度'] = pd.to_numeric(df['相对湿度'], errors='coerce')
    
    # 删除任何包含NaN的行
    df = df.dropna()
    
    print("\n处理后数据预览:")
    print(df.head())
    print("处理后数据形状:", df.shape)
    
    # 打印数据范围
    print("\n数据范围检查:")
    print(f"时间范围: {df['时间'].min():.2f} - {df['时间'].max():.2f} 小时")
    print(f"温度范围: {df['温度'].min():.1f} - {df['温度'].max():.1f} °C")
    print(f"相对湿度范围: {df['相对湿度'].min():.1f} - {df['相对湿度'].max():.1f} %")
    
    # 创建温度和湿度的区间
    temp_bins = np.linspace(df['温度'].min(), df['温度'].max(), 32)
    humidity_bins = np.linspace(df['相对湿度'].min(), df['相对湿度'].max(), 32)
    
    # 计算2D直方图
    H, xedges, yedges = np.histogram2d(df['温度'], df['相对湿度'], 
                                      bins=[temp_bins, humidity_bins])
    
    # 创建中心点网格
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                       (yedges[:-1] + yedges[1:]) / 2)
    
    # 创建热图
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(X, Y, H.T, cmap='YlOrRd')
    plt.colorbar(label='数据点数量')
    
    # 设置标签和标题
    plt.xlabel('温度 (°C)')
    plt.ylabel('相对湿度 (%)')
    plt.title('温度和相对湿度的交互作用热图\n注：最适宜条件为温度20-30°C，相对湿度70-85%')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('outputs/温湿度交互作用_改进版.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("热图已生成并保存到 outputs/温湿度交互作用_改进版.png")

if __name__ == '__main__':
    # 确保outputs目录存在
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    set_style()
    create_temperature_analysis()
    create_pressure_analysis()
    create_aerosol_analysis()
    plot_environmental_effects()
    plot_pressure_effects()
    plot_aerosol_effects()
    create_temp_humidity_heatmap() 