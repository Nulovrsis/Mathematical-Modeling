"""
气溶胶粒子散布可视化模块

本模块提供用于可视化气溶胶粒子散布特性的工具函数，专注于：
1. 粒子落地距离的统计分布分析
2. 粒子落地时间与散布距离的关系分析

这些可视化方法可以帮助更全面地理解气溶胶粒子的扩散特性和动力学行为。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    # 检查常见的中文字体路径
    font_paths = [
        # Windows 常见中文字体路径
        r'C:\Windows\Fonts\simhei.ttf',          # 黑体
        r'C:\Windows\Fonts\simsun.ttc',          # 宋体
        r'C:\Windows\Fonts\msyh.ttc',            # 微软雅黑
        r'C:\Windows\Fonts\STKAITI.TTF',         # 楷体
    ]
    
    # 尝试找到第一个存在的字体
    chinese_font_path = None
    for path in font_paths:
        if os.path.exists(path):
            chinese_font_path = path
            break
    
    # 如果找到了中文字体
    if chinese_font_path:
        # 创建字体属性对象
        font_prop = FontProperties(fname=chinese_font_path)
        # 设置默认字体
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop

# 初始化中文字体设置
chinese_font = setup_chinese_font()

def plot_distance_distribution(landing_positions, mean_distance, radius_95, effective_radius, title=None):
    """
    绘制粒子落地距离分布直方图
    
    参数:
    landing_positions: numpy.ndarray - 粒子落地位置数组，形状为 [n, 3]
    mean_distance: float - 平均散布距离
    radius_95: float - 95%置信区间半径
    effective_radius: float - 有效散布半径
    title: str - 图表标题，默认为None使用标准标题
    
    返回:
    fig: matplotlib.figure.Figure - 生成的图表对象
    """
    if landing_positions is None or len(landing_positions) == 0:
        print("错误: 没有有效的落地位置数据")
        return None
    
    # 过滤得到落地粒子的位置
    landed_positions = landing_positions[landing_positions[:, 2] == 0]
    
    if len(landed_positions) == 0:
        print("错误: 没有粒子落地")
        return None
    
    # 计算水平散布距离 (从原点)
    horizontal_distances = np.sqrt(landed_positions[:, 0]**2 + landed_positions[:, 1]**2)
    
    # 数值安全检查：移除任何无穷大或NaN值
    valid_distances = horizontal_distances[np.isfinite(horizontal_distances)]
    valid_distances = valid_distances[valid_distances <= 100.0]  # 限制最大合理距离
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制距离分布直方图
    n, bins, patches = ax.hist(valid_distances, bins=30, alpha=0.7, color='skyblue', 
                              edgecolor='black', density=True)
    
    # 添加核密度估计曲线
    kde = stats.gaussian_kde(valid_distances)
    x = np.linspace(0, np.max(valid_distances), 1000)
    ax.plot(x, kde(x), 'r-', linewidth=2, label='核密度估计')
    
    # 添加重要距离标记
    ax.axvline(mean_distance, color='g', linestyle='--', linewidth=2, 
              label=f'平均距离: {mean_distance:.2f} m')
    ax.axvline(radius_95, color='purple', linestyle='-.', linewidth=2,
              label=f'95%置信区间: {radius_95:.2f} m')
    ax.axvline(effective_radius, color='orange', linestyle=':', linewidth=2,
              label=f'有效散布半径: {effective_radius:.2f} m')
    
    # 设置坐标轴和标题，使用中文字体
    if chinese_font:
        ax.set_xlabel('散布距离 (m)', fontproperties=chinese_font)
        ax.set_ylabel('概率密度', fontproperties=chinese_font)
        if title:
            ax.set_title(title, fontproperties=chinese_font)
        else:
            ax.set_title('气溶胶粒子散布距离分布', fontproperties=chinese_font)
        legend = ax.legend(prop=chinese_font)
    else:
        ax.set_xlabel('Dispersion Distance (m)')
        ax.set_ylabel('Probability Density')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Aerosol Particle Dispersion Distance Distribution')
        ax.legend()
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_time_distance_scatter(landing_positions, landing_times, title=None):
    """
    绘制粒子落地时间与距离关系的散点图
    
    参数:
    landing_positions: numpy.ndarray - 粒子落地位置数组，形状为 [n, 3]
    landing_times: numpy.ndarray - 粒子落地时间数组，形状为 [n]
    title: str - 图表标题，默认为None使用标准标题
    
    返回:
    fig: matplotlib.figure.Figure - 生成的图表对象
    """
    if landing_positions is None or landing_times is None:
        print("错误: 没有有效的落地位置或时间数据")
        return None
    
    # 过滤得到落地粒子的位置和时间
    landed_indices = np.where(landing_positions[:, 2] == 0)[0]
    
    if len(landed_indices) == 0:
        print("错误: 没有粒子落地")
        return None
    
    landed_positions = landing_positions[landed_indices]
    landing_times = landing_times[landed_indices]
    
    # 计算水平散布距离 (从原点)
    horizontal_distances = np.sqrt(landed_positions[:, 0]**2 + landed_positions[:, 1]**2)
    
    # 数值安全检查：移除任何无穷大或NaN值
    valid_indices = np.isfinite(horizontal_distances) & np.isfinite(landing_times)
    valid_distances = horizontal_distances[valid_indices]
    valid_times = landing_times[valid_indices]
    
    # 限制最大合理值
    max_dist_mask = valid_distances <= 100.0
    valid_distances = valid_distances[max_dist_mask]
    valid_times = valid_times[max_dist_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点图，使用相同的颜色（蓝色）
    scatter = ax.scatter(valid_times, valid_distances, 
                       color='green', alpha=0.7, s=30, edgecolor='k')
    
    # 设置坐标轴和标题，使用中文字体
    if chinese_font:
        ax.set_xlabel('落地时间 (s)', fontproperties=chinese_font)
        ax.set_ylabel('散布距离 (m)', fontproperties=chinese_font)
        if title:
            ax.set_title(title, fontproperties=chinese_font)
        else:
            ax.set_title('气溶胶粒子落地时间与距离关系', fontproperties=chinese_font)
    else:
        ax.set_xlabel('Landing Time (s)')
        ax.set_ylabel('Dispersion Distance (m)')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Aerosol Particle Landing Time vs. Distance')
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def load_example_data():
    """
    加载示例数据用于演示
    
    返回:
    landing_positions: numpy.ndarray - 随机生成的粒子落地位置
    landing_times: numpy.ndarray - 随机生成的粒子落地时间
    mean_distance: float - 平均散布距离
    radius_95: float - 95%置信区间半径
    effective_radius: float - 有效散布半径
    """
    # 创建随机落地位置数据 (1000个粒子)
    np.random.seed(42)  # 设置随机种子以便结果可重现
    
    # 基本参数
    n_particles = 1000
    mean_dist = 2.0  # 平均散布距离
    std_dist = 0.4   # 标准差
    
    # 随机生成角度 (0-2π)
    angles = np.random.uniform(0, 2*np.pi, n_particles)
    
    # 使用高斯分布生成距离
    distances = np.random.normal(mean_dist, std_dist, n_particles)
    distances = np.abs(distances)  # 确保距离为正值
    
    # 转换为笛卡尔坐标
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    z = np.zeros(n_particles)  # 所有粒子都在z=0平面上
    
    # 组合为坐标数组
    landing_positions = np.column_stack((x, y, z))
    
    # 生成落地时间（与距离正相关，但有随机噪声）
    base_times = 0.5 + 0.7 * distances  # 基础时间与距离正相关
    noise = np.random.normal(0, 0.2, n_particles)  # 随机噪声
    landing_times = base_times + noise
    landing_times = np.maximum(landing_times, 0.1)  # 确保时间为正值
    
    # 计算统计指标
    horizontal_distances = np.sqrt(x**2 + y**2)
    mean_distance = np.mean(horizontal_distances)
    std_distance = np.std(horizontal_distances)
    radius_95 = np.percentile(horizontal_distances, 95)
    effective_radius = mean_distance + 3 * std_distance
    
    return landing_positions, landing_times, mean_distance, radius_95, effective_radius

def main():
    """主函数: 演示程序的使用"""
    print("气溶胶粒子散布可视化程序")
    print("------------------------")
    
    # 加载示例数据
    landing_positions, landing_times, mean_distance, radius_95, effective_radius = load_example_data()
    
    print(f"数据统计信息:")
    print(f"粒子数量: {len(landing_positions)}")
    print(f"平均散布距离: {mean_distance:.2f} m")
    print(f"95%置信区间半径: {radius_95:.2f} m")
    print(f"有效散布半径: {effective_radius:.2f} m")
    
    # 绘制距离分布直方图
    print("\n生成落地距离分布直方图...")
    distance_fig = plot_distance_distribution(
        landing_positions, mean_distance, radius_95, effective_radius)
    
    if distance_fig:
        distance_fig.savefig('distance_distribution.png', dpi=300, bbox_inches='tight')
        print("已保存到 distance_distribution.png")
    
    # 绘制时间-距离散点图
    print("\n生成落地时间与距离关系散点图...")
    time_distance_fig = plot_time_distance_scatter(landing_positions, landing_times)
    
    if time_distance_fig:
        time_distance_fig.savefig('time_distance_scatter.png', dpi=300, bbox_inches='tight')
        print("已保存到 time_distance_scatter.png")
    
    print("\n可视化完成！")

if __name__ == "__main__":
    main() 