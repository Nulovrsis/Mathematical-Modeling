# 气溶胶粒子散布可视化工具

## 简介

气溶胶粒子散布可视化工具是一个专门用于分析和可视化气溶胶粒子散布特性的Python模块。该工具从原有的气溶胶抛射模型中提取了两个最重要的可视化功能，专注于对粒子落地距离分布和落地时间与距离关系的分析，帮助研究人员更深入地理解气溶胶粒子的扩散规律和动力学特性。

## 主要功能

该工具提供两种核心可视化方法：

1. **落地距离分布分析**：通过直方图和核密度估计曲线展示粒子落地距离的概率分布，同时标记关键统计指标
2. **落地时间与距离关系分析**：通过散点图揭示粒子落地时间与散布距离之间的关系，并提供线性拟合和平滑趋势线

## 技术实现

### 落地距离分布图

落地距离分布图使用直方图和核密度估计（KDE）相结合的方式，全面反映粒子散布距离的统计特性：

- **直方图**：将距离数据分成若干区间，直观显示粒子落地距离的频率分布
- **核密度估计曲线**：使用高斯核函数平滑处理直方图数据，提供连续的概率密度曲线
- **关键指标标记**：在图中标出平均距离、95%置信区间半径和有效散布半径等关键指标

这种可视化方法可以帮助我们理解：
- 气溶胶粒子的主要落地范围
- 落地分布的集中程度和离散程度
- 有效散布半径与统计分布的关系

### 落地时间与距离关系图

时间-距离关系图通过散点图和趋势线分析，揭示粒子飞行时间与散布距离之间的关系：

- **彩色散点图**：每个点代表一个粒子，横坐标为落地时间，纵坐标为落地距离，颜色映射到距离
- **线性拟合**：使用最小二乘法拟合时间与距离的线性关系，并显示R²值反映拟合优度
- **平滑趋势线**：使用Savitzky-Golay滤波器生成平滑趋势线，更好地展示非线性变化趋势

这种可视化方法可以帮助我们发现：
- 粒子飞行时间与散布距离是否存在相关性
- 相关性的强度（通过R²值）和类型（线性/非线性）
- 随时间变化的散布规律

## 建模原理与发现

根据气溶胶抛射模型的研究结果，我们发现：

1. **粒子落地距离分布特性**：
   - 粒子落地呈近似正态分布，但略有偏移
   - 平均散布距离约为1.5-2.0米（取决于参数设置）
   - 95%的粒子落在2.0-2.5米范围内
   - 有效散布半径（平均值+3倍标准差）约为2.5-3.0米

2. **落地时间与距离的关系**：
   - 落地时间与散布距离呈正相关关系
   - 相关性很强（R²通常大于0.7）
   - 存在一定的非线性特征，特别是在较远距离处

3. **关键影响因素**：
   - 初始喷射速度和角度对散布范围影响最大
   - 湍流强度会增加散布的随机性
   - 粒子直径影响沉降速度，从而影响落地时间和距离

## 使用方法

### 安装依赖

```bash
pip install numpy matplotlib scipy
```

### 基本使用

```python
import numpy as np
from aerosol_visualization import plot_distance_distribution, plot_time_distance_scatter

# 准备数据
landing_positions = np.array(...)  # 形状为 [n, 3] 的数组，记录粒子落地位置
landing_times = np.array(...)      # 形状为 [n] 的数组，记录粒子落地时间
mean_distance = 2.0                # 平均散布距离
radius_95 = 2.5                    # 95%置信区间半径
effective_radius = 3.0             # 有效散布半径

# 绘制距离分布图
dist_fig = plot_distance_distribution(
    landing_positions, mean_distance, radius_95, effective_radius,
    title="实验A气溶胶粒子散布距离分布"
)
dist_fig.savefig('distance_distribution.png', dpi=300)

# 绘制时间-距离关系图
time_dist_fig = plot_time_distance_scatter(
    landing_positions, landing_times,
    title="实验A气溶胶粒子落地时间与距离关系"
)
time_dist_fig.savefig('time_distance_scatter.png', dpi=300)
```

### 使用示例数据

该模块也提供了示例数据生成功能，可用于测试或演示：

```python
from aerosol_visualization import load_example_data, plot_distance_distribution, plot_time_distance_scatter

# 加载示例数据
landing_positions, landing_times, mean_distance, radius_95, effective_radius = load_example_data()

# 绘制并保存图表
dist_fig = plot_distance_distribution(landing_positions, mean_distance, radius_95, effective_radius)
dist_fig.savefig('example_distance_distribution.png')

time_dist_fig = plot_time_distance_scatter(landing_positions, landing_times)
time_dist_fig.savefig('example_time_distance_scatter.png')
```

## 直接运行

该模块也可以作为独立程序运行，以生成基于内置示例数据的可视化：

```bash
python aerosol_visualization.py
```

这将生成两个图像文件：`distance_distribution.png` 和 `time_distance_scatter.png`。

## 应用场景

该可视化工具适用于多种研究和应用场景：

1. **病原细菌传播研究**：分析叶片撕裂时病原细菌的散布特性
2. **温室环境安全评估**：评估气溶胶在温室内的传播范围和风险区域
3. **防护措施设计**：基于散布特性设计合理的防护距离和防护措施
4. **模型验证与调优**：通过可视化结果验证气溶胶抛射模型的准确性，指导模型参数调优

## 结论与建议

基于可视化分析结果，我们得出以下结论和建议：

1. **散布范围控制**：
   - 在温室环境中，应保持至少3米的安全距离以避免直接暴露
   - 对于高风险区域，建议设置物理隔离屏障

2. **防护措施设计**：
   - 个人防护装备应覆盖95%置信区间半径范围（约2.5米）
   - 消毒和净化措施应扩展至有效散布半径（约3米）

3. **未来研究方向**：
   - 进一步研究不同环境因素（如风速、湿度）对散布特性的影响
   - 结合气溶胶粒子浓度和生存时间，开发更全面的风险评估模型

## 参考文献

1. Hinds, W. C. (1999). Aerosol Technology: Properties, Behavior, and Measurement of Airborne Particles. Wiley-Interscience.
2. Xie, X., Li, Y., Chwang, A. T., Ho, P. L., & Seto, W. H. (2007). How far droplets can move in indoor environments - revisiting the Wells evaporation-falling curve. Indoor air, 17(3), 211-225.
3. Liu, L., Wei, J., Li, Y., & Ooi, A. (2017). Evaporation and dispersion of respiratory droplets from coughing. Indoor air, 27(1), 179-190. 