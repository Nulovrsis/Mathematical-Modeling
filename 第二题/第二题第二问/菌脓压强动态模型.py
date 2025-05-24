import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys
import traceback

# 在文件开头添加Cardinal模型
def cardinal_temperature_model(T):
    """Cardinal温度模型 - 来自第一题"""
    # 模型参数（来自第一题的优化结果）
    r_opt = 0.3  # 最优生长率
    T_min = 10.0  # 最低生长温度
    T_opt = 25.0 # 最适生长温度
    T_max = 40.0 # 最高生长温度
    
    # 计算生长率
    if T < T_min or T > T_max:
        return 0.0
    
    numerator = (T - T_min) * (T_max - T)
    denominator = (T_opt - T_min) * (T_max - T_opt)
    exponent = (T_max - T_opt) / (T_opt - T_min)
    
    return r_opt * numerator / denominator * ((T_max - T) / (T_max - T_opt)) ** exponent

# 使用Cardinal模型作为生长率计算函数
calculate_growth_rate = cardinal_temperature_model

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
first_q_third_dir = os.path.join(parent_dir, '第一题第三问')
sys.path.append(first_q_third_dir)

try:
    # 直接使用绝对路径导入
    sys.path.append('D:/A题/A题/第一题第三问')
    from 种群增长模型 import calculate_growth_rate
    print("成功导入第一题的生长率计算函数")
except ImportError as e:
    print(f"警告：无法导入最佳增殖速率模型，将使用默认模型: {e}")
    print(f"当前Python路径: {sys.path}")
    def calculate_growth_rate(T):
        """备用的生长率计算函数"""
        T_min, T_max, T_opt, r_opt = 4.0, 45.0, 28.0, 0.3
        if T <= T_min or T >= T_max:
            return 0.0
        numerator = (T - T_max) * (T - T_min)**2
        denominator = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - 
                                        (T_opt - T_max)*(T_opt + T_min - 2*T))
        return r_opt * (numerator / denominator)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
os.makedirs('outputs', exist_ok=True)

# 模型参数
class ModelParameters:
    def __init__(self):
        # 基础参数（题目给出）
        self.apoplast_volume = 0.0125  # mL (取0.01-0.015的中间值)
        self.bacteria_volume = 5e-13  # cm³
        self.bacteria_mass = 6e-13    # g
        self.bacteria_density_in_pus = 1e8  # 每毫升菌脓中的细菌数量
        
        # EPS相关参数
        self.eps_production_rate = 2.5e-14  # g/(细菌·小时)，提高EPS产生率
        self.eps_water_absorption = 10.0    # 每克EPS可以吸收的水量(g)
        
        # 水势相关参数
        self.bacteria_potential_factor = 0.00005  # 菌脓水势因子
        self.leaf_potential_base = -0.15   # 基础叶片水势（MPa）
        self.humidity_effect = 0.01        # 湿度对叶片水势的影响因子
        
        # 生长影响参数
        self.water_effect_threshold = 0.25   # 水势影响的阈值（MPa），提高阈值
        self.water_effect_sensitivity = 4.0  # 降低敏感度，使影响更平缓

# 加载环境数据
def load_environmental_data():
    """加载环境数据"""
    try:
        # 尝试不同的可能路径
        possible_paths = [
            '../附件1.xlsx',
            'D:/A题/A题/附件1.xlsx',
            'D:/A题/附件1.xlsx',
            './附件1.xlsx'
        ]
        
        data_file = None
        for path in possible_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if data_file is None:
            raise FileNotFoundError("未能在任何可能的路径中找到附件1.xlsx")
            
        print(f"成功找到数据文件: {os.path.abspath(data_file)}")
        
        # 读取Excel文件
        df = pd.read_excel(data_file)
        print(f"原始数据形状: {df.shape}")
        
        # 删除前两行（标题行）
        df = df.iloc[2:].reset_index(drop=True)
        
        # 重命名列
        df.columns = ['序号', '时间', '温度', '相对湿度']
        print(f"列名: {df.columns.tolist()}")
        print("数据预览（处理前）:")
        print(df.head())
        
        # 提取并转换时间列
        time_str = df['时间'].astype(str).str.strip()
        time_minutes = []
        
        for t in time_str:
            try:
                if ':' in t:
                    parts = t.split(':')
                    if len(parts) == 3:
                        h, m, s = map(int, parts)
                        minutes = h * 60 + m + s/60.0
                    else:
                        minutes = np.nan
                else:
                    minutes = np.nan
                time_minutes.append(minutes)
            except:
                print(f"无法解析时间: {t}")
                time_minutes.append(np.nan)
        
        time_minutes = np.array(time_minutes)
        
        # 提取温度和湿度数据
        temp_data = pd.to_numeric(df['温度'], errors='coerce')
        rh_data = pd.to_numeric(df['相对湿度'], errors='coerce')
        
        # 创建新的DataFrame
        new_df = pd.DataFrame({
            '时间': time_minutes / 60.0,  # 转换为小时
            '温度': temp_data,
            '相对湿度': rh_data
        })
        
        # 删除任何包含NaN的行
        new_df = new_df.dropna()
        
        print("\n处理后数据预览:")
        print(new_df.head())
        print(f"处理后数据形状: {new_df.shape}")
        
        # 验证数据范围
        print("\n数据范围检查:")
        print(f"时间范围: {new_df['时间'].min():.2f} - {new_df['时间'].max():.2f} 小时")
        print(f"温度范围: {new_df['温度'].min():.1f} - {new_df['温度'].max():.1f} °C")
        print(f"相对湿度范围: {new_df['相对湿度'].min():.1f} - {new_df['相对湿度'].max():.1f} %")
        
        if new_df.empty:
            raise ValueError("处理后的数据为空")
            
        return new_df
        
    except Exception as e:
        print(f"加载环境数据失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)  # 如果无法加载环境数据，直接退出程序

def calculate_bacteria_potential(bacteria_count, eps_amount, params):
    """计算菌脓水势（与EPS含量正相关）"""
    # 计算EPS吸收的水量
    water_absorbed = eps_amount * params.eps_water_absorption  # g水
    
    # 计算体积膨胀效应（假设水的密度为1 g/mL）
    volume_expansion = water_absorbed / 1.0  # mL
    
    # 计算相对体积膨胀率（确保持续增长）
    relative_volume_expansion = 2.0 * volume_expansion / params.apoplast_volume
    
    # 计算细菌密度相对值（确保持续增长）
    relative_bacteria_density = 1.5 * bacteria_count / (params.bacteria_density_in_pus * params.apoplast_volume)
    
    # 计算累积效应的水势（使用更平滑的非线性函数）
    bacteria_potential = 0.18 * (
        1 - np.exp(-3.0 * relative_volume_expansion)  # EPS吸水效应
    ) + 0.12 * (
        1 - np.exp(-3.0 * relative_bacteria_density)  # 细菌密度效应
    )
    
    return bacteria_potential

def calculate_leaf_potential(RH, params):
    """计算叶片水势（受环境湿度影响）"""
    # 使用相对湿度计算叶片水势
    # 湿度越高，水势越接近0；湿度越低，水势越小（负值越大）
    leaf_potential = params.leaf_potential_base * (1 - 0.5*RH/100)  # 降低湿度的影响
    return leaf_potential  # 返回负值

def calculate_growth_effects(bacteria_count, bacteria_potential, leaf_potential, temp, params):
    """计算各种因素对生长的影响"""
    
    # 1. 温度影响（使用第一题的模型）
    base_rate = calculate_growth_rate(temp)
    
    # 2. 水势影响（考虑水势差）
    pressure = bacteria_potential - leaf_potential  # 水势差
    # 当压强接近临界值时，生长受到抑制，但不会完全停止
    relative_pressure = pressure / params.water_effect_threshold
    water_effect = 0.2 + 0.8 / (1 + np.exp(params.water_effect_sensitivity * (relative_pressure - 1)))  # 确保最小值为0.2
    
    # 3. 密度依赖性（考虑最大承载量，但保持生长）
    K = 1e7  # 环境承载量
    density_effect = max(0.1, (1 - bacteria_count/K))  # 确保最小值为0.1
    
    return base_rate, water_effect, density_effect

# 主要模拟函数
def simulate_pressure_dynamics(params, time_span, time_points):
    """模拟压强动态变化"""
    # 加载环境数据
    env_data = load_environmental_data()
    
    # 处理环境数据
    times = np.linspace(0, time_span, time_points)
    
    # 由于环境数据是24小时循环的，我们需要重复数据来覆盖整个模拟时间
    # 首先创建完整的24小时数据
    full_day_times = env_data['时间'].values
    full_day_temps = env_data['温度'].values
    full_day_rh = env_data['相对湿度'].values
    
    # 对每个模拟时间点，计算对应的24小时周期内的时间
    cyclic_times = times % 24
    
    # 使用插值获取温度和湿度值
    temp_data = np.interp(cyclic_times, full_day_times, full_day_temps, period=24)
    rh_data = np.interp(cyclic_times, full_day_times, full_day_rh, period=24)
    
    print("\n环境数据处理:")
    print(f"模拟时间范围: 0 - {time_span} 小时")
    print(f"温度范围: {temp_data.min():.1f} - {temp_data.max():.1f} °C")
    print(f"相对湿度范围: {rh_data.min():.1f} - {rh_data.max():.1f} %")
    
    # 初始条件
    N0 = 1e3  # 增加初始细菌数量
    eps0 = 1e-12  # 设置少量初始EPS
    
    # 存储结果
    results = {
        'time': times,
        'bacteria_count': np.zeros(time_points),
        'eps_amount': np.zeros(time_points),
        'bacteria_potential': np.zeros(time_points),
        'leaf_potential': np.zeros(time_points),
        'pressure': np.zeros(time_points),
        'temperature': temp_data,
        'humidity': rh_data,
        'growth_rate': np.zeros(time_points),
        'base_rate': np.zeros(time_points),
        'water_effect': np.zeros(time_points),
        'density_effect': np.zeros(time_points)
    }
    
    # 初始值
    results['bacteria_count'][0] = N0
    results['eps_amount'][0] = eps0
    
    # 模拟时间演化
    for i in range(1, time_points):
        dt = times[i] - times[i-1]
        
        # 当前环境条件
        current_temp = temp_data[i]
        current_RH = rh_data[i]
        
        # 计算菌脓水势和叶片水势
        bacteria_potential = calculate_bacteria_potential(
            results['bacteria_count'][i-1],
            results['eps_amount'][i-1],
            params
        )
        leaf_potential = calculate_leaf_potential(current_RH, params)
        
        # 计算压强 = 菌脓水势 - 叶片水势
        pressure = bacteria_potential - leaf_potential
        
        # 存储水势和压强结果
        results['bacteria_potential'][i-1] = bacteria_potential
        results['leaf_potential'][i-1] = leaf_potential
        results['pressure'][i-1] = pressure
        
        # 计算各个影响因素
        base_rate, water_effect, density_effect = calculate_growth_effects(
            results['bacteria_count'][i-1],
            bacteria_potential,
            leaf_potential,
            current_temp,
            params
        )
        
        # 存储影响因素
        results['base_rate'][i-1] = base_rate
        results['water_effect'][i-1] = water_effect
        results['density_effect'][i-1] = density_effect
        
        # 计算总生长率
        growth_rate = base_rate * water_effect * density_effect
        results['growth_rate'][i-1] = growth_rate
        
        # 更新细菌数量
        dN = growth_rate * results['bacteria_count'][i-1] * dt
        results['bacteria_count'][i] = results['bacteria_count'][i-1] + dN
        
        # 更新EPS总量
        dEPS = results['bacteria_count'][i] * params.eps_production_rate * dt
        results['eps_amount'][i] = results['eps_amount'][i-1] + dEPS
    
    # 计算最后一个时间点的值
    results['bacteria_potential'][-1] = bacteria_potential
    results['leaf_potential'][-1] = leaf_potential
    results['pressure'][-1] = pressure
    base_rate, water_effect, density_effect = calculate_growth_effects(
        results['bacteria_count'][-1],
        bacteria_potential,
        leaf_potential,
        temp_data[-1],
        params
    )
    results['base_rate'][-1] = base_rate
    results['water_effect'][-1] = water_effect
    results['density_effect'][-1] = density_effect
    results['growth_rate'][-1] = base_rate * water_effect * density_effect
    
    return results

# 可视化结果
def plot_results(results):
    """可视化结果"""
    # 设置全局样式
    plt.style.use('seaborn-v0_8-darkgrid')  # 使用seaborn的深色网格样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#F0F0F0'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 1.5
    
    # 创建图形和子图
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])  # 生长率影响因素
    
    # 定义颜色方案
    colors = {
        'bacteria': '#2E86C1',    # 深蓝色
        'eps': '#28B463',         # 深绿色
        'temp': '#E74C3C',        # 红色
        'humidity': '#3498DB',    # 浅蓝色
        'pressure': '#8E44AD',    # 紫色
        'water': '#2C3E50',       # 深灰色
        'marker': '#E74C3C',      # 标记线颜色
        'base_rate': '#E67E22',   # 橙色
        'water_effect': '#27AE60', # 绿色
        'density_effect': '#8E44AD' # 紫色
    }
    
    # 1. 细菌数量变化
    ax1.plot(results['time'], results['bacteria_count'], 
             color=colors['bacteria'], 
             label='细菌数量',
             alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('细菌数量 (个/mL)')
    ax1.set_title('细菌生长曲线')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax1.axvline(x=36, color=colors['marker'], linestyle='--', 
                label='36小时标记', alpha=0.5)
    
    # 2. EPS累积
    ax2.plot(results['time'], results['eps_amount']*1e12, 
             color=colors['eps'], 
             label='EPS总量',
             alpha=0.8)
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('EPS总量 (pg)')
    ax2.set_title('EPS累积曲线')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax2.axvline(x=36, color=colors['marker'], linestyle='--', alpha=0.5)
    
    # 3. 环境条件
    temp_line = ax3.plot(results['time'], results['temperature'], 
                        color=colors['temp'], 
                        label='温度',
                        alpha=0.8)[0]
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('温度 (°C)', color=colors['temp'])
    ax3.tick_params(axis='y', labelcolor=colors['temp'])
    ax3.set_title('环境条件变化')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    ax3_twin = ax3.twinx()
    humidity_line = ax3_twin.plot(results['time'], results['humidity'], 
                                 color=colors['humidity'], 
                                 label='相对湿度',
                                 alpha=0.8)[0]
    ax3_twin.set_ylabel('相对湿度 (%)', color=colors['humidity'])
    ax3_twin.tick_params(axis='y', labelcolor=colors['humidity'])
    
    # 合并两个图例
    lines = [temp_line, humidity_line]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='best', framealpha=0.9, fancybox=True, shadow=True)
    
    # 4. 压强和水势关系
    ax4.plot(results['time'], results['pressure'], 
             color=colors['pressure'], 
             label='压强 (菌脓水势-叶片水势)',
             alpha=0.8)
    ax4.plot(results['time'], results['bacteria_potential'], 
             color='#2ecc71', 
             label='菌脓水势',
             linestyle='--',
             alpha=0.8)
    ax4.plot(results['time'], results['leaf_potential'], 
             color='#e74c3c', 
             label='叶片水势',
             linestyle=':',
             alpha=0.8)
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('水势/压强 (MPa)')
    ax4.set_title('压强与水势关系\n(压强 = 菌脓水势 - 叶片水势)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax4.axvline(x=36, color=colors['marker'], linestyle='--', alpha=0.5)
    
    # 5. 生长率影响因素
    ax5.plot(results['time'], results['base_rate'], 
             color=colors['base_rate'], 
             label='温度影响',
             alpha=0.8)
    ax5.plot(results['time'], results['water_effect'], 
             color=colors['water_effect'], 
             label='水势影响（压强相关）',
             alpha=0.8)
    ax5.plot(results['time'], results['density_effect'], 
             color=colors['density_effect'], 
             label='密度依赖性',
             alpha=0.8)
    ax5.plot(results['time'], results['growth_rate'], 
             color='black', 
             label='总生长率',
             alpha=0.8)
    ax5.set_xlabel('时间 (小时)')
    ax5.set_ylabel('影响因子值')
    ax5.set_title('细菌生长率的影响因素\n(总生长率 = 温度影响 × 水势影响 × 密度依赖性)')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax5.axvline(x=36, color=colors['marker'], linestyle='--', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('outputs/菌脓压强动态模型结果_改进版.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

def main():
    # 初始化参数
    params = ModelParameters()
    
    # 运行模拟（增加数据点数量）
    results = simulate_pressure_dynamics(params, time_span=48, time_points=5000)
    
    # 可视化结果
    plot_results(results)
    
    # 输出36小时时的临界压强
    critical_time_index = np.abs(results['time'] - 36).argmin()
    critical_pressure = results['pressure'][critical_time_index]
    print(f"\n36小时时的临界压强（Pcrit）: {critical_pressure:.3f} MPa")
    
    # 保存结果到文本文件
    with open('outputs/临界压强结果_改进版.txt', 'w', encoding='utf-8') as f:
        f.write(f"临界压强（Pcrit）计算结果：\n")
        f.write(f"时间点：36小时\n")
        f.write(f"临界压强：{critical_pressure:.3f} MPa\n")
        f.write(f"\n模型参数：\n")
        f.write(f"质外体体积：{params.apoplast_volume} mL\n")
        f.write(f"单个细菌体积：{params.bacteria_volume} cm³\n")
        f.write(f"单个细菌质量：{params.bacteria_mass} g\n")
        f.write(f"菌脓中细菌密度：{params.bacteria_density_in_pus} 个/mL\n")
        f.write(f"EPS产生速率：{params.eps_production_rate} g/(细菌·小时)\n")
        f.write(f"EPS吸水率：{params.eps_water_absorption} g水/g EPS\n")

if __name__ == "__main__":
    main() 