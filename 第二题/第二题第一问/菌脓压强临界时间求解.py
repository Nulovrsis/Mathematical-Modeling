import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import sys
import traceback

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelParameters:
    def __init__(self):
        # 基础参数
        self.apoplast_volume = 0.0125  # mL
        self.bacteria_volume = 5e-13  # cm³
        self.bacteria_mass = 6e-13    # g
        self.bacteria_density_in_pus = 1e8  # 每毫升菌脓中的细菌数量
        
        # EPS相关参数
        self.eps_production_rate = 2.5e-14  # g/(细菌·小时)
        self.eps_water_absorption = 10.0    # 每克EPS可以吸收的水量(g)
        
        # 水势相关参数
        self.bacteria_potential_factor = 0.00005  # 菌脓水势因子
        self.leaf_potential_base = -0.15   # 基础叶片水势（MPa）
        self.humidity_effect = 0.01        # 湿度对叶片水势的影响因子
        
        # 生长影响参数
        self.water_effect_threshold = 0.25   # 水势影响的阈值（MPa）
        self.water_effect_sensitivity = 4.0  # 敏感度
        
        # 临界压强（Pa）
        self.P_crit = 2e5  # 0.2 MPa

def cardinal_temperature_model(T):
    """Cardinal温度模型"""
    r_opt = 0.3  # 最优生长率
    T_min = 10.0  # 最低生长温度
    T_opt = 25.0  # 最适生长温度
    T_max = 40.0  # 最高生长温度
    
    if T < T_min or T > T_max:
        return 0.0
    
    numerator = (T - T_min) * (T_max - T)
    denominator = (T_opt - T_min) * (T_max - T_opt)
    exponent = (T_max - T_opt) / (T_opt - T_min)
    
    return r_opt * numerator / denominator * ((T_max - T) / (T_max - T_opt)) ** exponent

class SystemState:
    def __init__(self, params):
        self.N = 1e3  # 初始细菌数量
        self.EPS = 1e-12  # 初始EPS量
        self.pressure = 0  # 初始压强
        self.params = params
        
    def calculate_bacteria_potential(self):
        """计算菌脓水势"""
        # 计算EPS吸收的水量
        water_absorbed = self.EPS * self.params.eps_water_absorption
        
        # 计算相对体积膨胀率
        relative_volume_expansion = 2.0 * water_absorbed / self.params.apoplast_volume
        
        # 计算细菌密度相对值
        relative_bacteria_density = 1.5 * self.N / (self.params.bacteria_density_in_pus * self.params.apoplast_volume)
        
        # 计算菌脓水势
        bacteria_potential = 0.18 * (1 - np.exp(-3.0 * relative_volume_expansion)) + \
                           0.12 * (1 - np.exp(-3.0 * relative_bacteria_density))
        
        return bacteria_potential
    
    def calculate_leaf_potential(self, RH):
        """计算叶片水势"""
        return self.params.leaf_potential_base * (1 - 0.5 * RH/100)
    
    def update(self, dN, dEPS, dt, RH):
        """更新系统状态"""
        self.N += dN * dt
        self.EPS += dEPS * dt
        
        # 更新压强
        bacteria_potential = self.calculate_bacteria_potential()
        leaf_potential = self.calculate_leaf_potential(RH)
        self.pressure = bacteria_potential - leaf_potential

def load_environmental_data():
    """加载环境数据"""
    try:
        # 尝试直接使用绝对路径
        data_file = 'D:/A题/A题/附件1.xlsx'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"未能找到文件: {data_file}")
            
        print(f"成功找到数据文件: {os.path.abspath(data_file)}")
        
        # 读取Excel文件
        df = pd.read_excel(data_file)
        
        # 删除前两行（标题行）
        df = df.iloc[2:].copy()  # 使用copy()避免视图警告
        df.columns = ['序号', '时间', '温度', '相对湿度']
        
        print("原始数据形状:", df.shape)
        print("数据预览:")
        print(df.head())
        
        # 转换时间为小时
        def convert_time_to_hours(time_str):
            try:
                if pd.isna(time_str):
                    return np.nan
                parts = str(time_str).strip().split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h + m/60 + s/3600
                return np.nan
            except:
                return np.nan
        
        # 应用时间转换
        df['时间'] = df['时间'].apply(convert_time_to_hours)
        
        # 转换温度和湿度为数值
        df['温度'] = pd.to_numeric(df['温度'], errors='coerce')
        df['相对湿度'] = pd.to_numeric(df['相对湿度'], errors='coerce')
        
        # 删除任何包含NaN的行
        df = df.dropna(subset=['时间', '温度', '相对湿度'])  # 只检查这些列
        
        print("\n处理后数据形状:", df.shape)
        print("处理后数据范围:")
        print(f"时间范围: {df['时间'].min():.2f} - {df['时间'].max():.2f} 小时")
        print(f"温度范围: {df['温度'].min():.1f} - {df['温度'].max():.1f} °C")
        print(f"相对湿度范围: {df['相对湿度'].min():.1f} - {df['相对湿度'].max():.1f} %")
        
        if df.empty:
            raise ValueError("处理后的数据为空")
        
        return df
        
    except Exception as e:
        print(f"加载环境数据失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def create_interpolation_functions(env_data):
    """创建温度和湿度的插值函数"""
    temp_interp = interp1d(env_data['时间'], env_data['温度'], 
                          kind='cubic', fill_value='extrapolate')
    rh_interp = interp1d(env_data['时间'], env_data['相对湿度'], 
                         kind='cubic', fill_value='extrapolate')
    return temp_interp, rh_interp

def get_environmental_conditions(t, temp_interp, rh_interp):
    """获取任意时刻的环境条件"""
    t_in_day = t % 24
    return temp_interp(t_in_day), rh_interp(t_in_day)

def calculate_growth_rate(state, temp):
    """计算细菌生长率"""
    # 基础生长率（温度依赖）
    base_rate = cardinal_temperature_model(temp)
    
    # 压强影响
    relative_pressure = state.pressure / state.params.water_effect_threshold
    pressure_effect = 0.2 + 0.8 / (1 + np.exp(state.params.water_effect_sensitivity * (relative_pressure - 1)))
    
    # 密度依赖
    K = 1e7  # 环境承载量
    density_effect = max(0.1, (1 - state.N/K))
    
    return base_rate * pressure_effect * density_effect

def find_burst_time(params, max_time=200, dt=0.01):
    """寻找破裂时间"""
    # 加载环境数据
    env_data = load_environmental_data()
    temp_interp, rh_interp = create_interpolation_functions(env_data)
    
    # 初始化系统状态
    state = SystemState(params)
    
    # 存储结果
    times = []
    bacteria_counts = []
    eps_amounts = []
    pressures = []
    temperatures = []
    humidities = []
    
    t = 0
    while t < max_time and state.pressure < params.P_crit/1e6:  # 转换为MPa
        # 获取环境条件
        temp, rh = get_environmental_conditions(t, temp_interp, rh_interp)
        
        # 计算变化率
        growth_rate = calculate_growth_rate(state, temp)
        dN = state.N * growth_rate
        dEPS = state.N * params.eps_production_rate
        
        # 更新状态
        state.update(dN, dEPS, dt, rh)
        
        # 记录数据
        times.append(t)
        bacteria_counts.append(state.N)
        eps_amounts.append(state.EPS)
        pressures.append(state.pressure)
        temperatures.append(temp)
        humidities.append(rh)
        
        t += dt
    
    return {
        'burst_time': t,
        'times': np.array(times),
        'bacteria_counts': np.array(bacteria_counts),
        'eps_amounts': np.array(eps_amounts),
        'pressures': np.array(pressures),
        'temperatures': np.array(temperatures),
        'humidities': np.array(humidities)
    }

def plot_results(results):
    """可视化结果"""
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. 细菌数量变化
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['times'], results['bacteria_counts'], 'b-', label='细菌数量')
    ax1.set_yscale('log')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('细菌数量 (个/mL)')
    ax1.set_title('细菌生长曲线')
    ax1.grid(True)
    ax1.legend()
    
    # 2. EPS累积
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['times'], results['eps_amounts']*1e12, 'g-', label='EPS总量')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('EPS总量 (pg)')
    ax2.set_title('EPS累积曲线')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 压强变化
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['times'], results['pressures'], 'r-', label='压强')
    ax3.axhline(y=0.2, color='k', linestyle='--', label='临界压强')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('压强 (MPa)')
    ax3.set_title('压强变化曲线')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 环境条件
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    l1 = ax4.plot(results['times'], results['temperatures'], 'r-', label='温度')
    l2 = ax4_twin.plot(results['times'], results['humidities'], 'b-', label='相对湿度')
    
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('温度 (°C)', color='r')
    ax4_twin.set_ylabel('相对湿度 (%)', color='b')
    ax4.set_title('环境条件变化')
    
    # 合并图例
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels)
    
    # 5. 生长率分析
    ax5 = fig.add_subplot(gs[2, :])
    growth_rates = np.diff(np.log(results['bacteria_counts'])) / np.diff(results['times'])
    ax5.plot(results['times'][1:], growth_rates, 'k-', label='瞬时生长率')
    ax5.set_xlabel('时间 (小时)')
    ax5.set_ylabel('生长率 (1/小时)')
    ax5.set_title('细菌生长率变化')
    ax5.grid(True)
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig('临界时间分析结果.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 初始化参数
    params = ModelParameters()
    
    # 计算破裂时间
    print("开始计算破裂时间...")
    results = find_burst_time(params)
    
    # 输出结果
    print(f"\n计算完成！")
    print(f"破裂时间（tburst）: {results['burst_time']:.2f} 小时")
    print(f"最终细菌数量: {results['bacteria_counts'][-1]:.2e} 个/mL")
    print(f"最终压强: {results['pressures'][-1]:.3f} MPa")
    
    # 可视化结果
    print("\n生成结果图像...")
    plot_results(results)
    print("结果图像已保存为'临界时间分析结果.png'")
    
    # 保存详细结果
    print("\n保存详细数据...")
    results_df = pd.DataFrame({
        '时间(小时)': results['times'],
        '细菌数量(个/mL)': results['bacteria_counts'],
        'EPS总量(g)': results['eps_amounts'],
        '压强(MPa)': results['pressures'],
        '温度(°C)': results['temperatures'],
        '相对湿度(%)': results['humidities']
    })
    results_df.to_excel('临界时间分析结果.xlsx', index=False)
    print("详细数据已保存为'临界时间分析结果.xlsx'")

if __name__ == "__main__":
    main() 