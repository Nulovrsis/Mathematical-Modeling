import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import matplotlib
import warnings
from matplotlib.font_manager import fontManager, FontProperties

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入配置
from config import DATA_PATHS, VISUALIZATION_PATHS, OUTPUT_PATHS, MODEL_PARAMS

# ==================== 字体配置 ====================

def setup_chinese_fonts():
    """设置中文字体和数学公式渲染"""
    
    # 清除matplotlib的字体缓存并重置
    matplotlib.rcdefaults()
    plt.rcdefaults()
    
    # Windows系统中文字体优先级
    font_candidates = [
        'Microsoft YaHei UI',
        'Microsoft YaHei', 
        'SimHei', 
        'SimSun',
        'FangSong',
        'KaiTi'
    ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in font_candidates:
        # 检查字体是否存在
        font_files = [f for f in fontManager.ttflist if font_name in f.name]
        if font_files:
            available_fonts.append(font_name)
    
    # 设置字体
    if available_fonts:
        chosen_font = available_fonts[0]
        print(f"使用字体: {chosen_font}")
        
        # 设置matplotlib全局字体参数
        plt.rcParams['font.sans-serif'] = [chosen_font] + available_fonts + ['DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 同步设置matplotlib后端参数
        matplotlib.rcParams['font.sans-serif'] = [chosen_font] + available_fonts + ['DejaVu Sans'] 
        matplotlib.rcParams['font.family'] = 'sans-serif'
        
    else:
        print("警告: 未找到中文字体，使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置其他字体参数
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['font.size'] = 12
    
    # 数学公式字体设置
    plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 使用兼容性更好的字体
    plt.rcParams['mathtext.default'] = 'regular'
    
    # 同步到matplotlib全局设置
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    
    # 测试中文显示
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)
        print("中文字体测试通过")
    except Exception as e:
        print(f"中文字体测试失败: {e}")
    
    return plt.rcParams['font.sans-serif'][0]

# 初始化字体设置
print("正在设置字体...")
current_font = setup_chinese_fonts()

# 尝试导入最佳增殖速率模型
try:
    # 使用配置中指定的best_model路径
    sys.path.append(os.path.dirname(DATA_PATHS['best_model']))
    from best_growth_model import calculate_growth_rate
    print("成功导入最佳增殖速率模型")
except ImportError:
    print("警告：无法导入最佳增殖速率模型，将使用默认模型")
    
    # 默认的增殖速率模型（Cardinal温度模型）
    def cardinal_model(T, r_opt, T_min, T_opt, T_max):
        """Cardinal温度模型"""
        if isinstance(T, (list, np.ndarray)):
            result = np.zeros_like(T, dtype=float)
            valid_idx = (T >= T_min) & (T <= T_max)
            if not np.any(valid_idx):
                return result
            
            T_valid = T[valid_idx]
            numerator = (T_valid - T_min) * (T_max - T_valid)
            denominator = (T_opt - T_min) * (T_max - T_opt)
            exponent = (T_max - T_opt) / (T_opt - T_min)
            result[valid_idx] = r_opt * numerator / denominator * ((T_max - T_valid) / (T_max - T_opt)) ** exponent
            return result
        else:
            if T < T_min or T > T_max:
                return 0.0
            numerator = (T - T_min) * (T_max - T)
            denominator = (T_opt - T_min) * (T_max - T_opt)
            exponent = (T_max - T_opt) / (T_opt - T_min)
            return r_opt * numerator / denominator * ((T_max - T) / (T_max - T_opt)) ** exponent
    
    def calculate_growth_rate(T):
        """根据温度T计算细菌增殖速率r"""
        # 使用配置文件中的参数值
        r_opt = MODEL_PARAMS['cardinal']['r_opt'] 
        T_min = MODEL_PARAMS['cardinal']['T_min']
        T_opt = MODEL_PARAMS['cardinal']['T_opt'] 
        T_max = MODEL_PARAMS['cardinal']['T_max']
        return cardinal_model(T, r_opt, T_min, T_opt, T_max)

# --- 改进的模型函数 ---

def humidity_effect(RH):
    """
    相对湿度对细菌生长的影响因子
    
    参数:
    - RH: 相对湿度（0-100%）
    
    返回:
    - 湿度影响因子（0-1之间的值）
    """
    # 假设湿度低于50%时，生长速率受到抑制
    # 湿度在80%-100%时最适宜生长
    if RH < 50:
        return 0.5 * RH / 50.0
    elif RH < 80:
        return 0.5 + 0.5 * (RH - 50) / 30.0
    else:
        return 1.0

def death_rate(T):
    """
    温度相关的死亡率函数
    
    参数:
    - T: 温度（°C）
    
    返回:
    - 死亡率（h^-1）
    """
    # 假设在极端温度下死亡率增加
    # 这里使用简单的分段函数
    if T < 10:
        return 0.05 * (10 - T) / 10.0  # 低温死亡率
    elif T > 35:
        return 0.05 * (T - 35) / 10.0  # 高温死亡率
    else:
        return 0.01  # 正常温度下的基础死亡率

def carrying_capacity(T, RH):
    """
    环境承载量函数，取决于温度和湿度
    
    参数:
    - T: 温度（°C）
    - RH: 相对湿度（0-100%）
    
    返回:
    - 环境承载量（细菌数量）
    """
    # 基础承载量
    base_K = 1e9
    
    # 温度影响因子（在最适温度下最大）
    T_opt = MODEL_PARAMS['cardinal']['T_opt']
    T_factor = 1.0 - 0.01 * min(abs(T - T_opt), 20)
    
    # 湿度影响因子
    H_factor = humidity_effect(RH)
    
    return base_K * T_factor * H_factor

def bacterial_growth_ode(t, y, temp_interp, rh_interp=None, use_logistic=True, 
                        use_death_rate=True, use_humidity=False):
    """
    细菌种群增长的常微分方程
    
    参数:
    - t: 时间点
    - y: 当前状态 [N]，N为细菌数量
    - temp_interp: 温度插值函数
    - rh_interp: 湿度插值函数（可选）
    - use_logistic: 是否使用Logistic模型（考虑环境承载量）
    - use_death_rate: 是否考虑死亡率
    - use_humidity: 是否考虑湿度影响
    
    返回:
    - dN/dt: 细菌数量变化率
    """
    N = y[0]
    
    # 获取当前时间点的温度
    T = temp_interp(t)
    
    # 计算生长率
    r = calculate_growth_rate(T)
    
    # 考虑湿度影响
    if use_humidity and rh_interp is not None:
        RH = rh_interp(t)
        r *= humidity_effect(RH)
        K = carrying_capacity(T, RH)
    else:
        RH = 100  # 默认湿度100%
        K = carrying_capacity(T, RH)
    
    # 考虑死亡率
    d = death_rate(T) if use_death_rate else 0
    
    # 计算dN/dt
    if use_logistic:
        dNdt = r * N * (1 - N / K) - d * N
    else:
        dNdt = r * N - d * N
    
    return [dNdt]

# --- 数据加载与预处理 ---

print("开始数据加载与预处理...")

# 使用配置中的温度数据路径
temperature_data_path = DATA_PATHS['temperature_data']

# 检查文件是否存在
if not os.path.exists(temperature_data_path):
    print(f"错误：未找到文件 {temperature_data_path}。请确保文件路径正确。")
else:
    try:
        # 读取温度实测数据
        # 注意：设置 header=1，将Excel的第二行作为列头 (索引为1)
        df_temp = pd.read_excel(temperature_data_path, header=1)

        # 定义正确的列名
        time_col_name = '时间'
        temp_col_name = '温度(℃)'
        humidity_col_name = '湿度(%)'

        # 验证列名是否存在
        required_cols = [time_col_name, temp_col_name]
        missing_cols = [col for col in required_cols if col not in df_temp.columns]
        
        if missing_cols:
            print(f"错误：读取文件 {temperature_data_path} 后，未找到预期的列: {missing_cols}")
            print("请检查Excel文件的第二行是否包含这些列名，或者是否有多余的空列影响读取。")
            print("\n读取到的列名:", df_temp.columns.tolist())
            print("\n文件头部预览:")
            print(df_temp.head())
        else:
            # 读取时间和温度数据
            time_points_raw = df_temp[time_col_name]
            temperatures = df_temp[temp_col_name]
            
            # 检查是否有湿度数据
            has_humidity_data = humidity_col_name in df_temp.columns
            if has_humidity_data:
                humidity_values = df_temp[humidity_col_name]
                print("检测到湿度数据，将在模型中考虑湿度影响")
            else:
                print("未检测到湿度数据，将使用默认湿度值(100%)")
                humidity_values = pd.Series([100.0] * len(time_points_raw))

            # --- 修改：将 HH:MM:SS 时间字符串转换为小时数 ---
            # 定义一个函数将 HH:MM:SS 字符串转换为总秒数
            def time_string_to_seconds(time_str):
                if pd.isna(time_str): # 处理可能的空值
                    return np.nan
                try:
                    # 尝试解析时间字符串
                    h, m, s = map(int, str(time_str).split(':'))
                    return h * 3600 + m * 60 + s
                except ValueError:
                    return np.nan # 解析失败返回 NaN

            # 将时间列转换为总秒数
            time_seconds = time_points_raw.apply(time_string_to_seconds)

            # 过滤掉时间解析失败的行
            valid_indices = time_seconds.dropna().index
            time_seconds = time_seconds.loc[valid_indices]
            time_points_raw = time_points_raw.loc[valid_indices]
            temperatures = temperatures.loc[valid_indices]
            humidity_values = humidity_values.loc[valid_indices]

            # 将总秒数转换为小时数，并使其从0开始
            if len(time_seconds) > 0:
                start_time_seconds = time_seconds.iloc[0]
                time_points_hours = (time_seconds - start_time_seconds) / 3600.0
            else:
                print("错误：时间列无法解析或为空。无法进行模拟。")
                exit() # 退出程序

            # --- 创建温度和湿度的插值函数 ---
            from scipy.interpolate import interp1d
            
            # 创建温度插值函数（使用三次样条插值，确保平滑）
            temp_interp = interp1d(time_points_hours, temperatures, kind='cubic', 
                                  bounds_error=False, fill_value=(temperatures.iloc[0], temperatures.iloc[-1]))
            
            # 创建湿度插值函数
            if has_humidity_data:
                humidity_interp = interp1d(time_points_hours, humidity_values, kind='cubic',
                                         bounds_error=False, fill_value=(humidity_values.iloc[0], humidity_values.iloc[-1]))
            else:
                # 如果没有湿度数据，创建一个恒定湿度的插值函数
                humidity_interp = lambda t: 100.0
            
            # --- 模型模拟 ---
            
            print("\n开始细菌种群增长模拟...")
            
            # 模型配置参数
            model_config = {
                'use_logistic': True,      # 是否使用Logistic模型（考虑环境承载量）
                'use_death_rate': True,    # 是否考虑死亡率
                'use_humidity': has_humidity_data,  # 是否考虑湿度影响
                'initial_bacteria': 1.0,   # 初始细菌数量
                'simulation_time': 24.0,   # 模拟时间（小时）
                'time_step': 0.1,          # 输出时间步长（小时）
                'solver_method': 'RK45'    # 数值解法（RK45为4阶Runge-Kutta方法）
            }
            
            print(f"模型配置: {model_config}")
            
            # 设定初始细菌数量
            N0 = model_config['initial_bacteria']
            
            # 设定模拟时间范围
            t_span = (0, model_config['simulation_time'])
            
            # 设定输出时间点
            t_eval = np.arange(0, model_config['simulation_time'] + model_config['time_step'], 
                              model_config['time_step'])
            
            # 定义ODE求解器参数
            ode_params = (temp_interp, humidity_interp, 
                         model_config['use_logistic'], 
                         model_config['use_death_rate'],
                         model_config['use_humidity'])
            
            # 使用高精度数值解法求解微分方程
            sol = solve_ivp(
                fun=bacterial_growth_ode,
                t_span=t_span,
                y0=[N0],
                t_eval=t_eval,
                method=model_config['solver_method'],
                args=ode_params,
                rtol=1e-6,  # 相对误差容限
                atol=1e-9   # 绝对误差容限
            )
            
            # 提取结果
            time_points = sol.t
            N_values = sol.y[0]
            lnN_values = np.log(N_values)
            
            # 计算对应时间点的温度和湿度
            temp_values = np.array([temp_interp(t) for t in time_points])
            if has_humidity_data:
                humidity_vals = np.array([humidity_interp(t) for t in time_points])
            else:
                humidity_vals = np.full_like(time_points, 100.0)
            
            # 计算每个时间点的增殖速率
            growth_rates = np.array([calculate_growth_rate(T) for T in temp_values])
            
            # 如果考虑湿度，调整增殖速率
            if model_config['use_humidity']:
                humidity_factors = np.array([humidity_effect(RH) for RH in humidity_vals])
                growth_rates *= humidity_factors
            
            # 如果考虑死亡率，计算净增长率
            if model_config['use_death_rate']:
                death_rates = np.array([death_rate(T) for T in temp_values])
                net_growth_rates = growth_rates - death_rates
            else:
                net_growth_rates = growth_rates
            
            # 如果使用Logistic模型，计算环境承载量
            if model_config['use_logistic']:
                carrying_capacities = np.array([carrying_capacity(T, RH) 
                                             for T, RH in zip(temp_values, humidity_vals)])
                # 计算考虑环境承载量后的实际增长率
                logistic_factors = 1.0 - N_values / carrying_capacities
                effective_growth_rates = net_growth_rates * logistic_factors
            else:
                effective_growth_rates = net_growth_rates
                carrying_capacities = np.full_like(time_points, np.nan)
            
            # 构建结果DataFrame
            results_df = pd.DataFrame({
                '模拟时间 (h)': time_points,
                '温度 (°C)': temp_values,
                '湿度 (%)': humidity_vals,
                '细菌数量 N(t)': N_values,
                'ln(N(t))': lnN_values,
                '基础增殖速率 r(T)': growth_rates,
                '净增长率': net_growth_rates,
                '有效增长率': effective_growth_rates,
                '环境承载量 K': carrying_capacities
            })
            
            print("\n细菌种群增长模拟结果：")
            # 为了避免输出过长，只显示前几行和后几行
            print(results_df.head())
            print("...")
            print(results_df.tail())
            
            # 可以选择将结果保存到新的Excel文件
            try:
                results_df.to_excel(OUTPUT_PATHS['simulation_results'], index=False)
                print(f"\n模拟结果已保存到 {OUTPUT_PATHS['simulation_results']}")
            except Exception as e:
                print(f"\n保存文件失败: {e}")
            
            # --- 比较不同模型的结果 ---
            
            print("\n比较不同模型的结果...")
            
            # 定义不同的模型配置
            model_variants = [
                {'name': '基础指数模型', 'logistic': False, 'death': False, 'humidity': False},
                {'name': '考虑死亡率', 'logistic': False, 'death': True, 'humidity': False},
                {'name': '考虑环境承载量', 'logistic': True, 'death': False, 'humidity': False},
                {'name': '完整模型', 'logistic': True, 'death': True, 'humidity': has_humidity_data}
            ]
            
            # 存储不同模型的结果
            model_results = {}
            
            # 运行不同的模型变体
            for variant in model_variants:
                print(f"模拟 {variant['name']} 模型...")
                
                # 配置ODE参数
                ode_params = (temp_interp, humidity_interp, 
                             variant['logistic'], 
                             variant['death'],
                             variant['humidity'])
                
                # 求解ODE
                sol = solve_ivp(
                    fun=bacterial_growth_ode,
                    t_span=t_span,
                    y0=[N0],
                    t_eval=t_eval,
                    method=model_config['solver_method'],
                    args=ode_params
                )
                
                # 存储结果
                model_results[variant['name']] = sol.y[0]
            
            # --- 可视化部分 ---
            
            print("\n生成可视化图表...")
            
            # 抑制matplotlib字体警告
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            
            # 重新定义中文文件名路径
            CHINESE_VISUALIZATION_PATHS = {
                'bacterial_count': 'outputs\\细菌lnN与环境因素变化图.png',
                'growth_rate_change': 'outputs\\细菌增长率变化图.png'
            }
            
            # 确保字体设置生效
            setup_chinese_fonts()
            
            # 1. 细菌lnN与温度随时间变化图（使用对数坐标）
            fig, ax1 = plt.subplots(figsize=(14, 8))

            # 左侧Y轴：细菌数量（对数坐标）
            ax1.set_xlabel('时间 (h)', fontsize=14)
            ax1.set_ylabel('ln(细菌数量) ln(CFU)', fontsize=14, color='tab:blue')
            ax1.plot(time_points, lnN_values, color='tab:blue', linewidth=2.5, label='ln(细菌数量)')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 右侧Y轴：温度
            ax2 = ax1.twinx()
            ax2.set_ylabel('温度 (°C)', fontsize=14, color='tab:red')
            ax2.plot(time_points, temp_values, color='tab:red', linewidth=2, label='温度')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            # 如果有湿度数据，添加湿度曲线
            if has_humidity_data:
                ax3 = ax1.twinx()
                ax3.spines['right'].set_position(('outward', 60))
                ax3.set_ylabel('湿度 (%)', fontsize=14, color='tab:green')
                ax3.plot(time_points, humidity_vals, color='tab:green', linestyle='--', linewidth=2, label='湿度')
                ax3.tick_params(axis='y', labelcolor='tab:green')
                ax3.set_ylim(0, 100)
                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax3.get_legend_handles_labels()
                ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
            else:
                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.title('病原细菌ln(N)与环境因素随时间变化', fontsize=16, fontweight='bold')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(CHINESE_VISUALIZATION_PATHS['bacterial_count'], dpi=300)
            print(f"  细菌ln(N)与环境因素变化图已保存: {CHINESE_VISUALIZATION_PATHS['bacterial_count']}")

            # 2. 增殖速率与环境因素关系图
            plt.figure(figsize=(14, 8))

            plt.plot(time_points, growth_rates, 'b-', linewidth=2, label='基础增殖速率 r(T)')
            plt.plot(time_points, net_growth_rates, 'g-', linewidth=2, label='净增长率')
            plt.plot(time_points, effective_growth_rates, 'r-', linewidth=2, label='有效增长率')
            
            plt.xlabel('时间 (h)', fontsize=14)
            plt.ylabel('增长率 (h⁻¹)', fontsize=14)
            plt.title('细菌增长率随时间变化', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(CHINESE_VISUALIZATION_PATHS['growth_rate_change'], dpi=300)
            print(f"  细菌增长率变化图已保存: {CHINESE_VISUALIZATION_PATHS['growth_rate_change']}")
            
            plt.show()
            
            print("可视化图表生成完成！")

    except FileNotFoundError:
        print(f"错误：未找到文件 {temperature_data_path}。请确保文件路径正确。")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        import traceback
        traceback.print_exc() # 打印详细错误信息

print("\n细菌种群增长模型模拟完成！")
