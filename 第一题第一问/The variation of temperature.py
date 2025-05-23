import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import signal, stats
import sys
import os
import matplotlib
import warnings
from matplotlib.font_manager import fontManager, FontProperties

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入配置
from config import DATA_PATHS, VISUALIZATION_PATHS, OUTPUT_PATHS

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
current_font = setup_chinese_fonts()

# ==================== 数据处理函数 ====================

def read_temperature_data(file_path):
    """读取温度数据文件"""
    encodings = ['utf-8', 'gbk', 'utf-8-sig', 'latin1']
    
    try:
        df = pd.read_excel(file_path, header=1)
        print(f"成功读取Excel文件: {file_path}")
        return df
    except Exception as e:
        print(f"Excel读取失败: {e}")
        print("尝试以CSV格式读取...")
        
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"成功使用编码 {enc} 读取CSV文件")
                return df
            except Exception as e:
                print(f"编码 {enc} 读取失败: {e}")
        
        raise ValueError(f"无法读取文件 {file_path}，请检查文件格式")

def time_to_minutes(tstr):
    """将时间字符串转换为分钟数"""
    if pd.isna(tstr) or not isinstance(tstr, str):
        return np.nan
    try:
        t = datetime.strptime(tstr.strip(), '%H:%M:%S')
        return t.hour * 60 + t.minute + t.second / 60
    except ValueError:
        print(f"时间格式错误: '{tstr}'")
        return np.nan

def detect_outliers(data, method='zscore', threshold=3):
    """检测异常值"""
    if method == 'zscore':
        z_scores = stats.zscore(data)
        return np.abs(z_scores) > threshold
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

def smooth_temperature_data(temp_data, method='savgol', window=11):
    """平滑温度数据"""
    if method == 'savgol':
        return signal.savgol_filter(temp_data, window_length=window, polyorder=3)
    elif method == 'moving_average':
        return temp_data.rolling(window=window, center=True).mean().fillna(temp_data)
    else:
        raise ValueError("method must be 'savgol' or 'moving_average'")

# ==================== 温度模型定义 ====================

def temp_model(x, A, phi, mu, A2, phi2, k):
    """
    复合正弦温度模型
    参数: x: 时间（小时）
    """
    main_cycle = A * np.sin(2 * np.pi * x / 24 + phi)
    secondary_cycle = A2 * np.sin(2 * np.pi * x / 12 + phi2)
    linear_trend = k * x
    return main_cycle + secondary_cycle + linear_trend + mu

def simple_temp_model(x, A, phi, mu):
    """简单正弦温度模型"""
    return A * np.sin(2 * np.pi * x / 24 + phi) + mu

def fit_temperature_model(time_hours, temperature, model_type='complex'):
    """拟合温度模型"""
    try:
        if model_type == 'complex':
            T_mean = temperature.mean()
            T_range = temperature.max() - temperature.min()
            bounds = ([0, -np.pi, T_mean-5, 0, -np.pi, -0.5],
                     [T_range, np.pi, T_mean+5, T_range/2, np.pi, 0.5])
            initial_guess = [T_range/2, 0, T_mean, T_range/4, 0, 0]
            popt, pcov = curve_fit(temp_model, time_hours, temperature,
                                  p0=initial_guess, bounds=bounds, maxfev=5000)
            return popt, pcov, temp_model
        else:
            T_mean = temperature.mean()
            T_range = temperature.max() - temperature.min()
            initial_guess = [T_range/2, 0, T_mean]
            popt, pcov = curve_fit(simple_temp_model, time_hours, temperature,
                                  p0=initial_guess, maxfev=5000)
            return popt, pcov, simple_temp_model
    except Exception as e:
        raise RuntimeError(f"模型拟合失败: {e}")

def calculate_model_metrics(observed, predicted):
    """计算模型评估指标"""
    residuals = observed - predicted
    ss_tot = np.sum((observed - observed.mean())**2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    return {'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'residuals': residuals}

# ==================== 数据分析函数 ====================

def analyze_day_night_temperature(df, day_start=6, day_end=18):
    """分析昼夜温度差异"""
    df['时段'] = df['分钟'].apply(lambda x: '白天' if day_start*60 <= x < day_end*60 else '夜间')
    day_temp = df[df['时段'] == '白天']['温度_平滑']
    night_temp = df[df['时段'] == '夜间']['温度_平滑']
    return {
        '白天均值': day_temp.mean(),
        '夜间均值': night_temp.mean(),
        '昼夜温差': day_temp.mean() - night_temp.mean(),
        '白天标准差': day_temp.std(),
        '夜间标准差': night_temp.std(),
        '白天最高温': day_temp.max(),
        '夜间最低温': night_temp.min()
    }

def calculate_temperature_rates(df):
    """计算温度变化率"""
    try:
        df['温度变化率'] = df['温度_平滑'].diff() / df['分钟'].diff() * 60
        rising_rates = df[df['温度变化率'] > 0]['温度变化率']
        falling_rates = df[df['温度变化率'] < 0]['温度变化率']
        return {
            '平均升温率': rising_rates.mean() if len(rising_rates) > 0 else 0,
            '平均降温率': falling_rates.mean() if len(falling_rates) > 0 else 0,
            '最大升温率': rising_rates.max() if len(rising_rates) > 0 else 0,
            '最大降温率': falling_rates.min() if len(falling_rates) > 0 else 0
        }
    except Exception as e:
        print(f"计算温度变化率时出错: {e}")
        return {}

def analyze_temperature_periodicity(temperature_data, sampling_interval=5):
    """分析温度周期性"""
    if len(temperature_data) < 50:
        return None
    try:
        detrended = signal.detrend(temperature_data)
        fft_data = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=sampling_interval/60)
        pos_mask = freqs > 0
        peak_idx = np.argmax(np.abs(fft_data[pos_mask]))
        peak_freq = freqs[pos_mask][peak_idx]
        period = 1/peak_freq if peak_freq != 0 else 0
        return {'主要周期': period, '主要频率': peak_freq, 'FFT振幅': np.abs(fft_data[pos_mask][peak_idx])}
    except Exception as e:
        print(f"周期性分析出错: {e}")
        return None

# ==================== 可视化函数 ====================

def create_temperature_time_plot(df, model_params=None, save_path=None):
    """创建温度时序图"""
    # 抑制字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 确保字体设置生效
    setup_chinese_fonts()
    
    plt.figure(figsize=(14, 8))
    plt.plot(df['分钟']/60, df['温度'], 'o', markersize=3, alpha=0.3, label='原始温度数据', color='lightblue')
    plt.plot(df['分钟']/60, df['温度_平滑'], '-', linewidth=2, label='平滑温度数据', color='blue')
    
    if '异常温度' in df.columns and any(df['异常温度']):
        outliers = df[df['异常温度']]
        plt.scatter(outliers['分钟']/60, outliers['温度'], color='red', marker='x', s=80, label='异常值', zorder=5)
    
    if model_params is not None:
        x_fit = np.linspace(0, 24, 1000)
        if len(model_params) == 6:
            A, phi, mu, A2, phi2, k = model_params
            y_fit = temp_model(x_fit, A, phi, mu, A2, phi2, k)
            model_name = '复合正弦拟合模型'
        else:
            A, phi, mu = model_params
            y_fit = simple_temp_model(x_fit, A, phi, mu)
            model_name = '简单正弦拟合模型'
        plt.plot(x_fit, y_fit, 'r-', linewidth=3, label=model_name)
    
    plt.axvspan(0, 6, color='navy', alpha=0.1, label='夜间')
    plt.axvspan(6, 18, color='gold', alpha=0.1, label='白天')
    plt.axvspan(18, 24, color='navy', alpha=0.1)
    
    T_max_idx = df['温度_平滑'].idxmax()
    T_min_idx = df['温度_平滑'].idxmin()
    T_max = df.loc[T_max_idx, '温度_平滑']
    T_min = df.loc[T_min_idx, '温度_平滑']
    plt.scatter([df.loc[T_max_idx, '分钟']/60], [T_max], color='red', s=100, marker='*', label=f'最高温度 ({T_max:.1f}℃)', zorder=5)
    plt.scatter([df.loc[T_min_idx, '分钟']/60], [T_min], color='blue', s=100, marker='*', label=f'最低温度 ({T_min:.1f}℃)', zorder=5)
    
    plt.xlabel('时间（小时）')
    plt.ylabel('温度（℃）')
    plt.title('温室24小时温度变化详细分析')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10)
    plt.xticks(np.arange(0, 25, 3))
    plt.xlim(0, 24)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def create_temperature_distribution_plot(temperature_data, save_path=None):
    """创建温度分布图"""
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    setup_chinese_fonts()
    
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(temperature_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    mu, std = stats.norm.fit(temperature_data)
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'正态分布拟合 (μ={mu:.2f}, σ={std:.2f})')
    
    stat, p_norm = stats.shapiro(temperature_data)
    norm_test_result = "符合" if p_norm > 0.05 else "不符合"
    plt.text(0.05, 0.9, f'Shapiro-Wilk检验: {norm_test_result}正态分布 (p={p_norm:.4f})',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('温度（℃）')
    plt.ylabel('频率密度')
    plt.title('温度分布直方图与正态分布拟合')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def create_model_components_plot(df, model_params, model_metrics, save_path=None):
    """创建模型组件分解图"""
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    setup_chinese_fonts()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(df['分钟']/60, df['温度_平滑'], s=30, alpha=0.6, label='平滑温度数据', color='lightblue')
    
    x_fit = np.linspace(0, 24, 1000)
    if len(model_params) == 6:
        A, phi, mu, A2, phi2, k = model_params
        y_complete = temp_model(x_fit, A, phi, mu, A2, phi2, k)
        plt.plot(x_fit, y_complete, 'r-', linewidth=3, label='复合正弦拟合模型')
        main_component = A * np.sin(2 * np.pi * x_fit / 24 + phi) + mu
        secondary_component = A2 * np.sin(2 * np.pi * x_fit / 12 + phi2)
        trend_component = k * x_fit
        plt.plot(x_fit, main_component, '--', color='green', alpha=0.7, linewidth=2, label='主周期成分（24小时）')
        if A2 > 0.2:
            plt.plot(x_fit, secondary_component + mu, '--', color='purple', alpha=0.7, linewidth=2, label='次周期成分（12小时）')
        if abs(k) > 0.01:
            plt.plot(x_fit, trend_component + mu, '--', color='blue', alpha=0.7, linewidth=2, label='线性趋势')
        equation_text = (f'T(t) = {A:.2f}×sin(2πt/24 + {phi:.2f}) + '
                        f'{A2:.2f}×sin(2πt/12 + {phi2:.2f}) + '
                        f'{k:.4f}×t + {mu:.2f}')
    else:
        A, phi, mu = model_params
        y_complete = simple_temp_model(x_fit, A, phi, mu)
        plt.plot(x_fit, y_complete, 'r-', linewidth=3, label='简单正弦拟合模型')
        equation_text = f'T(t) = {A:.2f}×sin(2πt/24 + {phi:.2f}) + {mu:.2f}'
    
    plt.text(0.5, 0.03, equation_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
             ha='center', fontsize=10, fontfamily='monospace')
    metrics_text = (f'拟合优度 R² = {model_metrics["r_squared"]:.4f}, '
                   f'RMSE = {model_metrics["rmse"]:.3f}℃')
    plt.text(0.5, 0.1, metrics_text, transform=plt.gca().transAxes, 
             ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('时间（小时）')
    plt.ylabel('温度（℃）')
    plt.title('温度变化拟合模型及其组成成分')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.xlim(0, 24)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

# ==================== 主程序 ====================

def main():
    """主程序"""
    print("=" * 50)
    print("温室温度变化分析程序")
    print("=" * 50)
    
    # 抑制matplotlib字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 1. 数据读取与预处理
    print("\n1. 数据读取与预处理...")
    
    try:
        temperature_file = DATA_PATHS['temperature_data']
        df = read_temperature_data(temperature_file)
        print(f"读取到的列名: {df.columns.tolist()}")
        
        temp_col = next((col for col in df.columns if '温度' in str(col)), None)
        if temp_col is None:
            raise ValueError("未找到温度列，请检查数据文件")
        
        df = df.rename(columns={temp_col: '温度'})
        df['分钟'] = df['时间'].apply(time_to_minutes)
        df = df.dropna(subset=['分钟', '温度']).reset_index(drop=True)
        print(f"成功读取 {len(df)} 条温度记录")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
    
    # 2. 异常值检测与数据平滑
    print("\n2. 异常值检测与数据平滑...")
    df['异常温度'] = detect_outliers(df['温度'], method='zscore', threshold=3)
    outlier_count = df['异常温度'].sum()
    print(f"检测到 {outlier_count} 个异常值")
    df['温度_平滑'] = smooth_temperature_data(df['温度'], method='savgol')
    df['温度_移动平均'] = smooth_temperature_data(df['温度'], method='moving_average')
    
    # 3. 基础统计分析
    print("\n3. 基础统计分析...")
    T = df['温度_平滑']
    stats_results = {
        '均值': T.mean(),
        '中位数': T.median(),
        '最大值': T.max(),
        '最小值': T.min(),
        '极差': T.max() - T.min(),
        '标准差': T.std(),
        '偏度': stats.skew(T),
        '峰度': stats.kurtosis(T)
    }
    T_max_time = df.loc[T.idxmax(), '时间']
    T_min_time = df.loc[T.idxmin(), '时间']
    print("温度统计结果:")
    for key, value in stats_results.items():
        if key in ['均值', '中位数', '最大值', '最小值', '极差', '标准差']:
            print(f"  {key}: {value:.2f}℃")
        else:
            print(f"  {key}: {value:.3f}")
    print(f"  最高温时间: {T_max_time}")
    print(f"  最低温时间: {T_min_time}")
    
    # 4. 昼夜分析
    print("\n4. 昼夜温度分析...")
    day_night_analysis = analyze_day_night_temperature(df)
    for key, value in day_night_analysis.items():
        print(f"  {key}: {value:.2f}℃")
    
    # 5. 温度变化率分析
    print("\n5. 温度变化率分析...")
    rate_analysis = calculate_temperature_rates(df)
    for key, value in rate_analysis.items():
        print(f"  {key}: {value:.3f}℃/小时")
    
    # 6. 周期性分析
    print("\n6. 温度周期性分析...")
    periodicity = analyze_temperature_periodicity(T.values)
    if periodicity:
        print(f"  主要周期: {periodicity['主要周期']:.2f} 小时")
        print(f"  主要频率: {periodicity['主要频率']:.4f} Hz")
    
    # 7. 温度模型拟合
    print("\n7. 温度模型拟合...")
    time_hours = df['分钟'] / 60
    try:
        popt, pcov, model_func = fit_temperature_model(time_hours, T, model_type='complex')
        if len(popt) == 6:
            A, phi, mu, A2, phi2, k = popt
            print(f"复合模型拟合参数:")
            print(f"  主周期振幅 A = {A:.2f}℃")
            print(f"  主周期相位 φ = {phi:.2f} rad")
            print(f"  平均温度 μ = {mu:.2f}℃")
            print(f"  次周期振幅 A₂ = {A2:.2f}℃")
            print(f"  次周期相位 φ₂ = {phi2:.2f} rad")
            print(f"  线性趋势 k = {k:.4f}℃/h")
        model_params = popt
    except Exception as e:
        print(f"复合模型拟合失败: {e}")
        print("尝试简单正弦模型...")
        try:
            popt, pcov, model_func = fit_temperature_model(time_hours, T, model_type='simple')
            A, phi, mu = popt
            print(f"简单模型拟合参数:")
            print(f"  振幅 A = {A:.2f}℃")
            print(f"  相位 φ = {phi:.2f} rad")
            print(f"  平均温度 μ = {mu:.2f}℃")
            model_params = popt
        except Exception as e:
            print(f"模型拟合完全失败: {e}")
            model_params = None
    
    # 8. 模型评估
    if model_params is not None:
        print("\n8. 模型评估...")
        if len(model_params) == 6:
            y_pred = temp_model(time_hours, *model_params)
        else:
            y_pred = simple_temp_model(time_hours, *model_params)
        model_metrics = calculate_model_metrics(T, y_pred)
        print(f"  拟合优度 R²: {model_metrics['r_squared']:.4f}")
        print(f"  均方根误差 RMSE: {model_metrics['rmse']:.3f}℃")
        print(f"  平均绝对误差 MAE: {model_metrics['mae']:.3f}℃")
        if len(model_params) == 6:
            np.savez(OUTPUT_PATHS['temperature_model_params'], A=model_params[0], phi=model_params[1], mu=model_params[2],
                    A2=model_params[3], phi2=model_params[4], k=model_params[5])
        else:
            np.savez(OUTPUT_PATHS['temperature_model_params'], A=model_params[0], phi=model_params[1], mu=model_params[2],
                    A2=0, phi2=0, k=0)
        print(f"模型参数已保存到: {OUTPUT_PATHS['temperature_model_params']}")
    
    # 9. 生成可视化图表
    print("\n9. 生成可视化图表...")
    try:
        fig1 = create_temperature_time_plot(df, model_params, VISUALIZATION_PATHS['temperature_time_plot'])
        print(f"  温度时序图已保存: {VISUALIZATION_PATHS['temperature_time_plot']}")
        fig2 = create_temperature_distribution_plot(T, VISUALIZATION_PATHS['temperature_distribution'])
        print(f"  温度分布图已保存: {VISUALIZATION_PATHS['temperature_distribution']}")
        if model_params is not None:
            fig3 = create_model_components_plot(df, model_params, model_metrics, VISUALIZATION_PATHS['temperature_sine_fit'])
            print(f"  模型拟合图已保存: {VISUALIZATION_PATHS['temperature_sine_fit']}")
        plt.show()
    except Exception as e:
        print(f"图表生成出错: {e}")
    
    print("\n=" * 50)
    print("温度变化分析完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()