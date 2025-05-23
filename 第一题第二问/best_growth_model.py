import numpy as np

# 二次多项式模型
def quadratic_model(T, a, b, c):
    return a * T**2 + b * T + c

# Cardinal温度模型
def cardinal_model(T, r_opt, T_min, T_opt, T_max):
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

# Ratkowsky模型
def ratkowsky_model(T, b, T_min, T_max):
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

# 模型参数值
quadratic_params = {}
quadratic_params['a'] = 0.0005740290648579283
quadratic_params['b'] = -0.025524382788736588
quadratic_params['c'] = 0.783364101693672

cardinal_params = {}
cardinal_params['r_opt'] = 0.4682379593240666
cardinal_params['T_min'] = 13.451697563194791
cardinal_params['T_opt'] = 29.999999999998895
cardinal_params['T_max'] = 44.99999999999999

ratkowsky_params = {}
ratkowsky_params['b'] = 0.0524798739559453
ratkowsky_params['T_min'] = 5.000000000000001
ratkowsky_params['T_max'] = 37.60775599040646

# 最佳模型信息
best_model_name = '二次多项式模型'

# 通用生长速率计算函数
def calculate_growth_rate(T):
    """根据温度T计算细菌增殖速率r"""
    return quadratic_model(T, quadratic_params['a'], quadratic_params['b'], quadratic_params['c'])
