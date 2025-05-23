import os
import sys
import traceback

try:
    # 导入配置
    import config
    print("成功导入配置文件")

    # 检查数据路径
    print("\n数据文件路径检查:")
    for key, path in config.DATA_PATHS.items():
        exists = os.path.exists(path)
        print(f"- {key}: {path} (存在: {exists})")

    # 检查输出路径
    print("\n输出目录:")
    print(f"- {config.OUTPUT_DIR} (存在: {os.path.exists(config.OUTPUT_DIR)})")

    # 检查可视化路径
    print("\n可视化文件路径检查:")
    for key, path in config.VISUALIZATION_PATHS.items():
        print(f"- {key}: {path}")

    # 检查模型参数
    print("\n模型参数:")
    for model, params in config.MODEL_PARAMS.items():
        print(f"- {model}:")
        for param_name, param_value in params.items():
            print(f"  - {param_name}: {param_value}")

except Exception as e:
    print(f"错误: {e}")
    traceback.print_exc() 