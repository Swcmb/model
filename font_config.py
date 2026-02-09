# -*- coding: utf-8 -*-
"""
字体配置模块
自动生成的matplotlib中文字体配置
"""

import matplotlib.pyplot as plt
import warnings

def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 高质量绘图设置
        plt.rcParams.update({
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "lines.antialiased": True,
            "patch.antialiased": True,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
        
        print("✓ 字体配置已加载: Microsoft YaHei")
        
    except Exception as e:
        warnings.warn(f"字体设置失败: {e}", UserWarning)
        # 使用默认设置
        plt.rcParams['axes.unicode_minus'] = False

# 自动执行字体设置
setup_chinese_font()
