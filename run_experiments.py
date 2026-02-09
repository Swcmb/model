#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验批量运行脚本
依次运行实验命令并收集错误信息
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_experiment(command, experiment_name):
    """
    运行单个实验并返回结果
    """
    print(f"\n{'='*60}")
    print(f"开始运行实验: {experiment_name}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 执行命令，捕获输出和错误
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=3600  # 设置1小时超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 记录结果
        status = "成功" if result.returncode == 0 else "失败"
        
        print(f"实验 {experiment_name} 运行完成")
        print(f"状态: {status}")
        print(f"耗时: {duration:.2f} 秒")
        
        if result.returncode != 0:
            print(f"错误输出:\n{result.stderr}")
        
        # 显示标准输出
        if result.stdout:
            print(f"标准输出:\n{result.stdout}")
            
        return {
            "name": experiment_name,
            "command": command,
            "status": status,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time
        }
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f"实验 {experiment_name} 超时 (超过1小时)")
        
        return {
            "name": experiment_name,
            "command": command,
            "status": "超时",
            "return_code": -1,
            "stdout": "",
            "stderr": "实验运行超时",
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = str(e)
        print(f"实验 {experiment_name} 运行出错: {error_msg}")
        
        return {
            "name": experiment_name,
            "command": command,
            "status": "异常",
            "return_code": -2,
            "stdout": "",
            "stderr": error_msg,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time
        }

def save_error_report(results, filename):
    """
    保存错误报告到文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"实验运行错误报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总共运行实验数: {len(results)}\n\n")
        
        failed_experiments = [r for r in results if r['status'] != '成功']
        f.write(f"失败实验数: {len(failed_experiments)}\n\n")
        
        if failed_experiments:
            f.write("="*80 + "\n")
            f.write("失败实验详情:\n")
            f.write("="*80 + "\n\n")
            
            for result in failed_experiments:
                f.write(f"实验名称: {result['name']}\n")
                f.write(f"状态: {result['status']}\n")
                f.write(f"返回码: {result['return_code']}\n")
                f.write(f"耗时: {result['duration']:.2f} 秒\n")
                f.write(f"命令: {result['command']}\n")
                f.write(f"错误信息:\n{result['stderr']}\n")
                f.write("-"*80 + "\n\n")
        else:
            f.write("所有实验均成功运行!\n")
            

def save_full_report(results, filename):
    """
    保存完整报告到文件（包括所有输出）
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"实验运行完整报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总共运行实验数: {len(results)}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write("="*80 + "\n")
            f.write(f"实验 {i}: {result['name']}\n")
            f.write("="*80 + "\n")
            f.write(f"状态: {result['status']}\n")
            f.write(f"返回码: {result['return_code']}\n")
            f.write(f"耗时: {result['duration']:.2f} 秒\n")
            f.write(f"命令: {result['command']}\n\n")
            
            f.write(f"标准输出:\n{result['stdout']}\n\n")
            f.write(f"错误输出:\n{result['stderr']}\n\n")

def main():
    # 定义实验命令列表
    experiments = [
        # 1. Baseline实验（对照组）
        {
            "name": "baseline",
            "command": "python main.py --seed=0 --epochs 10 --run_name baseline --fusion_strategy self_attention --moco_type basic --use_co_attention false"
        },
        
        # 2. 协作注意力类型实验
        {
            "name": "co_attention_pairwise",
            "command": "python main.py --seed=0 --epochs 10 --run_name co_attention_pairwise --fusion_strategy co_attention --co_attention_type pairwise --moco_type basic"
        },
        {
            "name": "co_attention_transformer",
            "command": "python main.py --seed=0 --epochs 10 --run_name co_attention_transformer --fusion_strategy co_attention --co_attention_type transformer --moco_type basic"
        },
        {
            "name": "co_attention_multihead",
            "command": "python main.py --seed=0 --epochs 10 --run_name co_attention_multihead --attention_config \"co_attention[type=multihead,num_heads=4]\" --moco_type basic"
        },
        {
            "name": "co_attention_gated_multihead",
            "command": "python main.py --seed=0 --epochs 10 --run_name co_attention_gated_multihead --attention_config \"co_attention[type=gated_multihead,num_heads=4,headwise_gate=true]\" --moco_type basic"
        },
        
        # 3. 融合策略类型实验
        {
            "name": "fusion_self_attention",
            "command": "python main.py --seed=0 --epochs 10 --run_name fusion_self_attention --fusion_strategy self_attention --moco_type basic"
        },
        {
            "name": "fusion_co_attention",
            "command": "python main.py --seed=0 --epochs 10 --run_name fusion_co_attention --fusion_strategy co_attention --co_attention_type transformer --moco_type basic"
        },
        {
            "name": "fusion_hybrid_0.7",
            "command": "python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.7 --fusion_strategy hybrid --fusion_weight 0.7 --co_attention_type transformer --moco_type basic"
        },
        {
            "name": "fusion_hybrid_0.3",
            "command": "python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.3 --fusion_strategy hybrid --fusion_weight 0.3 --co_attention_type transformer --moco_type basic"
        },
        {
            "name": "fusion_gated_transformer",
            "command": "python main.py --seed=0 --epochs 10 --run_name fusion_gated_transformer --attention_config \"gated_transformer[num_heads=8,headwise_gate=true]\" --moco_type basic"
        },
        
        # 4. 门控机制实验
        {
            "name": "gating_none",
            "command": "python main.py --seed=0 --epochs 10 --run_name gating_none --fusion_strategy self_attention --moco_type basic"
        },
        {
            "name": "gating_headwise",
            "command": "python main.py --seed=0 --epochs 10 --run_name gating_headwise --attention_config \"gated_transformer[num_heads=8,headwise_gate=true]\" --moco_type basic"
        },
        {
            "name": "gating_elementwise",
            "command": "python main.py --seed=0 --epochs 10 --run_name gating_elementwise --attention_config \"gated_transformer[num_heads=8,elementwise_gate=true]\" --moco_type basic"
        },
        {
            "name": "gating_combined",
            "command": "python main.py --seed=0 --epochs 10 --run_name gating_combined --attention_config \"gated_transformer[num_heads=8,headwise_gate=true,elementwise_gate=true]\" --moco_type basic"
        },
        
        # 5. MoCo多视图对比学习类型实验
        {
            "name": "moco_basic",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_basic --moco_type basic --moco_K 4096 --fusion_strategy self_attention"
        },
        {
            "name": "moco_enhanced_default",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_enhanced_default --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-3 --fusion_strategy self_attention"
        },
        {
            "name": "moco_enhanced_large_hidden",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_enhanced_large_hidden --moco_type enhanced --moco_gate_hidden 128 --moco_prune_threshold 1e-3 --fusion_strategy self_attention"
        },
        {
            "name": "moco_enhanced_low_prune",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_enhanced_low_prune --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-4 --fusion_strategy self_attention"
        },
        {
            "name": "moco_enhanced_custom",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_enhanced_custom --moco_config \"enhanced[K=8192,gate_hidden=128,prune_threshold=5e-4]\" --fusion_strategy self_attention"
        },
        
        # 6. MoCo队列大小实验
        {
            "name": "moco_queue_small",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_queue_small --fusion_strategy self_attention --moco_type basic --moco_K 1024"
        },
        {
            "name": "moco_queue_large",
            "command": "python main.py --seed=0 --epochs 10 --run_name moco_queue_large --fusion_strategy self_attention --moco_type basic --moco_K 8192"
        }
    ]
    
    print(f"开始批量运行实验，总计 {len(experiments)} 个实验")
    print("注意：已移除所有涉及 tri_entity_mode 和 tri_fusion_strategy 参数的实验，因为这些参数在当前代码中未实现")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 存储所有实验结果
    results = []
    
    # 依次运行实验
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 准备运行实验...")
        result = run_experiment(exp["command"], exp["name"])
        results.append(result)
        
        # 即使实验失败也继续运行下一个实验
        if result['status'] != '成功':
            print(f"警告: 实验 {exp['name']} 运行失败，将继续运行下一个实验...")
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_report_filename = f"experiment_errors_{timestamp}.txt"
    full_report_filename = f"experiment_full_report_{timestamp}.txt"
    save_error_report(results, error_report_filename)
    save_full_report(results, full_report_filename)
    
    # 输出总结
    total_experiments = len(results)
    successful_experiments = len([r for r in results if r['status'] == '成功'])
    failed_experiments = total_experiments - successful_experiments
    
    print(f"\n{'='*60}")
    print("实验运行完成总结")
    print(f"{'='*60}")
    print(f"总实验数: {total_experiments}")
    print(f"成功实验数: {successful_experiments}")
    print(f"失败实验数: {failed_experiments}")
    print(f"成功率: {successful_experiments/total_experiments*100:.1f}%")
    print(f"错误报告已保存至: {error_report_filename}")
    print(f"完整报告已保存至: {full_report_filename}")
    
    if failed_experiments > 0:
        print("\n失败的实验:")
        for result in results:
            if result['status'] != '成功':
                print(f"  - {result['name']}: {result['status']}")

if __name__ == "__main__":
    main()