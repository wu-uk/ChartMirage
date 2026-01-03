import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置专业绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os

# Try to find project root by looking for 'outputs' directory
if os.path.exists("outputs"):
    base_dir = "."
elif os.path.exists("../../outputs"):
    base_dir = "../.."
else:
    base_dir = ".."

def plot_metrics():
    try:
        df = pd.read_csv(os.path.join(base_dir, "outputs/results/benchmark_summary.csv"))
    except Exception as e:
        print(f"Error reading summary: {e}")
        return


    metrics = ['False Positive Rate (FPR)', 'False Negative Rate (FNR)']
    methods = [col for col in df.columns if col != 'Metric']
    method_labels = ['ChartMirage (Ours)', 'Baseline A (CLIP)', 'Baseline B (Ragas)']
    
    data = {}
    for i, method in enumerate(methods):
        values = []
        for metric in metrics:
            val_str = df[df['Metric'] == metric][method].values[0]
            values.append(float(val_str.strip('%')))
        data[method_labels[i]] = values

    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2ecc71', '#3498db', '#e74c3c'] # 绿, 蓝, 红

    for i, (label, values) in enumerate(data.items()):
        rects = ax.bar(x + i*width, values, width, label=label, color=colors[i], edgecolor='white', linewidth=1)
        ax.bar_label(rects, padding=5, fmt='%.1f%%', fontweight='bold')

    ax.set_ylabel('Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Performance: ChartMirage vs Baselines', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x + width, ['Lower is Better\n(False Positive Rate)', 'Lower is Better\n(False Negative Rate)'])
    ax.legend(frameon=True, facecolor='white', shadow=True)
    ax.set_ylim(0, 100)
    
    # 添加轻微的阴影效果
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    sns.despine(left=True)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'outputs/plots/defense_comparison_overall_v2.png'), dpi=300)
    print(f"Saved {os.path.join(base_dir, 'outputs/plots/defense_comparison_overall_v2.png')}")

def plot_radar_chart():
    """绘制雷达图，展示多维度对比"""
    try:
        df_summary = pd.read_csv(os.path.join(base_dir, "outputs/results/benchmark_summary.csv"))
        df_raw = pd.read_csv(os.path.join(base_dir, "outputs/results/benchmark_raw_results.csv"))
    except: return

    # 定义维度
    labels = ['FPR (Inv)', 'FNR (Inv)', 'Fake Text Def', 'Fake Image Def']
    num_vars = len(labels)

    # 计算得分 (100 - Rate) 越大越好
    def get_scores(method_idx, pred_col):
        # FPR/FNR scores
        fpr = float(df_summary.iloc[0, method_idx+1].strip('%'))
        fnr = float(df_summary.iloc[1, method_idx+1].strip('%'))
        
        # Scenarios scores
        text_fail = (df_raw[df_raw['Type'] == 'Semantic Attack (Fake Text)'][pred_col] == 'SAFE').mean() * 100
        img_fail = (df_raw[df_raw['Type'] == 'Semantic Attack (Fake Image)'][pred_col] == 'SAFE').mean() * 100
        
        return [100-fpr, 100-fnr, 100-text_fail, 100-img_fail]

    methods_data = {
        'ChartMirage (Ours)': get_scores(0, 'My Pred'),
        'Baseline A (CLIP)': get_scores(1, 'A Pred'),
        'Baseline B (Ragas)': get_scores(2, 'B Pred')
    }

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (label, values) in enumerate(methods_data.items()):
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=3, label=label)
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=10)
    ax.set_title('Multi-dimensional Defense Capability', y=1.1, fontsize=18, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(base_dir, 'outputs/plots/defense_radar_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {os.path.join(base_dir, 'outputs/plots/defense_radar_comparison.png')}")

def plot_ablation():
    """绘制消融实验对比"""
    try:
        df = pd.read_csv(os.path.join(base_dir, "outputs/results/ablation_results.csv"))
    except:
        print("Ablation results not found.")
        return

    # 1. 性能对比 (FPR & FNR)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, df['FPR (%)'], width, label='FPR (Lower is Better)', color='#e74c3c', alpha=0.8)
    rects2 = ax1.bar(x + width/2, df['FNR (%)'], width, label='FNR (Lower is Better)', color='#3498db', alpha=0.8)
    
    ax1.set_ylabel('Rate (%)', fontweight='bold')
    ax1.set_title('Ablation Study: Performance Impact', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Config'], rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 110)

    # 2. 成本对比 (Latency) - 使用次坐标轴
    ax2 = ax1.twinx()
    ax2.plot(x, df['Avg Latency (s)'], color='#2ecc71', marker='o', linewidth=3, markersize=10, label='Avg Latency (Cost)')
    ax2.set_ylabel('Avg Latency (seconds)', color='#2ecc71', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'outputs/plots/ablation_study_analysis.png'), dpi=300)
    print(f"Saved {os.path.join(base_dir, 'outputs/plots/ablation_study_analysis.png')}")

    # 3. 成本对比 (Token Usage)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
    
    # 只对比有 VLM 调用（Tokens > 0）的配置
    df_tokens = df[df['Est. Tokens'] > 0].copy()
    
    bars = ax.bar(df_tokens['Config'], df_tokens['Est. Tokens'], color=colors[:len(df_tokens)], alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Estimated Tokens', fontweight='bold')
    ax.set_title('Ablation Study: Token Consumption (Cost)', fontsize=16, fontweight='bold')
    ax.set_xticklabels(df_tokens['Config'], rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'outputs/plots/ablation_token_cost.png'), dpi=300)
    print(f"Saved {os.path.join(base_dir, 'outputs/plots/ablation_token_cost.png')}")

if __name__ == "__main__":
    plot_metrics()
    plot_radar_chart()
    plot_ablation()
