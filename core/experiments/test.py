import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'] # 尝试兼容中文显示
plt.rcParams['axes.unicode_minus'] = False

# 创建一个 2x2 的画布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Stitch Fix 数字化战略分析关键图表 (FY2023 - FY2026 Q1)', fontsize=20, y=0.95)

# -----------------------------------------------------------
# 图表 1: 营收增长率反转趋势 (The Financial U-Turn)
# -----------------------------------------------------------
ax1 = axes[0, 0]
periods = ['FY 2023', 'FY 2024', 'FY 2025', 'FY 2026 Q1']
growth_rates = [-21.0, -16.0, -5.3, 7.3]
colors_g = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in growth_rates]

bars = ax1.bar(periods, growth_rates, color=colors_g, alpha=0.8)
ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax1.set_title('图1: 营收同比增长率 (YoY Revenue Growth)', fontsize=14)
ax1.set_ylabel('增长率 (%)')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    offset = -3 if height < 0 else 1
    ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
             f'{height}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=12, fontweight='bold')

# -----------------------------------------------------------
# 图表 2: 用户规模 vs 单客价值 (Clients vs RPAC) - 双轴图
# -----------------------------------------------------------
ax2 = axes[0, 1]
clients = [3.297, 2.633, 2.309, 2.307] # millions
rpac = [497, 525, 549, 559] # dollars

x = np.arange(len(periods))
width = 0.35

ax2_client = ax2.bar(x, clients, width, label='活跃用户数 (百万)', color='#45b7d1', alpha=0.6)
ax2.set_ylabel('活跃用户数 (百万)', color='#45b7d1', fontsize=12)
ax2.set_ylim(0, 4.0)

ax2_rpac = ax2.twinx()
ax2_rpac.plot(x, rpac, color='#ff9f43', marker='o', linewidth=3, label='单客净营收 (RPAC $)')
ax2_rpac.set_ylabel('RPAC ($)', color='#ff9f43', fontsize=12)
ax2_rpac.set_ylim(400, 600)

ax2.set_xticks(x)
ax2.set_xticklabels(periods)
ax2.set_title('图2: 战略背离 - 规模缩减 vs 价值提升', fontsize=14)

# 添加图例
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_rpac.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

# -----------------------------------------------------------
# 图表 3: 竞争格局雷达图 (Competitive Radar)
# -----------------------------------------------------------
ax3 = axes[1, 0] # 这里需要特殊的极坐标处理，我们在下面单独清理并在该子图位置绘制

# 由于 subplot 是矩形的，我们需要在对应位置画极坐标
# 清除原有的 ax3 坐标轴
ax3.remove()
ax3 = fig.add_subplot(2, 2, 3, polar=True)

categories = ['库存效率\n(Inventory Efficiency)', '个性化深度\n(Personalization)', '物流成本控制\n(Low Logistics Cost)', '可扩展性\n(Scalability)', '价格竞争力\n(Price)']
N = len(categories)

# 数据 (1-5分，5分最好)
# Stitch Fix: 库存重(2), 个性化极强(5), 物流成本高(2), 扩展难(3), 价格贵(2)
values_sf = [2, 5, 2, 3, 2]
# Amazon: 库存强(5), 个性化弱(2), 物流无敌(5), 扩展强(5), 价格优(4)
values_amzn = [5, 2, 5, 5, 4]
# Pure AI Apps: 库存无限(5), 个性化强(4), 物流成本0(5), 扩展极强(5), 价格优(3)
values_ai = [5, 4, 5, 5, 3]

# 闭合数据
values_sf += values_sf[:1]
values_amzn += values_amzn[:1]
values_ai += values_ai[:1]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 绘图
ax3.plot(angles, values_sf, linewidth=2, linestyle='solid', label='Stitch Fix', color='#6c5ce7')
ax3.fill(angles, values_sf, '#6c5ce7', alpha=0.1)

ax3.plot(angles, values_amzn, linewidth=2, linestyle='dashed', label='Amazon', color='#b2bec3')

ax3.plot(angles, values_ai, linewidth=2, linestyle='dotted', label='Pure AI Apps', color='#00b894')
ax3.fill(angles, values_ai, '#00b894', alpha=0.1)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, size=9)
ax3.set_title('图3: 竞争优势雷达图', fontsize=14, y=1.1)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# -----------------------------------------------------------
# 图表 4: 人机协同工作流 (Human-in-the-Loop Ratio)
# -----------------------------------------------------------
ax4 = axes[1, 1]
labels = ['AI 算法筛选\n(Algorithm)', '人类造型师决策\n(Stylist)']
sizes = [75, 25] # 数据来源：报告提及 75% 选品由 AI 决定
colors = ['#0984e3', '#fdcb6e']
explode = (0, 0.1) 

ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
ax4.set_title('图4: 选品决策权重 (The "Art & Science" Mix)', fontsize=14)
ax4.axis('equal')

plt.tight_layout()
plt.show()