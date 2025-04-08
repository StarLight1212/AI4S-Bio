import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 读取CSV文件
data = pd.read_csv('../453_avi_tag_NNK_hTfR1.csv')  # 请替换为你的实际文件路径

# 数据清洗步骤：
# 1. 过滤AA_sequence列长度不为8的行
# 2. 移除log2_enrichment中的非有限值(inf/NaN)
clean_data = data[
    (data['AA_sequence'].str.len() == 8) &  # 只保留AA长度为8的序列
    (np.isfinite(data['hTfR1_avi_mean__V8L5_8_mean_log2_enr']))  # 只保留有限值
].copy()

# 提取清洗后的目标列
log2_enr_clean = clean_data['hTfR1_avi_mean__V8L5_8_mean_log2_enr']

# 打印清洗报告
print("=== 数据清洗报告 ===")
print(f"原始数据行数: {len(data)}")
print(f"移除AA长度≠8的行数: {len(data) - len(data[data['AA_sequence'].str.len() == 8])}")
print(f"移除非有限值的行数: {len(data[data['AA_sequence'].str.len() == 8]) - len(clean_data)}")
print(f"最终有效数据行数: {len(clean_data)}")
clean_data.to_csv('../cleaned_data.csv', index=False)

# 绘制histogram
plt.figure(figsize=(12, 7))
n, bins, patches = plt.hist(
    log2_enr_clean,
    bins=60,
    color='#1f77b4',
    edgecolor='white',
    alpha=0.8,
    density=False  # 显示计数而非密度
)

# 添加统计信息
mean_val = np.mean(log2_enr_clean)
median_val = np.median(log2_enr_clean)
plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Mean = {mean_val:.2f}'
)
plt.axvline(
    median_val,
    color='green',
    linestyle=':',
    linewidth=2,
    label=f'Median = {median_val:.2f}'
)

# 美化图形
plt.title(
    'Distribution of Log2 Enrichment (hTfR1_avi/V8L5_8)\n'
    f'[Filtered: AA length=8, N={len(clean_data)} sequences]',
    fontsize=14,
    pad=20
)
plt.xlabel('Log2 Enrichment Ratio', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(fontsize=10)

# 添加网格和调整布局
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# 显示并保存
plt.show()
plt.savefig(
    'filtered_log2_enrichment_distribution.png',
    dpi=300,
    bbox_inches='tight',
    transparent=False
)

# 筛选log2_enr > 9的序列
high_enrich = clean_data[clean_data['hTfR1_avi_mean__V8L5_8_mean_log2_enr'] > 8]

# 统计结果
print("\n=== 高富集序列分析(log2_enr > 9) ===")
print(f"高富集序列数量: {len(high_enrich)}")
print(f"占总有效序列比例: {len(high_enrich)/len(clean_data)*100:.2f}%")
print("\n高富集序列的统计描述:")
print(high_enrich['hTfR1_avi_mean__V8L5_8_mean_log2_enr'].describe())

# 输出前20个高富集序列(按富集度降序)
print("\nTop 20高富集序列:")
top20 = high_enrich.sort_values('hTfR1_avi_mean__V8L5_8_mean_log2_enr', ascending=False).head(20)
print(top20[['AA_sequence', 'hTfR1_avi_mean__V8L5_8_mean_log2_enr']].to_string(index=False))


