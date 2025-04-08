import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import matplotlib
matplotlib.use('TkAgg')

# 读取并清洗数据
data = pd.read_csv('../453_avi_tag_NNK_hTfR1.csv')  # 替换为你的文件路径
clean_data = data[
    (data['AA_sequence'].str.len() == 8) &
    (np.isfinite(data['hTfR1_avi_mean__V8L5_8_mean_log2_enr']))
    ].copy()

# 筛选高富集序列 (log2_enr > 8)
high_enrich = clean_data[clean_data['hTfR1_avi_mean__V8L5_8_mean_log2_enr'] > 8]
print(f"Found {len(high_enrich)} sequences with log2 enrichment > 8")

if len(high_enrich) == 0:
    print("No sequences meet the criteria, exiting...")
    exit()

# 准备WebLogo数据（修正后的版本）
try:
    counts_matrix = logomaker.alignment_to_matrix(
        sequences=high_enrich['AA_sequence'].tolist(),
        to_type='counts'
    )

    # 确保矩阵包含所有标准氨基酸（处理NNK文库可能出现的缺失氨基酸）
    standard_aas = list('ACDEFGHIKLMNPQRSTVWY')
    for aa in standard_aas:
        if aa not in counts_matrix.columns:
            counts_matrix[aa] = 0
    counts_matrix = counts_matrix[standard_aas]  # 统一列顺序

    # 创建WebLogo图
    plt.figure(figsize=(12, 4))
    logo = logomaker.Logo(
        counts_matrix,
        font_name='Arial',
        color_scheme='chemistry',
        vpad=0.2,
        width=0.8
    )

    # 美化图形
    plt.title(
        f'Sequence Logo of High-Enrichment Peptides (log2_enr > 8)\n'
        f'N = {len(high_enrich)} sequences | Max enrichment = {high_enrich["hTfR1_avi_mean__V8L5_8_mean_log2_enr"].max():.2f}',
        fontsize=14,
        pad=10
    )
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Information (bits)', fontsize=12)
    plt.xticks(range(8), labels=range(1, 9))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 保存和显示
    plt.tight_layout()
    plt.savefig('high_enrichment_weblogo.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print(f"Error generating logo: {str(e)}")
    print("Sample sequences for debugging:")
    print(high_enrich['AA_sequence'].head())