import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('../datapack/result.csv')

# 设置模型名称为索引
data.set_index('model', inplace=True)

# 提取需要绘制的指标
metrics_performance = ['r2', 'pearsonr', 'spearmanr']
metrics_error = ['root_mean_squared_error', 'mean_absolute_error']

# 绘制性能指标图
plt.figure(figsize=(12, 6))
for metric in metrics_performance:
    plt.plot(data.index, data[metric], marker='o', label=metric)

plt.title('Model Performance Metrics (R², Pearson, Spearman)')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('performance_metrics.png')  # 保存图像
plt.show()

# 绘制误差指标图
plt.figure(figsize=(12, 6))
for metric in metrics_error:
    plt.plot(data.index, data[metric], marker='o', label=metric)

plt.title('Model Error Metrics (RMSE, MAE)')
plt.xlabel('Model')
plt.ylabel('Error')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('error_metrics.png')  # 保存图像
plt.show()