import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable, Union
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

# 配置日志
logging.basicConfig(
    filename='./logging.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    label: str
    model_path: Path
    model_name: str
    time_limit: Optional[int] = None
    presets: str = 'best_quality'


@dataclass
class DataConfig:
    """数据配置类"""
    label: str
    data_path: Path
    train_test_ratio: float = 0.8
    sample_fraction: float = 1.0
    random_state: int = 42
    positive_ratio: float = 0.5  # 正样本比例


class DataProcessor:
    """数据处理类"""

    def __init__(self, config: DataConfig):
        self.data_path = config.data_path
        self.label = config.label
        self.train_test_ratio = config.train_test_ratio
        self.sample_fraction = config.sample_fraction
        self.random_state = config.random_state
        self.positive_ratio = config.positive_ratio

    def process(self) -> Tuple[Union[str, Path], Union[str, Path]]:
        """处理数据并返回训练和测试数据路径"""
        # 读取CSV文件
        df = pd.read_csv(self.data_path)

        # 创建氨基酸到token的映射
        amino_acid_to_token = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

        # 提取AA序列信息并转换为token表示
        aa_sequences = df['aa_x'].tolist()  # 假设AA序列在'AA'列中
        token_sequences = [[amino_acid_to_token[aa] for aa in seq] for seq in aa_sequences]  # 转换为token表示
        mean_production_values = df['GAS1_virus_S'].tolist()

        # 创建DataFrame用于训练
        train_df = pd.DataFrame({'AA': token_sequences, 'GAS1_virus_S': mean_production_values})

        # 去除mean_Production中的无效值
        train_df = train_df[train_df['GAS1_virus_S'].notnull() & (train_df['GAS1_virus_S'] != float('inf'))]

        # 直接在连续的数据中均匀随机采样
        balanced_train_df = train_df.sample(frac=0.1, random_state=self.random_state)  # 打乱顺序

        # 数据划分
        train_data, test_data = train_test_split(balanced_train_df, test_size=1 - self.train_test_ratio, random_state=self.random_state)

        # 保存划分后的数据
        train_path = Path('../datapack/train_data.csv')
        test_path = Path('../datapack/test_data.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        return train_path, test_path


class AutoMLTrainer:
    """AutoML训练类"""

    def __init__(self, config: ModelConfig):
        self.label = config.label
        self.model_path = config.model_path
        self.model_name = config.model_name
        self.time_limit = config.time_limit
        self.presets = config.presets
        # self.metrics = MetricsFactory.get_metrics()
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置参数"""
        if not self.model_path.exists():
            self.model_path.mkdir(parents=True)

    def train_and_evaluate(self, train_data: Union[str, Path], test_data: Union[str, Path]) -> None:
        """训练模型并评估"""
        try:
            train_data = TabularDataset(str(train_data))
            test_data = TabularDataset(str(test_data))

            logger.info("Starting model training...")

            # 使用rmse作为主要评估指标
            predictor = TabularPredictor(
                label=self.label,
                path=str(self.model_path),
                problem_type='regression'
            ).fit(
                train_data,
                presets=self.presets,
                time_limit=self.time_limit
            )
            print(predictor)

            # 导出模型排行榜和评估指标
            leaderboard = predictor.leaderboard(test_data, extra_metrics=['root_mean_squared_error', 'mean_absolute_error', 'r2', 'pearsonr', 'spearmanr', 'median_absolute_error'], silent=False)
            leaderboard.to_csv(self.model_path / (self.label + 'leaderboard.csv'))
            logger.info("Leaderboard exported to 'leaderboard.csv'.")
            print(leaderboard)

            # 保存模型
            predictor.save(self.model_path / self.model_name)

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise


def main(input_csv: str, output_csv: str):
    """主函数"""
    try:
        # 配置
        data_config = DataConfig(
            data_path=Path(input_csv),
            label='GAS1_virus_S',
            train_test_ratio=0.8,
            sample_fraction=1.0,
            positive_ratio=0.5  # 设置正样本比例
        )

        model_config = ModelConfig(
            label=data_config.label,
            model_path=Path('model_path'),
            model_name='model_name',
            time_limit=60*30
        )

        # 数据处理
        processor = DataProcessor(data_config)
        train_path, test_path = processor.process()

        # 模型训练和评估
        trainer = AutoMLTrainer(config=model_config)
        trainer.train_and_evaluate(train_path, test_path)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    # 调用主函数
    main('../datapack/aav2_output.csv', '../datapack/output_predictions.csv')