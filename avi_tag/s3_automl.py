import os
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor
import parameters  # Import the parameters module with cal_pep function

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
    time_limit: Optional[int] = 60 * 10  # 20 minutes for each task
    presets: str = 'best_quality'


@dataclass
class DataConfig:
    """数据配置类"""
    label: str
    data_path: Path
    train_test_ratio: float = 0.85
    sample_num: int = 150000
    random_state: int = 42


class DataProcessor:
    """数据处理类"""

    def __init__(self, config: DataConfig):
        self.data_path = config.data_path
        self.label = config.label
        self.train_test_ratio = config.train_test_ratio
        self.sample_num = config.sample_num
        self.random_state = config.random_state

    def process(self, label: str) -> Tuple[Union[str, Path], Union[str, Path]]:
        """处理数据并返回训练和测试数据路径"""
        # 检查特征文件是否存在
        features_file = Path(f'./datapack/aa_features_{label}.csv')

        if features_file.exists():
            logger.info(f"Loading existing features from {features_file}")
            features_df = pd.read_csv(features_file)
        else:
            # 读取CSV文件
            total_gen_seq = pd.read_csv(self.data_path)

            # 过滤掉包含'_'的序列
            filtered_sequences = total_gen_seq[~total_gen_seq['AA_sequence'].str.contains('_', na=False)]

            # 提取氨基酸序列
            seqs = filtered_sequences['AA_sequence'].tolist()

            # 使用parameters.cal_pep()提取674个特征
            logger.info("Calculating peptide features...")
            seq_features = [parameters.cal_pep(seq) for seq in seqs]

            # 将特征列表转换为DataFrame
            features_df = pd.DataFrame(seq_features)

            # 添加标签列
            features_df[label] = filtered_sequences[label].values

            # 创建datapack目录（如果不存在）
            Path('./datapack').mkdir(exist_ok=True)

            # 保存特征文件
            features_df.to_csv(features_file, index=False)
            logger.info(f"Saved features to {features_file}")

        # 采样数据
        if self.sample_num > 0 and len(features_df) > self.sample_num:
            features_df = features_df.sample(n=self.sample_num, random_state=self.random_state)
            logger.info(f"Sampled {self.sample_num} instances from the dataset")

        # 数据划分
        train_data, test_data = train_test_split(
            features_df,
            test_size=1 - self.train_test_ratio,
            random_state=self.random_state
        )

        # 保存划分后的数据
        train_path = Path(f'./datapack/train_data_{label}.csv')
        test_path = Path(f'./datapack/test_data_{label}.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Data processed successfully. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        return train_path, test_path


class AutoMLTrainer:
    """AutoML训练类"""

    def __init__(self, config: ModelConfig):
        self.label = config.label
        self.model_path = config.model_path
        self.model_name = config.model_name
        self.time_limit = config.time_limit
        self.presets = config.presets
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

            logger.info(f"Starting model training for {self.label}...")
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Available features: {list(train_data.columns)}")

            predictor = TabularPredictor(
                label=self.label,
                path=str(self.model_path),
                problem_type='regression'
            ).fit(
                train_data,
                presets=self.presets,
                time_limit=self.time_limit
            )

            # 导出模型排行榜和评估指标
            leaderboard = predictor.leaderboard(
                test_data,
                extra_metrics=['rmse', 'mae', 'r2', 'pearsonr', 'spearmanr', 'median_absolute_error'],
                silent=False
            )
            leaderboard.to_csv(self.model_path / (self.label + '_leaderboard.csv'))
            logger.info(f"Leaderboard exported to '{self.model_path / (self.label + '_leaderboard.csv')}'")
            print(leaderboard)

            # 保存模型
            predictor.save(self.model_path / self.model_name)
            logger.info(f"Model saved to {self.model_path / self.model_name}")

        except Exception as e:
            logger.error(f"Error during model training for {self.label}: {str(e)}")
            raise


def main(data_path: str, task: str, model_dir: str):
    """主函数"""
    try:
        # 配置
        data_config = DataConfig(
            data_path=Path(data_path),
            label=task,
        )

        model_config = ModelConfig(
            label=data_config.label,
            model_path=Path(model_dir),
            model_name=f'{task}_model'
        )

        # 数据处理
        processor = DataProcessor(data_config)
        train_path, test_path = processor.process(task)

        # 模型训练和评估
        trainer = AutoMLTrainer(config=model_config)
        trainer.train_and_evaluate(train_path, test_path)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoML task trainer")
    parser.add_argument('--data', type=str, required=True, default='../cleaned_data.csv',
                        help="Path to input CSV data file.")
    parser.add_argument('--task', type=str, default='hTfR1_avi_mean__V8L5_8_mean_log2_enr',
                        required=True, help="The target task for prediction.")
    parser.add_argument('--model_dir', type=str, default='./model', help="Directory to store the trained models.")

    args = parser.parse_args()

    main(args.data, args.task, args.model_dir)

    # python s3_automl.py --data=../cleaned_data.csv --task=hTfR1_avi_mean__V8L5_8_mean_log2_enr