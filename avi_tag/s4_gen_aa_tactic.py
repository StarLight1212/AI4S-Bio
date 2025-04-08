import random
import argparse
from pathlib import Path
from typing import List
import pandas as pd

# 标准氨基酸字母表（20种标准氨基酸）
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def generate_unique_sequences(count: int, length: int = 8) -> List[str]:
    """
    生成无重复的随机氨基酸序列
    :param count: 需要生成的序列数量
    :param length: 每个序列的长度
    :return: 无重复的氨基酸序列列表
    """
    sequences = set()
    max_possible = len(AMINO_ACIDS) ** length
    if count > max_possible:
        raise ValueError(f"Cannot generate {count} unique sequences. "
                         f"Maximum possible for length {length} is {max_possible}")

    print(f"Generating {count} unique {length}-mer sequences...")
    while len(sequences) < count:
        seq = ''.join(random.choices(AMINO_ACIDS, k=length))
        sequences.add(seq)

        # 进度显示
        if len(sequences) % 100000 == 0:
            print(f"Generated {len(sequences)} sequences...")

    return list(sequences)


def save_sequences(sequences: List[str], output_file: str) -> None:
    """保存序列到CSV文件"""
    df = pd.DataFrame(sequences, columns=['AA_sequence'])
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} sequences to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate unique random amino acid sequences')
    parser.add_argument('--output', type=str, default='random_sequences.csv',
                        help='Output CSV file path')
    parser.add_argument('--count', type=int, default=1_000_000,
                        help='Number of sequences to generate')
    parser.add_argument('--length', type=int, default=8,
                        help='Length of each sequence')

    args = parser.parse_args()

    try:
        # 生成序列
        sequences = generate_unique_sequences(args.count, args.length)

        # 保存到文件
        save_sequences(sequences, args.output)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()