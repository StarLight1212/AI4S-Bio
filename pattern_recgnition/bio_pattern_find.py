import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from enum import Enum


class SequenceType(Enum):
    DNA = "dna"
    PROTEIN = "protein"


class SequenceProcessor:
    def __init__(self, seq_type=SequenceType.DNA):
        self.seq_type = seq_type
        self.sequence_encoder = LabelEncoder()

        # 定义核苷酸和氨基酸字母表
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        # 根据序列类型选择适当的字母表
        self.alphabet = self.nucleotides if seq_type == SequenceType.DNA else self.amino_acids
        self.alphabet_size = len(self.alphabet)

    def validate_sequence(self, sequence):
        """验证序列是否合法"""
        valid_chars = set(self.alphabet)
        seq_chars = set(sequence.upper())
        invalid_chars = seq_chars - valid_chars
        if invalid_chars:
            raise ValueError(f"发现非法字符: {invalid_chars}")
        return True

    def load_fasta(self, fasta_file):
        """从FASTA文件加载序列"""
        sequences = []
        ids = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            if self.validate_sequence(seq):
                sequences.append(seq)
                ids.append(record.id)
        return sequences, ids

    def encode_sequences(self, sequences):
        """将序列编码为数值形式"""
        # 创建字母表到数字的映射
        encoding_dict = {char: idx for idx, char in enumerate(self.alphabet)}

        # 将序列转换为数值数组
        encoded_seqs = []
        for seq in sequences:
            try:
                encoded_seq = [encoding_dict[nt] for nt in seq.upper()]
                encoded_seqs.append(encoded_seq)
            except KeyError as e:
                raise ValueError(f"序列含有非法字符: {e}")

        return np.array(encoded_seqs, dtype=object)  # 使用 dtype=object

    def one_hot_encode(self, sequences):
        """将序列转换为one-hot编码"""
        # 创建one-hot编码矩阵
        n_sequences = len(sequences)
        seq_length = len(sequences[0])
        one_hot = np.zeros((n_sequences, seq_length, self.alphabet_size))

        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                idx = self.alphabet.index(char.upper())
                one_hot[i, j, idx] = 1

        return one_hot

    def find_motifs(self, sequences, motif_length, min_frequency=0.1):
        """查找序列中的motif模式"""
        from collections import Counter

        all_motifs = []
        for seq in sequences:
            for i in range(len(seq) - motif_length + 1):
                motif = seq[i:i + motif_length]
                all_motifs.append(motif)

        # 计算每个motif的频率
        motif_counts = Counter(all_motifs)
        total_counts = sum(motif_counts.values())

        # 筛选频率超过阈值的motifs
        significant_motifs = [(motif, count / total_counts)
                              for motif, count in motif_counts.items()
                              if count / total_counts >= min_frequency]

        return sorted(significant_motifs, key=lambda x: x[1], reverse=True)

    def calculate_position_weight_matrix(self, sequences, motif_length):
        """计算位置权重矩阵（PWM）"""
        # 初始化计数矩阵
        counts_matrix = np.zeros((motif_length, self.alphabet_size))

        # 统计每个位置的字符频率
        for seq in sequences:
            for i in range(len(seq) - motif_length + 1):
                motif = seq[i:i + motif_length]
                for pos, char in enumerate(motif):
                    idx = self.alphabet.index(char.upper())
                    counts_matrix[pos, idx] += 1

        # 转换为频率
        pwm = counts_matrix / len(sequences)
        return pwm

    def find_conserved_regions(self, sequences, window_size=10, conservation_threshold=0.7):
        """识别保守区域"""
        if not sequences:
            return []

        # 找到最短序列的长度
        min_length = min(len(seq) for seq in sequences)

        # 如果窗口大小大于最短序列长度，调整窗口大小
        if window_size > min_length:
            window_size = min_length
            print(f"警告：窗口大小已调整为最短序列长度 {min_length}")

        conserved_regions = []

        # 只遍历到最短序列长度
        for i in range(min_length - window_size + 1):
            # 对每个序列取相同长度的窗口
            window_sequences = []
            for seq in sequences:
                window = seq[i:i + window_size]
                if len(window) == window_size:  # 确保窗口长度正确
                    window_sequences.append(window)

            if window_sequences:  # 确保有有效的窗口序列
                conservation_score = self._calculate_conservation_score(window_sequences)
                if conservation_score >= conservation_threshold:
                    conserved_regions.append((i, i + window_size, conservation_score))

        return conserved_regions

    def _calculate_conservation_score(self, sequences):
        """计算序列保守性得分"""
        if not sequences:
            return 0.0

        seq_length = len(sequences[0])
        total_score = 0.0

        for pos in range(seq_length):
            # 获取该位置的所有字符
            chars = [seq[pos] for seq in sequences]
            # 计算最常见字符的频率
            most_common = max(set(chars), key=chars.count)
            conservation = chars.count(most_common) / len(chars)
            total_score += conservation

        return total_score / seq_length


# 示例使用
def demo():
    # DNA序列处理示例
    dna_processor = SequenceProcessor(SequenceType.DNA)
    dna_sequences = [
        "ATGCCGTAAT",
        "ATGCCTAATG",
        "ATGCCGTAAT"
    ]

    # 蛋白质序列处理示例
    protein_processor = SequenceProcessor(SequenceType.PROTEIN)
    protein_sequences = [
        "MKWVTFISLLLLFSSAYS",  # 18个氨基酸
        "MKWVTFISLLLLFSSAYS",  # 18个氨基酸
        "MKWVTFISLLLLFSSAYS",  # 18个氨基酸
        "MKWVTFISLLLLFSSAYS"   # 18个氨基酸
    ]

    # DNA序列分析
    print("DNA序列分析:")
    encoded_dna = dna_processor.encode_sequences(dna_sequences)
    dna_motifs = dna_processor.find_motifs(dna_sequences, motif_length=4)
    print(f"DNA Motifs: {dna_motifs[:5]}")

    # 蛋白质序列分析
    print("\n蛋白质序列分析:")
    encoded_protein = protein_processor.encode_sequences(protein_sequences)
    protein_motifs = protein_processor.find_motifs(protein_sequences, motif_length=3)
    print(f"Protein Motifs: {protein_motifs[:5]}")

    # 识别保守区域
    # 使用较小的窗口大小，例如5
    conserved_regions = protein_processor.find_conserved_regions(
        protein_sequences,
        window_size=5,  # 使用较小的窗口大小
        conservation_threshold=0.7
    )
    print(f"\n保守区域: {conserved_regions}")


if __name__ == "__main__":
    demo()
