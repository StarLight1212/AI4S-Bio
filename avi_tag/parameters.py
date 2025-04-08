import os

from featured_data_generated import (
    BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
)


# Helper class for dictionary-like object creation
class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


# Calculate peptide features
def cal_pep(seq):
    seq_desc = {}
    seq = str(seq)
    # Calculate various sequence descriptors
    seq_desc.update(AAComposition.CalculateAAComposition(seq))
    seq_desc.update(AAComposition.CalculateDipeptideComposition(seq))
    seq_desc.update(Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(seq, lamba=5))
    seq_desc.update(CTD.CalculateCTD(seq))
    seq_desc.update(QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(seq, maxlag=5))
    seq_desc.update(PseudoAAC._GetPseudoAAC(seq, lamda=4))
    seq_desc.update(PseudoAAC.GetAPseudoAAC(seq, lamda=4))
    seq_desc.update(BasicDes.cal_discriptors(seq))

    return seq_desc


# General configuration
config = {
    'config_json': '',  # Path to load config.json
    'ignore_gpu': False,
    'seed': 1238,
    'tiny': False,
    'SEED': 88,
    'tb_toplevel': 'tb',
    'savepath_toplevel': 'output',
    'runname': 'default',
    'datapath': 'data',
    'loadpath': 'auto',
    'vocab_path': 'auto',
    'phase': -1,
    'part': 0,
    'partN': 1,
    'resume_result_json': True
}

# Model configurations
bert = Bunch(
    date='20230418',
    hidden_size=128,
    vocab=22,
    dropout=0.05,
    attn_heads=16,
    n_layers=1,
    seq_len=7,
    res_layers=5,
    hidden=2048,
    MAX_LEN=50,
    bz=128,
    epoch=15
)

roberta = Bunch(
    date='20230418',
    hidden_size=128,
    vocab=22,
    dropout=0.1,
    attn_heads=32,
    n_layers=1,
    seq_len=7,
    res_layers=5,
    hidden=4096,
    MAX_LEN=50,
    bz=128,
    epoch=15
)


GAN = Bunch(
    date='20230506',
    model_name='SeqGAN',
    pretrain_model='roberta',
    lr=2e-6,
    gen_num=1000,
    train_len=7,
    gen_len=10,
    vocab=22,
    hidden_size=128,
    dropout=0.05,
    attn_heads=16,
    n_layers=1,
    n_epochs=9,
    res_layers=5,
    Bert_hidden=2048,
    MAX_LEN=50,
    batch_size=256,
    latent_dim=32,
)

GAN_Infer = Bunch(
    train_date='20230506',
    date='20230617',
    model_name='SeqGAN',
    pretrain_model='roberta',
    n_epochs=1,
)

GAN_Infer_noPre = Bunch(
    train_date='20230506',
    date='20230617',
    model_name='SeqGAN',
    n_epochs=9,
)

# Ranking parameters
Ranking_model_name = 'SeqGAN'
Ranking_pretrain_model = 'roberta'



# Ranking configuration
RA_infer_date = '20230515'
date = '20230515'


# RA configuration
RA = Bunch(
    date='20230625',
    RA_model=['SA', 'GA', 'EDA', 'EDG', 'NONE'],
)

# Random configuration
Random = Bunch(
    date='20230515',
    model_name='Random',
)
