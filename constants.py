import torch

# Path or parameters for data
DATA_DIR = r'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\data'
# MODEL_DIR = r'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\trained-model'
SP_DIR = f'{DATA_DIR}\sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'src-10000-en.en'
TRG_RAW_DATA_NAME = 'trg-10000-en.es'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'test.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 3208
character_coverage = 1.0
model_type = 'unigram'

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 1e-4
batch_size = 80
seq_len = 200
num_heads = 8
num_layers = 1
d_model = 512
d_ff = 2048  # the dimensionality of the feed-forward network's inner layer within each Transformer block
d_k = d_model // num_heads  # the dimension of the keys (and queries) in the self-attention mechanism
drop_out_rate = 0.1
num_epochs = 100
beam_size = 8
ckpt_dir = 'saved_model'
