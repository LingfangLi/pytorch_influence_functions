from tqdm import tqdm
from constants import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import sentencepiece as spm
import numpy as np

src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")



# given data type: string, test/train
def get_data_loader(file_name):
    # Getting source/target file_name as list
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r') as f:
        src_text_list_original = f.readlines()


    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r') as f:
        trg_text_list_original = f.readlines()

    # EncodeAsId and padding source
    src_text_list = process_src(src_text_list_original)
    input_trg_list, output_trg_list = process_trg(trg_text_list_original)  # (sample_num, L)

    # print(f"The shape of source list: {np.shape(src_text_list)}")
    # print(f"The shape of input trg data: {np.shape(input_trg_list)}")  # (8000, 200)
    # print(f"The shape of output trg data: {np.shape(output_trg_list)}") # (8000, 200)

    # make an instance of CustomDataset, then use make_mask function to create the encoder(B, 1, L) mask and decoder mask(B, L, L)
    custom_dataset = CustomDataset(src_text_list, input_trg_list, output_trg_list)
    e_mask, d_mask = custom_dataset.make_mask()
    # print(f'the e_mask is {e_mask}, and shape is {np.shape(e_mask)}')
    # print(f'the d_mask is {d_mask}, and shape is {np.shape(d_mask)}')


    # convert list into tensor
    src_list = torch.LongTensor(src_text_list)
    input_trg_list = torch.LongTensor(input_trg_list)
    output_trg_list = torch.LongTensor(output_trg_list)

    # model need the first 4 parameters to get prediction, and output_trg_list is label
    dataset = TensorDataset( src_list, input_trg_list, e_mask, d_mask, output_trg_list)
    # print(f"type of them: {type(src_list)}, {type(input_trg_list)}, {type(output_trg_list)}") # <class 'list'>
    # print(f"The shape of {src_list} and {input_trg_list}")
    if file_name == 'TRAIN_NAME':
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    return dataloader,src_text_list_original,trg_text_list_original


# input one sentence
def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text


# input: source sentence list
def process_src(text_list):
    tokenized_list = []
    for text in tqdm(text_list):
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list  # tokenized as ID and plus end of sign id


# target sentence list. make target input and target output. target input: contains the target sequences (e.g.,
# sentences in the target language) that are input to the decoder during training or inference.  provide the decoder
# with the previous tokens of the target sequence up to a certain point to predict the next token. target output:
# ground truth, typically shifted versions of the input_trg_list sequences, start with first actual word.
def process_trg(text_list):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list):
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list


class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(
            output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        # a mask that excludes padding tokens
        e_mask = (self.src_data != pad_id).unsqueeze(1)  # (B, 1, L) 它通过比较源数据src_data与填充标识符pad_id来生成，结果为一个
        # 布尔张量，其中True表示非填充位置，False表示填充位置。unsqueeze(1)操作是为了增加一个维度，使得掩码可以在多头注意力机制中与注意力得分张量进行广播操作。
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        # a mask that both excludes padding tokens and prevents future-looking
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false # 这个掩码有两个目的：一是像e_mask一样排除填充位置的
        # 影响，二是防止解码器在生成当前词时"看到"未来的词。这是通过nopeak_mask实现的，后者是一个下三角形状的布尔张量，允许位置i只关注到i之前
        # 的位置（包括i自身），从而实现自回归特性。d_mask最终通过将填充位置掩码和nopeak_mask进行逻辑与操作合并，既排除了填充位置的影响，也实现了防止提前看到未来信息的目的。

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]
