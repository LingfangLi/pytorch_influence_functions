from layers import *


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # embedding and positional encoder in encoder and decoder. Then run transformer from encoder to decoder, get output.
    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input)  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input)  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output))  # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output  # the matrix that contain probabilities of words in vocabulary for each words in sequence.(B, L, trg_vocab_size)


# connect multi-layer encoder together

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # loop each encoder blocks
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    # loop each encoder block
    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)  # 经过所有编码器层和最后一次层归一化处理后的序列特征表示


# connect multi-layer encoder together
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    # Loop decoder blocks
    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)
