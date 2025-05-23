import torch
import torch.nn as nn

from models.encoder import Encoder, EncoderLayer,  DynamicConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer, DynamicAttention
from models.embed import DataEmbedding


class Dyformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Dyformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        if attn == 'prob':
            Attn = ProbAttention
        elif attn == 'dynamic':  # 新增动态注意力机制选项
            Attn = DynamicAttention
        else:
            Attn = FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                DynamicConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

        # 新增：门控机制的参数
        self.gate_encoder = nn.Parameter(torch.ones(e_layers))
        self.gate_decoder = nn.Parameter(torch.ones(d_layers))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        seq_len = enc_out.size(1)

        attns_list = []
        for i, layer in enumerate(self.encoder.layers):
            enc_out_residual = enc_out
            enc_out, attns = layer(enc_out, attn_mask=enc_self_mask)
            # 新增：门控机制
            gate_value = torch.sigmoid(self.gate_encoder[i])
            enc_out = enc_out_residual + gate_value * enc_out
            attns_list.append(attns)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        for i, layer in enumerate(self.decoder.layers):
            dec_out_residual = dec_out
            dec_out = layer(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            # 新增：门控机制
            gate_value = torch.sigmoid(self.gate_decoder[i])
            dec_out = dec_out_residual + gate_value * dec_out

        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns_list
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
