import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, encoder_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = encoder_layers  # 添加 layers 属性
        self.conv_layers = conv_layers
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, attn_mask=attn_mask)
            if self.conv_layers is not None and i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, dropout=0.0, activation='gelu'):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, attn = self.attention_layer(x, x, x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x, attn


class ConvLayer(nn.Module):
    def __init__(self, d_model):
        super(ConvLayer, self).__init__()
        self.down_conv = nn.Conv1d(in_channels=d_model,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, T, E]
        x_stack = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s = x[:, -inp_len:, :]
            x_s = encoder(x_s, attn_mask)
            x_stack.append(x_s)
        x_stack = torch.cat(x_stack, -2)

        return x_stack

# 新增动态卷积层定义
class DynamicConvLayer(nn.Module):
    def __init__(self, d_model, kernel_size=3, reduction=4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.reduction = reduction

        # 动态生成卷积核参数
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.GELU(),
            nn.Linear(d_model // reduction, kernel_size * 1 * 1)  # 输出kernel参数
        )
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1)  # 用于最终特征融合

    def forward(self, x):
        # x: [Batch, SeqLen, d_model]
        B, L, _ = x.shape

        # 生成动态卷积核 [B, kernel_size]
        kernel_params = self.attention(x.mean(dim=1))  # 全局平均后生成参数
        kernel = F.softmax(kernel_params.view(B, 1, self.kernel_size), dim=-1)

        # 应用动态卷积
        x = x.permute(0, 2, 1)  # [B, d_model, L]
        x = F.conv1d(x, kernel, padding=(self.kernel_size - 1) // 2, groups=B)
        x = self.conv(x.permute(0, 2, 1))  # 恢复形状并融合特征

        return x