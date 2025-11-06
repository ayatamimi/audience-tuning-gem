import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class EncoderEnhanced(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class DecoderEnhanced(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    



class EnhancedFlatVQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=144,
        n_res_block=2,
        n_res_channel=72,
        embed_dim=64,
        codebook_dim=64,
        n_embed=456,
        decay=0.99,
        rotation_trick=False,
        kmeans_init=False,
        learnable_codebook = False,
        ema_update = True,
        threshold_ema_dead_code = 2
    ):
        super().__init__()

        self.enc_b = EncoderEnhanced(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = VectorQuantize(dim=embed_dim, codebook_dim=codebook_dim, codebook_size=n_embed, decay=decay,learnable_codebook=learnable_codebook,ema_update=ema_update, rotation_trick=rotation_trick, kmeans_init = kmeans_init, threshold_ema_dead_code=threshold_ema_dead_code)
        self.dec = DecoderEnhanced(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.vocab_size = n_embed
        self.diversity_threshold=int(n_embed)

    def forward(self, input):
        quant_b, diff, indices_b = self.encode(input)
        dec = self.decode(quant_b)
        return dec, diff, indices_b

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_b = self.quantize_conv_b(enc_b)
        B, C, H, W = enc_b.shape
        quant_b = enc_b.permute(0, 2, 3, 1).reshape(B, H * W, C)
        quant_b,id_b, diff_b = self.quantize_b(quant_b)
        quant_b = quant_b.view(B, H, W, C).permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_b, diff_b, id_b

    def decode(self, quant_b):
        dec = self.dec(quant_b)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        return dec