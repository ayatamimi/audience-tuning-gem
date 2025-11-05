import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn
import numpy as np


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
# Pytorch implementation borrowed from https://github.com/rosinality/vq-vae-2-pytorch and modified 



class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        #First initialization of codebooks, currently set to intialize with some input, for test you need to set this to False
        self.initialized = True
        # Initialize tracking for unused codebooks
        window_size = 1
        self.usage_history = []
        self.usage_threshold = window_size

        self.batch_avg_buffer = torch.zeros((window_size, dim), device=embed.device)
        self.batch_avg_index = 0
        self.batch_avg_count = 0

    def forward(self, input):
        if not self.initialized:
            self.initialize_codebook(input)

        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        num_used_codebooks = -1
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            del dist, embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            del embed_onehot_sum, embed_sum

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # Track codebook usage
            unique_indices= torch.unique(torch.flatten(embed_ind, start_dim=1), return_counts=False)
            num_used_codebooks = unique_indices.shape[0]
            self.usage_history.append(num_used_codebooks)
            if len(self.usage_history) > self.usage_threshold:
                self.usage_history.pop(0)

            # Reinitialize unused codebooks if conditions are met
            if len(self.usage_history) == self.usage_threshold:
                if self.usage_history[-1] <= self.usage_history[0] and num_used_codebooks < self.n_embed:
                    unused_codebooks = torch.ones(self.n_embed, dtype=torch.bool)
                    unused_codebooks[unique_indices] = False
                    unused_codebook_indices = torch.where(unused_codebooks)[0]
                    if len(unused_codebook_indices) > 0:
                        avg_current_batch = torch.mean(flatten, dim=0)
                        self.embed.data[:, unused_codebook_indices] = avg_current_batch.unsqueeze(1)


        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind, num_used_codebooks

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
    def initialize_codebook(self, input):
        flatten = input.reshape(-1, self.dim)
        self.embed.data.copy_(flatten[:self.n_embed].t())
        self.embed_avg.data.copy_(self.embed.data)
        self.initialized = True


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


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
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


class FlatVQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=64,
        n_res_block=2,
        n_res_channel=72,
        embed_dim=16,
        n_embed=16,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.dec = Decoder(
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
        quant_b, diff, _ ,diversity_penalty,codebook_usage= self.encode(input)
        dec = self.decode(quant_b)

        return dec, diff,diversity_penalty,codebook_usage

    def encode(self, input):
        enc_b = self.enc_b(input)
        quant_b = enc_b.permute(0, 2, 3, 1)
        quant_b, diff_b, id_b, num_used_codebooks = self.quantize_b(quant_b)
        diversity_penalty = max(0.0, self.diversity_threshold - num_used_codebooks)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_b, diff_b, id_b, diversity_penalty, num_used_codebooks

    def decode(self, quant_b):
        dec = self.dec(quant_b)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        return dec

    
