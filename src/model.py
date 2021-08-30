import math
import typing

import numpy as np
import revlib
import torch
import torch.nn.functional

from src.dataclass import Context

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter], gain: float):
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
    return inp


@torch.jit.script
def norm(out: torch.Tensor) -> torch.Tensor:
    out = out - out.mean(1, keepdim=True)
    return out / (torch.norm(out, 2, 1, True) * out.size(1) ** -0.5 + 1e-5)


@torch.jit.script
def conv(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pad = weight.size()[-1] - 1
    if pad:
        inp = torch.nn.functional.pad(inp, (pad, 0))
    return torch.nn.functional.conv1d(inp, weight)


@torch.jit.script
def drop_conv(inp: torch.Tensor, weight: torch.Tensor, p: float, train: bool) -> torch.Tensor:
    batch, features, sequence = inp.size()
    if 0 < p < 1:
        if train:
            mask = torch.randn((features,), device=inp.device, dtype=inp.dtype) < p
            inp = torch.masked_select(inp, mask.view(1, -1, 1)).view(batch, -1, sequence)
            weight = torch.masked_select(weight, mask.view(-1, 1, 1)).view(-1, weight.size(1), weight.size(2))
        elif torch.numel(inp) > torch.numel(weight):
            weight = weight * p
        else:
            inp = inp * p
    return conv(inp, weight)


@torch.jit.script
def feed_forward(inp: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, p: float,
                 train: bool) -> torch.Tensor:
    inp = drop_conv(inp, w0, p, train)
    inp = torch.relu(inp)
    inp = drop_conv(inp, w1, p, train)
    inp = torch.relu(inp)
    inp = drop_conv(inp, w2, p, train)
    return inp


class FeedForward(torch.nn.Module):
    def __init__(self, ctx: Context, init_scale: float):
        super().__init__()
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        self.w0 = torch.nn.Conv1d(ctx.model.features, intermediate, (1,), bias=False).weight
        self.w1 = torch.nn.Conv1d(intermediate, intermediate, (ctx.model.conv_kernel_size,), bias=False).weight
        self.w2 = torch.nn.Conv1d(intermediate, ctx.model.features, (1,), bias=False).weight
        orthonormal(self.w0, 1 / ctx.model.activation_std)
        orthonormal(self.w1, 1 / ctx.model.activation_std)
        orthonormal(self.w2, init_scale)
        self.dropout_probability = 1 - ctx.model.dropout_probability

    def forward(self, inp: torch.Tensor):
        return feed_forward(inp, self.w0, self.w1, self.w2, self.dropout_probability, self.training)


@torch.jit.script
def linear_attention(depth: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                     divisor: torch.Tensor, init_scale: float) -> torch.Tensor:
    return norm(depth.cumsum(1) / divisor * scale + shift) * init_scale


def get_coupling(beta: float):
    @torch.jit.script
    def momentum_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
        return other_stream * beta + fn_out * (1 - beta)

    @torch.jit.script
    def momentum_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
        return (output - fn_out * (1 - beta)) / beta

    return momentum_coupling_forward, momentum_coupling_inverse


class LinearAttention(torch.nn.Module):
    """
    One idea would be to run linear attention at every step in an rnn
    """

    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()

        self.embedding = torch.nn.Embedding(ctx.dataset.classes, ctx.model.features * 2)
        self.embedding.weight.data.mul_(ctx.model.input_embedding_std * 2 ** -0.5)

        init_scale = ctx.model.depth ** -0.5
        pos_embd = torch.arange(0, ctx.model.sequence_length).unsqueeze(0) + 1
        feature_embd = torch.arange(0, ctx.model.features).unsqueeze(1) + 1
        additive = (feature_embd % 2).to(torch.float)
        feature_embd = (feature_embd - additive) / 2
        additive *= math.pi
        feature_embd *= 8 / ctx.model.features
        feature_embd -= math.log(ctx.dataset.classes / 2 / math.pi)
        feature_embd = torch.exp(feature_embd) + additive
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float))
        feature_embd = torch.sin(pos_embd * feature_embd).mul(ctx.model.position_embedding_std * 2 ** -0.5).unsqueeze(0)
        self.register_buffer("pos_embd", feature_embd)

        momentum_coupling_forward, momentum_coupling_inverse = get_coupling(ctx.model.momentumnet_beta)
        self.stem = revlib.ReversibleSequential(*([layer
                                                   for _ in range(ctx.model.depth)
                                                   for layer in [LinearAttentionCell(self, ctx, init_scale),
                                                                 torch.nn.Identity()]
                                                   ] * ctx.model.weight_shared_blocks),
                                                coupling_forward=[momentum_coupling_forward,
                                                                  revlib.additive_coupling_forward],
                                                coupling_inverse=[momentum_coupling_inverse,
                                                                  revlib.additive_coupling_inverse])
        self.output = torch.nn.Conv1d(ctx.model.features * 2, ctx.dataset.classes, (1,))
        torch.nn.init.zeros_(self.output.weight.data)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        return torch.nn.functional.cross_entropy(self.output(self.stem(self.embedding(inp).transpose(1, 2))), tgt)


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.pos_embd = lambda: base.pos_embd
        self.divisor = lambda: base.divisor
        self.depth = FeedForward(ctx, 1)
        self.scale = FeedForward(ctx, 2 ** -0.5)
        self.shift = FeedForward(ctx, 2 ** -0.5)
        self.init_scale = init_scale

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp + self.pos_embd()
        return linear_attention(self.depth(out), self.scale(out), self.shift(out), self.divisor(), self.init_scale)