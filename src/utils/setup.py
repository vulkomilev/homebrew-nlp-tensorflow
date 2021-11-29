import math
import random

import numpy as np

from src.dataclass import Context
from src.model import LinearAttention, Trainer
from src.utils.formatting import pretty_print
import tensorflow as tf


def setup_torch(seed: int):


    random.seed(seed)
    np.random.seed(seed)


def get_model(ctx: Context, load_model: bool) -> Trainer:
    mod = Trainer(LinearAttention(ctx))

    if ctx.model.print_on_init:
        pretty_print(str(mod))

    #parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mod.parameters()))
    #base = int(math.log10(parameters) / 3)
    #pretty_print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')
    if load_model:
        mod.load()

    return mod


def encode(prompt: str) -> tf.Tensor:
    return tf.convert_to_tensor(np.frombuffer(prompt.encode('UTF-8'), dtype=np.uint8))


def decode(output: tf.Tensor) -> str:
    return ''.join(chr(c) for c in output.view(-1).unbind(0))
