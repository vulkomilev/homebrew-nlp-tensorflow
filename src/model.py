import copy
import typing

import numpy as np
import revlib
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from src.optimizers.build import build_optimizer
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from src.dataclass import Context
#tf.compat.v1.enable_eager_execution()
def split_norm(inp: tf.Tensor) -> tf.Tensor:
    scale0, scale1, shift = tf.split(inp,3, 1)
    return tf.norm(tf.add(tf.multiply(scale0 , scale1) , shift))

def norm(out: tf.Tensor) -> tf.Tensor:
    out = out - out.mean(1, keepdim=True)
    return tf.divide(out,  (tf.add(tf.math.pow(tf.multiply (tf.norm(out, (2, 1)) , out.size(1)) , -0.5 ), 1e-5)))


def conv(inp: tf.Tensor, weight: tf.Tensor, groups: int, use_pad: bool) -> tf.Tensor:
    if use_pad and weight.size()[-1] - 1 > 0:
        inp = tf.pad(inp, (weight.size()[-1] - 1, 0))
    local_conv1d = tf.keras.layers.Conv1D(kernel_initializer=weight,groups=groups)
    return local_conv1d(inp)

def drop_conv(inp: tf.Tensor, weight: tf.Tensor, p: float, train: bool, groups: int, pad: bool) -> tf.Tensor:
    batch, features, sequence = inp.size()
    if 0 < p < 1:
        if train:
            mask = tf.random.uniform((features,))
            inp = tf.boolean_mask(inp, mask.view(1, -1, 1)).view(batch, -1, sequence)
            weight = tf.boolean_mask(weight, mask.view(-1, 1, 1)).view(-1, weight.size(1), weight.size(2))
        elif tf.size(inp) > tf.size(weight):
            weight = tf.multiply(weight , p)
        else:
            inp = tf.multiply(inp , p)
    return conv(inp, weight, groups, pad)

def orthonormal(inp: typing.Union[tf.Tensor, tf.Variable, typing.List[int]], gain: float):
    original_input = inp
    if isinstance(inp, list):
        inp = tf.zeros(inp)
    if isinstance(inp, tf.Variable):
        inp = inp
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    g1 = tf.random.Generator.from_seed(1)

    a = g1.normal(flat_shape)
    u, _, v = tf.linalg.svd(a, full_matrices=False)

    inp = (u if u.shape == flat_shape else v)*gain
    if isinstance(original_input, list):
        return tf.Variable(inp)
    return original_input

def moe(inp: tf.Tensor, w: typing.List[tf.Variable],
        gate: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    local_conv1d = tf.keras.layers.Conv1D(kernel_initializer=gate)
    out = local_conv1d(inp)
    gates = tf.nn.softmax(out, dim=1)
    one_hot = tf.one_hot(tf.argmax(out, dim=1), out.shape[1])
    gumbel = one_hot.transpose(1, 2) - gates.detach() + gates
    inp_t = inp.transpose(1, 2)
    batch, features, sequence = inp.size()
    out = tf.zeros((batch * sequence, w[0].size(1)),  dtype=inp.dtype)
    for expert, g, param in zip(one_hot.unbind(-1), gumbel.unbind(1), w):
        tmp =tf.boolean_mask(inp_t * g.unsqueeze(2), expert.unsqueeze(2)).view(-1, features).mm(param)
        out = out.boolean_mask(expert.view(-1, 1), tmp)
    loss = tf.math.reduce_sum(tf.math.reduce_mean(gates, dim=(0, 2)) * tf.math.reduce_mean(one_hot.float(), dim=(0, 1)))
    return loss, out.view(batch, sequence, -1).transpose(1, 2)

def moe_check(inp: tf.Tensor, w_gate: tf.Tensor, w: typing.List[tf.Variable], dropout_probability: float,
              training: bool, groups: int) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    if w:
        return moe(inp, w, w_gate)
    return (tf.zeros([1], device=inp.device, dtype=inp.dtype),
            drop_conv(inp, w_gate, dropout_probability, training, groups, False))

def linear_attention(inp: tf.Tensor, divisor: tf.Tensor, w0_gate: tf.Tensor,
                     w0: typing.List[tf.Variable], w1: tf.Tensor, w2_gate: tf.Tensor,
                     w2: typing.List[tf.Variable], input_cache: tf.Tensor, cumsum_cache: tf.Tensor,
                     init_scale: float, bottleneck_group: int, dropout_probability: float, training: bool,
                     caching: bool, idx: int
                     ) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    kernel_size = w1.shape(2)
    pad = True
    if not training and caching:
        if idx - 1 > kernel_size and inp.size(2) == 1:
            pad = False
            inp = tf.concat([input_cache, inp], -1)
        input_cache = tf.Tensor(inp[:, :, -kernel_size + 1:])
    loss0, inp = moe_check(inp, w0_gate, w0, dropout_probability, training, 1)
    depth, scale, shift =  tf.split(inp,3, 1)
    cum = tf.math.cumsum(depth,axis=-1)
    if not training and caching:
        cum = cum + cumsum_cache
        scale = scale[:, :, -1:]
        shift = shift[:, :, -1:]
        cum = cum[:, :, -1:]
        if idx - 1 > kernel_size:
            cumsum_cache = cum
    leaky_layer = tf.keras.layers.LeakyReLU(alpha=0.02)
    inp = leaky_layer((tf.norm(tf.divide (cum , tf.add(tf.matmul (divisor , scale) , shift)))))
    inp = drop_conv(inp, w1, dropout_probability, training, bottleneck_group, pad)
    inp = leaky_layer(split_norm(inp), 0.02)
    cumsum_cache = tf.Tensor(cumsum_cache)
    loss1, inp = moe_check(inp, w2_gate, w2, dropout_probability, training, 1)

    return loss0, loss1, input_cache, cumsum_cache, tf.matmul(inp , init_scale)

def get_coupling(beta_tmp: float):
    def momentum_coupling_forward(other_stream: tf.Tensor, fn_out: tf.Tensor, beta: float) -> tf.Tensor:
        return tf.add(tf.matmul(other_stream , beta) , fn_out)

    def momentum_coupling_inverse(output: tf.Tensor, fn_out: tf.Tensor, beta: float) -> tf.Tensor:
        return (output - fn_out) / beta

    def _wrapped_momentum_coupling_forward(x, y):
        return momentum_coupling_forward(x, y, beta_tmp)

    def _wrapped_momentum_coupling_inverse(x, y):
        return momentum_coupling_inverse(x, y, beta_tmp)

    return _wrapped_momentum_coupling_forward, _wrapped_momentum_coupling_inverse

def in_features(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    local_conv1d = tf.keras.layers.Conv1D(out_features,(kernel_size,), groups=groups)
    return orthonormal(local_conv1d.weight, 1 / std)
def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    local_conv = tf.keras.layers.Conv1D(in_features, out_features, (kernel_size,), groups=groups)
    local_conv.build(in_features)
    return orthonormal( local_conv.kernel, 1 / std)

class Trainer(object):
    def __init__(self,model):
        super(Trainer, self).__init__()

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()

    def softargmax(self,x, beta=1e10):
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)
    @tf.function
    def _forward_backward(self, src_arr: tf.Tensor, tgt_arr: tf.Tensor) -> tf.Tensor:

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss = 0
            for (s, t), _ in zip(src_arr, tgt_arr):
                src = s.squeeze(0)
                tgt = t.squeeze(0)
                model_out = self.model(np.array(src))
                local_tgt = []

                #for row in tgt:
                #    lc = [0.0] * 256
                #    lc[np.argmax(row).numpy()] = 1.0
                #    local_tgt.append(lc)

                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(tgt,model_out)
                #loss += tf.keras.losses.CategoricalCrossentropy()(model_out[:,1],local_tgt)

            gradients = tape.gradient(loss,  self.model.trainable_variables)
        print('loss',loss.numpy())

        print('-------------------------------')
        #print(model_out)
        return gradients,loss

    def _clip_gradient(self,gradients):
        for p in gradients:
            if p is None:
                continue
            if type(p) == tf.IndexedSlices:
               p = p.values
            len_p = 0
            p = p[:1000]
            for row in p:
                len_p += 1
                g_norm = tf.clip_by_value(row,clip_value_min=self.model.ctx.optimizer.agc.zero_division_eps,clip_value_max=1000)
                p_norm = tf.clip_by_value(row,clip_value_min=self.model.ctx.optimizer.agc.eps,clip_value_max=1000)
                grad_scale = tf.clip_by_value((p_norm / g_norm * self.model.ctx.optimizer.agc.gradient_clipping),clip_value_min=-1000,clip_value_max=1)
                row = row* grad_scale


    def accumulated_step(self, dataloader) -> tf.Tensor:

        gradients,loss = self._forward_backward(dataloader, range(self.model.ctx.optimizer.gradient_accumulation_steps))
        # add sum into the self.__forward_backward gradient decent
        #sum(self._forward_backward(s.squeeze(0), t.squeeze(0)) for (s, t), _ in  zip(dataloader, range(self.model.ctx.optimizer.gradient_accumulation_steps)))

        #print( "Gradients")
        #print( gradients)

        #self.model.ctx.optimizer()
        last_gradient_arr = []

        if 'gradients_vars' in dir(self):

            for element in self.gradients_vars:
                if element is not None:
                    if type(element) == tf.IndexedSlices:
                        if 'prev_step' in dir(element.values):
                         last_gradient_arr.append(element.values.prev_step)
                    else:
                        if 'prev_step' in dir(element):
                           last_gradient_arr.append(element.prev_step)

        self.gradients_vars  = gradients
        if len(last_gradient_arr) == len(self.gradients_vars):
            for g,p,i in zip(last_gradient_arr,self.gradients_vars,range(len(self.gradients_vars))):
                if type(self.gradients_vars[i]) == tf.IndexedSlices:
                    p.values.prev_step = g
                else:
                    p.prev_step = g

                self.gradients_vars[i] = p
        self._clip_gradient(gradients)
        return loss

    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    def gradients(self) -> tf.Variable:
        for p in self.gradients_vars:
            if type(p) == tf.IndexedSlices:
                p = p.values
            yield p
    def  set_gradients(self,index,gradient) -> None:
            self.gradients_vars[index]  =gradient
    def save(self):
        pass
        # implement the save function

    def load(self):
        pass
        # implement the load function

class MomentumNetSide():
    def __init__(self, beta: float):

        self.beta = beta

    def forward(self, inp: tf.Tensor):
        return tf.matmul(inp , self.beta)
'''
class LinearAttention(tf.keras.Model):
    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()
        self.ctx = ctx
        self.embedding =tf.keras.layers.Embedding(ctx.dataset.classes, ctx.model.features * 2)
        self.embedding.build(ctx.dataset.classes)

        orthonormal(self.embedding.embeddings, ctx.model.input_embedding_std * 2 ** -0.5)

        pos_embd = tf.range(0, ctx.model.sequence_length)
        #self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float).to(ctx.model.device))

        self.cell = LinearAttentionCell(self, ctx, 1)

        local_conv1d = tf.keras.layers.Conv1D(filters=ctx.dataset.classes, kernel_size=(1,))
        self.local_output = local_conv1d

    def call(self, inp: tf.Tensor,traing=None,mask=None):
        return self.local_output(self.cell(self.embedding(inp).transpose()))

    def reset_cache(self):
        for mod in self.stem.modules():
            if isinstance(mod, LinearAttentionCell):
                mod.reset_cache()
'''
class LinearAttention(tf.keras.Model):
    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()
        self.ctx = ctx
        self.embedding =tf.keras.layers.Embedding(ctx.dataset.classes, ctx.model.features * 2)
        self.embedding.build(ctx.dataset.classes)

        orthonormal(self.embedding.embeddings, ctx.model.input_embedding_std * 2 ** -0.5)

        pos_embd = tf.range(0, ctx.model.sequence_length)
        #self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float).to(ctx.model.device))

        self.cell = LinearAttentionCell(self, ctx, 1)
        self.local_lstm = tf.keras.layers.LSTM(256, return_sequences=True)
        local_conv1d = tf.keras.layers.Dense(256)#tf.keras.layers.Conv1D(filters=ctx.dataset.classes, kernel_size=(1,))#tf.keras.layers.Dense(256)#
        self.local_output = local_conv1d#local_conv1d

    def call(self, inp: tf.Tensor,traing=None,mask=None):
        #return self.embedding(inp).transpose()
        return  self.local_lstm(self.local_output(self.cell(self.embedding(inp).transpose())))

    def reset_cache(self):
        for mod in self.stem.modules():
            if isinstance(mod, LinearAttentionCell):
                mod.reset_cache()
class ParameterStore(object):
    """
    Something (likely deepspeed) changes all parameters in a ParameterList to [1] even though standalone parameters
    work. That's why a torch.nn.ModuleList of ParameterStores needs to be initialized.
    """

    def __init__(self, param: tf.Tensor):
        super(ParameterStore, self).__init__()
        self.param = tf.Variable(param)

    def __repr__(self):
        return (f'{self.__class__.__name__}(shape={str(list(self.param.size()))}, device={self.param.device}, '
                f'dtype={self.param.dtype})')

class AuxLoss():
    def forward(self,ctx, inp: tf.Tensor):
        ctx.save_for_backward(inp)
        return inp

    def backward(segl,ctx, grad_outputs: tf.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()

class LinearAttentionCell(tf.keras.layers.Layer):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.dropout_probability = 1 - ctx.model.dropout_probability
        self.bottleneck_group = ctx.model.bottleneck_group
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        experts = ctx.model.moe.num_experts
        moe_in_output = ctx.model.moe.use_in_output
        moe_in_input = ctx.model.moe.use_in_input
        param0 = ParameterStore(orthonormal([ctx.model.features, intermediate * 3], ctx.model.activation_std))
        param2 = ParameterStore(orthonormal([intermediate, ctx.model.features], 1))
        self.w0_gate = conv_weight(ctx.model.features, experts if moe_in_input else (3 * intermediate), 1, 1, 1)
        self.w0 = [copy.deepcopy(param0) for _ in range(experts * moe_in_input)]
        self.w1 = conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
                              ctx.model.activation_std)
        self.w2_gate = conv_weight(intermediate, experts if moe_in_output else ctx.model.features, 1, 1, 1)
        self.w2 = [copy.deepcopy(param2) for _ in range(experts * moe_in_output)]
        # Below is done to ignore pytorch's errors when calling .register_buffer without giving up the IDEs autocomplete
        self.idx: int = 0
        self._input_cache = tf.zeros([])
        self._cumsum_cache = tf.zeros([])

    def reset_cache(self):
        self._cumsum_cache = tf.zeros([])
        self._input_cache = tf.zeros([])
        self.idx = 0

    def forward(self, inp: tf.Tensor) -> tf.Tensor:
        if self.training:
            div = self.divisor()
        elif self.caching:
            self.idx += inp.size(2)
            div = tf.Tensor([self.idx])
        else:
            self.idx = inp.size(2)
            div = tf.range(self.idx, device=inp.device).view(1, 1, -1) + 1
        loss0, loss1, self._input_cache, self._cumsum_cache, out = linear_attention(inp,
                                                                                    div,
                                                                                    self.w0_gate,
                                                                                    [store.param for store in self.w0],
                                                                                    self.w1,
                                                                                    self.w2_gate,
                                                                                    [store.param for store in self.w2],
                                                                                    self._input_cache,
                                                                                    self._cumsum_cache,
                                                                                    self.init_scale,
                                                                                    self.bottleneck_group,
                                                                                    self.dropout_probability,
                                                                                    self.training, self.caching,
                                                                                    self.idx)
        AuxLoss.apply(loss0 + loss1)
        return out

    def momentum(self, init_scale: float):
        out = copy.deepcopy(self)
        out.init_scale = init_scale
        return out
