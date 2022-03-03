import torch
import wandb

from src.dataclass import Context
from src.dataset import get_dataset
from src.utils.formatting import WandbLog
from src.utils.setup import get_model
import tensorflow as tf

import tensorflow.compat.v1 as tf1

def _model_fn(features, labels, mode, params):
    ctx = params['ctx']
    load_model = params['load_model']
    steps = params['steps']
    wandb.init(project=ctx.log.wandb.project, entity=ctx.log.wandb.entity, config=ctx.serialize())
    ctx = Context(wandb.config)

    mod = get_model(ctx, load_model)
    #    wandb.watch(mod, log=ctx.log.wandb.model_log_type, log_freq=ctx.log.wandb.log_frequency)

    data = get_dataset(ctx)
    log = WandbLog(ctx, len(data))
    mean_loss = tf.zeros([])  # , device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)
    local_adam = tf.keras.optimizers.Adam()
    i = 0
    cluster_resolver = tf1.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    print("All devices: ", tf1.config.list_logical_devices('TPU'))
    tpu_config = tf1.estimator.tpu.TPUConfig(iterations_per_loop=10)
    config = tf1.estimator.tpu.RunConfig(
        cluster=cluster_resolver,
        save_checkpoints_steps=None,
        tpu_config=tpu_config)

    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    while True:
        i += 1

        loss = mod.accumulated_step(data,strategy)
        if ctx.optimizer.sharpness_aware_minimization.enabled:
            for j, p in enumerate(mod.gradients()):
                if p is not None:
                    if ctx.optimizer.sharpness_aware_minimization.adaptive:
                        p *= tf.math.square(p)
                    p *= ctx.optimizer.sharpness_aware_minimization.step_size
                    p = tf.math.add(p, p)
                    p.prev_step = p
                    mod.set_gradients(j, p)
            loss = mod.accumulated_step(data,strategy)
        local_adam.apply_gradients(zip(mod.gradients_vars, mod.model.trainable_variables))
        # mod.optimizer.step()
        # FIX THIS
        # mod.scheduler.step()
        # for p in mod.optimizer.param_groups:  # OneCycle resets beta2 to 0.990
        #    p['betas'] = p['betas'][0], mod.ctx.optimizer.beta2
        # print(mean_loss, loss)
        mean_loss += loss
        # with torch.no_grad():
        #    if mod.ctx.log.loss_steps_per_print and i % mod.ctx.log.loss_steps_per_print == 0:
        #        log(mean_loss, mod.optimizer.param_groups[0]['lr'], mod.optimizer.param_groups[0]['betas'])
        #        mean_loss.zero_()
        #    if mod.ctx.model.steps_per_checkpoint and i % mod.ctx.model.steps_per_checkpoint == 0:
        #        mod.save()
        print(i, '/', steps)
        if steps and i > steps:
            return


    logits = tf1.layers.Dense(1)(features)
    loss = tf1.losses.mean_squared_error(labels=labels, predictions=logits)
    optimizer = tf1.train.AdagradOptimizer(0.05)

    train_op = optimizer.minimize(loss, global_step=tf1.train.get_global_step())

    params['batch_size']
    return tf1.estimator.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

def train_model(ctx: Context, steps=None, load_model: bool = False):
    wandb.init(project=ctx.log.wandb.project, entity=ctx.log.wandb.entity, config=ctx.serialize())
    ctx = Context(wandb.config)

    mod = get_model(ctx, load_model)
#    wandb.watch(mod, log=ctx.log.wandb.model_log_type, log_freq=ctx.log.wandb.log_frequency)

    data = get_dataset(ctx)
    log = WandbLog(ctx, len(data))
    mean_loss = tf.zeros([])#, device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)
    local_adam = tf.keras.optimizers.Adam()
    i = 0
    cluster_resolver = tf1.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    print("All devices: ", tf1.config.list_logical_devices('TPU'))
    tpu_config = tf1.estimator.tpu.TPUConfig(iterations_per_loop=10)
    config = tf1.estimator.tpu.RunConfig(
        cluster=cluster_resolver,
        save_checkpoints_steps=None,
        tpu_config=tpu_config)
    estimator = tf1.estimator.tpu.TPUEstimator(
        model_fn=_model_fn,
        config=config,
        train_batch_size=8,
        eval_batch_size=8)

    while True:
        i += 1

        loss = mod.accumulated_step(data)
        if ctx.optimizer.sharpness_aware_minimization.enabled:
                for j,p in enumerate(mod.gradients()):
                    if p is not None:
                        if ctx.optimizer.sharpness_aware_minimization.adaptive:
                            p *= tf.math.square(p)
                        p *= ctx.optimizer.sharpness_aware_minimization.step_size
                        p  = tf.math.add(p,p)
                        p.prev_step = p
                        mod.set_gradients(j,p)
                loss = mod.accumulated_step(data)
        local_adam.apply_gradients(zip(mod.gradients_vars, mod.model.trainable_variables))
        #mod.optimizer.step()
            # FIX THIS
        #mod.scheduler.step()
        # for p in mod.optimizer.param_groups:  # OneCycle resets beta2 to 0.990
        #    p['betas'] = p['betas'][0], mod.ctx.optimizer.beta2
        #print(mean_loss, loss)
        mean_loss += loss
        #with torch.no_grad():
        #    if mod.ctx.log.loss_steps_per_print and i % mod.ctx.log.loss_steps_per_print == 0:
        #        log(mean_loss, mod.optimizer.param_groups[0]['lr'], mod.optimizer.param_groups[0]['betas'])
        #        mean_loss.zero_()
        #    if mod.ctx.model.steps_per_checkpoint and i % mod.ctx.model.steps_per_checkpoint == 0:
        #        mod.save()
        print(i,'/',steps)
        if steps and i > steps:
            return
