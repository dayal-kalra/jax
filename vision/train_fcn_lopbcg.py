# in use imports

import utils.model_utils as model_utils
import utils.train_utils as train_utils
import utils.data_utils as data_utils

import jax
from jax import numpy as jnp
import optax
from flax import linen as nn

from typing import Tuple

#usual imports
import pandas as pd
import argparse
import pickle as pl

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

"""### Model definition and train state definition"""

def linear_warmup(step, lr_init, lr_trgt, warm_steps):
    "lr = lr_max * step / step_max"
    rate = 1.0 / warm_steps
    lr = lr_init + (lr_trgt) * rate * (step)
    return min(lr_trgt, lr)

def create_train_state(config: argparse.ArgumentParser, batch: Tuple):
    x, y = batch

    x = x[:config.batch_size, ...]
    y = y[:config.batch_size, ...]

    # create model
    model = models[config.model](width = config.width, depth = config.depth, out_dim = config.out_dim, use_bias = config.use_bias, varw = config.varw, act_name = config.act_name, scale = config.scale)
    
    # initialize using the init seed
    key = jax.random.PRNGKey(config.init_seed)
    init_params = model.init(key, x)['params']
    
    #debugging: check shapes and norms
    shapes = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    #print(shapes)
    norms = jax.tree_util.tree_map(lambda x: config.width * jnp.var(x), init_params)
    #print(norms)
    
    # create an optimizer
    opt = optax.inject_hyperparams(optax.sgd)(learning_rate = config.lr_trgt, momentum = config.momentum)
    # create a train state
    state = train_utils.TrainState.create(apply_fn = model.apply, params = init_params, opt = opt)

    return state

def exponential_search(state, batch, grads_init, loss_init, loss_next, lr_next = 1e-04):
    """
    Descritpion: Exponential search to find the lr_upper
    Inputs:
        lr_next: has to be greater than zero
    """
    
    frwd_passes = 0 # count the number of extra forward passes

    while loss_next < loss_init:
        
        if lr_next >= config.lr_trgt: break
        # increase the learning rate as the loss increases
        lr_next *= 2.0
        state.update_learning_rate(learning_rate = lr_next)

        # estimate candidate parameters
        updates, opt_state_next = state.opt.update(grads_init, state.opt_state, state.params)
        params_next = optax.apply_updates(state.params, updates)
        #params_next = jax.tree_map(lambda p, g: p - config.lr*g, state.params, grads_step)
        
        # compute the loss at next step
        loss_next = train_utils.loss_step(state, batch, params_next, config.loss_fn)
        frwd_passes += 1
        
        print(f'fp: {frwd_passes}, lr_next: {lr_next:0.4f}, Loss init: {loss_init:0.4f}, Loss next: {loss_next:0.4f}')

    return loss_next, lr_next, frwd_passes

def binary_search_loss(config, state, batch, grads_init, loss_init, loss_upper, lr_lower, lr_upper):

    frwd_passes = 0
    
    loss_lower = loss_init
    print(f'Loss lower: {loss_lower:0.4f}, loss init: {loss_init:0.4f}')

    while loss_upper > loss_init * (1+config.eps):
        
        lr_mid = (lr_lower + lr_upper) / 2.0

        state.update_learning_rate(learning_rate = lr_mid)

        # estimate candidate parameters
        updates, opt_state_next = state.opt.update(grads_init, state.opt_state, state.params)
        params_mid = optax.apply_updates(state.params, updates)

        loss_mid = train_utils.loss_step(state, batch, params_mid, config.loss_fn)
        frwd_passes += 1

        if loss_mid < loss_init:
            lr_lower = lr_mid
            loss_lower = loss_mid
        else:
            lr_upper = lr_mid
            loss_upper = loss_mid
        
        print(f'fp: {frwd_passes}, lr_lower: {lr_lower:0.4f}, lr_upper: {lr_upper:0.4f}, loss_init: {loss_init:0.4f}, loss_upper: {loss_upper:0.4f}')

    return loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes


def binary_search_lr(config, state, batch, grads_init, loss_init, loss_upper, lr_lower, lr_upper):

    frwd_passes = 0

    loss_lower = loss_init
    print(f'Loss lower: {loss_lower:0.4f}, loss init: {loss_init:0.4f}')

    while lr_upper > lr_lower * (1+config.eps):

        lr_mid = (lr_lower + lr_upper) / 2.0

        state.update_learning_rate(learning_rate = lr_mid)

        # estimate candidate parameters
        updates, opt_state_next = state.opt.update(grads_init, state.opt_state, state.params)
        params_mid = optax.apply_updates(state.params, updates)

        loss_mid = train_utils.loss_step(state, batch, params_mid, config.loss_fn)
        frwd_passes += 1

        if loss_mid < loss_init:
            lr_lower = lr_mid
            loss_lower = loss_mid
        else:
            lr_upper = lr_mid
            loss_upper = loss_mid

        print(f'fp: {frwd_passes}, lr_lower: {lr_lower:0.4f}, lr_upper: {lr_upper:0.4f}, loss_init: {loss_init:0.4f}, loss_upper: {loss_upper:0.4f}')

    return loss_lower, loss_upper, lr_lower, lr_upper, frwd_passes


def train_and_evaluate(config: argparse.ArgumentParser, train_ds: Tuple, test_ds: Tuple):
    "train model acording the config"
    
    # create a train state
    state = create_train_state(config, train_ds)
    
    state_fn = state.apply_fn

    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = config.sgd_seed
    rng = jax.random.PRNGKey(seed)
    train_batches = train_utils.data_stream(seed, train_ds, config.batch_size)
    test_batches = train_utils.data_stream(seed, test_ds, config.batch_size)
    
    train_loss_init, test_accuracy_init = train_utils.compute_metrics(state_fn, state.params, config.loss_fn, train_batches, config.num_test, config.batch_size)
    test_loss_init, test_accuracy_init = train_utils.compute_metrics(state_fn, state.params, config.loss_fn, test_batches, config.num_test, config.batch_size)

    step_results = list()
    divergence = False

    # prepare an initial guess for the eigenvectors of the hessian
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)
    key = jax.random.PRNGKey(93)
    vs_init = jax.random.normal(key, shape = (flat_params.shape[0], config.topk))

    batches = train_utils.data_stream(seed, train_ds, config.batch_size, augment = config.use_augment)
    num_batches = config.num_batches # make it a non config variable
    
    ### Zeroth order practical persistent catapult 
   
    xtra_frwd_passes = 0
    
    # this is for a fair comparison with the linear warmup
    config.lr = config.lr_trgt / config.warm_steps
    state.update_learning_rate(learning_rate = config.lr)
    
    print(f'Learning rate before line search: {config.lr:0.6f}')

    # pick a random batch
    batch = next(batches)
    imgs, targets = batch

    # estimate the initial loss and gradients; we need only the gradients at initialization
    grads_init, _, loss_init = train_utils.gradients_step(state, batch, config.loss_fn)
    
    # estimate candidate parameters
    updates, opt_state_next = state.opt.update(grads_init, state.opt_state, state.params)
    params_next = optax.apply_updates(state.params, updates)
    #params_next = jax.tree_map(lambda p, g: p - config.lr*g, state.params, grads_step)
    
    # estimate the loss at candidate parameters
    loss_next = train_utils.loss_step(state, batch, params_next, config.loss_fn)
            
    xtra_frwd_passes += 1
    # start with lr_upper being config.lr
    lr_upper = config.lr

    if config.lr < config.lr_trgt:

        ## Exponential search for the learning rate upper bound
        loss_upper, lr_upper, frwd_passes = exponential_search(state, batch, grads_init, loss_init, loss_next, lr_next = lr_upper)
        lr_lower = lr_upper / 2.0
    
        xtra_frwd_passes += frwd_passes

        ## Binary search / bisection method
        loss_upper, loss_lower, lr_lower, lr_upper, frwd_passes = binary_search_loss(config, state, batch, grads_init, loss_init, loss_upper, lr_lower, lr_upper)
        
        xtra_frwd_passes += frwd_passes
        print(f'lr lower: {lr_lower:0.4f}, lr upper: {lr_upper:0.4f}')
        print(f'Loss init {loss_init:0.4f}, Loss lower: {loss_lower:0.4f}, Loss upper: {loss_upper:0.4f}')
        print(f'Extra forward passes: {xtra_frwd_passes}')

    exit(0)

    ### Training the model

    for step in range(config.num_steps):     
        
        #get the next batch and calculate the step
        batch = next(batches)
        imgs, targets = batch
        
        # update the learning rate in the warmup phase
        if step < config.warm_steps+1:
            config.lr = linear_warmup(state.step+1, lr_upper, config.lr_trgt, config.warm_steps) # state.step + 1 used because there is not training step yet
            state.update_learning_rate(learning_rate = config.lr)
        
        #train for one step
        state, logits_step, loss_step, sharpness_step, vs_step, n_iter = train_utils.train_sharpness_lobpcg_step(state, batch, config.loss_fn, vs_init) # state, logits, loss, eigs, eigvs, n_iter
        sharpness_step = sharpness_step.squeeze()
        # estimate accuracy from logits
        accuracy_step = train_utils.compute_accuracy(logits_step, targets)
        
        result = jnp.array([state.step, config.lr, config.warm_steps, xtra_frwd_passes, loss_step, accuracy_step, sharpness_step, n_iter])
        step_results.append(result)
        #check for divergence
        if (jnp.isnan(loss_step) or jnp.isinf(loss_step)): divergence = True; break
        
        print(f't: {state.step}, lr: {config.lr:0.4f}, loss: {loss_step:0.4f}, accuracy: {accuracy_step:0.4f}, sharpness: {sharpness_step:0.4f}, n_iter: {n_iter}')
        
    step_results = jnp.asarray(step_results)
    step_results = jax.device_get(step_results)

    df = pd.DataFrame(step_results, columns = ['step', 'lr', 'warm_steps', 'forward_passes', 'loss_step', 'accuracy_step', 'sharpness_step', 'n_iter'])
    
    path = f'{save_dir}/dynamics_{config.dataset}_P{config.num_examples}_{config.model}_n{config.width}_d{config.depth}_scale{config.scale}_varw{config.varw}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_{config.opt}_{config.schedule_name}_lrinit{config.lr_init}_lrtrgt{config.lr_trgt:0.4f}_eps{config.eps}_Twarm{config.warm_steps}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}_M{config.measure_batches}_{config.sharpness_method}.tab'
    
    df.to_csv(path, sep = '\t')
    return divergence


# Add more models later on
models = {'fcn_mup': model_utils.fcn_int, 'fcn_sp': model_utils.fcn_sp, 'myrtle_sp': model_utils.Myrtle, 'myrtle_mup': model_utils.Myrtle_int}
#models = {'WideResNet16': model_utils.WideResNet16, 'WideResNet20': model_utils.WideResNet20, 'WideResNet28': model_utils.WideResNet28, 'WideResNet40': model_utils.WideResNet40}
loss_fns = {'mse': train_utils.mse_loss, 'xent': train_utils.cross_entropy_loss}
activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
# Dataset parameters
parser.add_argument('--dataset', type = str, default = 'cifar10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--num_examples', type = int, default = 50000)
# Model parameters
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--width', type = int, default = 512)
parser.add_argument('--depth', type = int, default = 4)
parser.add_argument('--varw', type = float, default = 2.0)
parser.add_argument('--scale', type = float, default = 0.0)
parser.add_argument('--bias', type = str, default = 'True') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 1)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'mse')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--num_steps', type = int, default = 2048)
parser.add_argument('--warm_steps', type = int, default = 512)
parser.add_argument('--lr_init', type = float, default = 1e-06)
parser.add_argument('--lr_trgt', type = float, default = 0.1)
parser.add_argument('--eps', type = float, default = 0.1)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 512)
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--test', type = str, default = 'True')
parser.add_argument('--measure_batches', type = int, default = 10)
parser.add_argument('--sharpness_method', type = str, default = 'lobpcg')

config = parser.parse_args()

# Model parameters
config.model = f'fcn_{config.abc}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False
config.act = activations[config.act_name]

config.loss_fn = loss_fns[config.loss_name]
# Optimization parameters
config.schedule_name = 'initv1_linear'
config.measure_batches = int(4096.0 / config.batch_size)

print(config)

save_dir = 'fcn_results'
data_dir = '/home/dayal'

(x_train, y_train), (x_test, y_test) = data_utils.load_image_data(data_dir, config.dataset, flatten = False, num_examples = config.num_examples)

config.num_train, config.num_test = x_train.shape[0], x_test.shape[0]

# #standardize the inputs
x_train = data_utils._standardize(x_train, abc = config.abc)
x_test = data_utils._standardize(x_test, abc = config.abc)

config.in_dim = int(jnp.prod(jnp.array(x_train.shape[1:])))

#get one hot encoding for the labels
y_train = data_utils._one_hot(y_train, config.out_dim)
y_test = data_utils._one_hot(y_test, config.out_dim)

config.num_batches = train_utils.estimate_num_batches(config.num_train, config.batch_size)
divergence = False

# train the model
divergence = train_and_evaluate(config, (x_train, y_train), (x_test, y_test))
