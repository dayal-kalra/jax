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

# for deterministic gpu computations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

def create_random_dataset(num_examples, in_dims, out_dim, seed = 63):
    "Create random normal data"
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, shape = (num_examples, *in_dims))
    key, _ = jax.random.split(key)
    y = jax.random.normal(key, shape = (num_examples, out_dim))
    y = jnp.argmax(y, axis = 1)
    return (x, y)

"""Model definition and train state definition"""

def create_train_state(config: argparse.ArgumentParser, batch: Tuple):
    x, y = batch

    x = x[:config.batch_size, ...]
    y = y[:config.batch_size, ...]

    # create model
    model = models[config.model](num_filters = config.width, widening_factor = config.widening_factor, num_classes = config.out_dim, act = config.act)

    # initialize using the init seed
    key = jax.random.PRNGKey(config.init_seed)
    init_params = model.init(key, x)['params']
    
    # debugging: check shapes and norms
    shapes = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    #print(shapes)
    norms = jax.tree_util.tree_map(lambda x: config.width * jnp.var(x), init_params)
    #print(norms)
    
    # count the number of parameters
    num_params = model_utils.count_parameters(init_params)
    print(f'The model has {num_params/1e6:0.4f}M parameters')

    # create an optimizer
    #opt = optax.inject_hyperparams(optax.sgd)(learning_rate = config.lr_trgt, momentum = config.momentum)
    opt = optax.sgd(config.lr_trgt, config.momentum)
    
    # create a train state
    state = train_utils.TrainState.create(apply_fn = model.apply, params = init_params, opt = opt)

    return state, num_params

def select_random_subset(x, y, batch_size, rng_key):
    """ Selects a random subset of batch size from a dataset (x, y) """
    # Generate a shuffled range of indices
    indices = jax.random.permutation(rng_key, len(x))

    # Select the first batch_size elements
    selected_x = x[indices[:batch_size], ...]
    selected_y = y[indices[:batch_size], ...]

    return selected_x, selected_y


def train_and_evaluate(config: argparse.ArgumentParser, train_ds: Tuple, test_ds: Tuple, random_batch: Tuple):
    "train model acording the config"
    
    # create a train state
    state, num_params = create_train_state(config, train_ds)
    state_fn = state.apply_fn

    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = config.sgd_seed
    rng = jax.random.PRNGKey(seed)
    train_batches = train_utils.data_stream(seed, train_ds, config.batch_size, augment = config.use_augment)
    test_batches = train_utils.data_stream(seed, test_ds, config.batch_size, augment = False)
    
    # store training results
    step_results = list()

    divergence = False

    subset_key = jax.random.PRNGKey(68)
    
    ########### TRAINING ##############

    for step in range(config.num_steps):     
        
        # every random_steps, train on a random batch

        if ((step+1) % config.random_steps == 0):

            ## measure the metrics for the random

            # select a random subset of the training set
            subset_key, _ = jax.random.split(subset_key)
            (x_train, y_train) = select_random_subset(*train_ds, config.batch_size, subset_key)

            #calculate loss and accuracy on the training dataset
            train_logits_step, train_loss_step = train_utils.loss_step(state, (x_train, y_train), config.loss_fn)
            train_accuracy_step = train_utils.compute_accuracy(train_logits_step, y_train)

            # calculate loss and accuracy on a batch of test dataset
            subset_key, _ = jax.random.split(subset_key)
            (x_test, y_test) = select_random_subset(*test_ds, config.batch_size, subset_key)
            
            test_logits_step, test_loss_step = train_utils.loss_step(state, (x_test, y_test), config.loss_fn)
            test_accuracy_step = train_utils.compute_accuracy(test_logits_step, y_test)

            ## train on the random dataset

            x_random, y_random = random_batch
            
            # train for a step
            state, random_logits_step, random_grads_step, random_loss_step = train_utils.train_step(state, random_batch, config.loss_fn)
            # estimate the norm of the gradients
            flat_grads_step, _ = jax.flatten_util.ravel_pytree(random_grads_step)
            grad_norm_step = jnp.linalg.norm(flat_grads_step)

            # estimate accuracy on the random dataset from the lohits
            random_accuracy_step = train_utils.compute_accuracy(random_logits_step, y_random)

            result = jnp.array([state.step, config.lr_trgt, train_loss_step, train_accuracy_step, test_loss_step, test_accuracy_step, random_loss_step, random_accuracy_step, grad_norm_step])
            step_results.append(result)

        else:

            ## measure the metrics for test and random set
            
            # calculate loss and accuracy on a batch of test dataset
            subset_key, _ = jax.random.split(subset_key)
            (x_test, y_test) = select_random_subset(*test_ds, config.batch_size, subset_key)
            
            test_logits_step, test_loss_step = train_utils.loss_step(state, (x_test, y_test), config.loss_fn)
            test_accuracy_step = train_utils.compute_accuracy(test_logits_step, y_test)
            
            # calculate loss and accuracy on the random batch
            x_random, y_random = random_batch
            
            random_logits_step, random_loss_step = train_utils.loss_step(state, random_batch, config.loss_fn)
            random_accuracy_step = train_utils.compute_accuracy(random_logits_step, y_random)

            ## train the model on the training set for one step

            batch = next(train_batches)
            imgs, targets = batch
        
            # train for one step
            state, train_logits_step, train_grads_step, train_loss_step = train_utils.train_step(state, batch, config.loss_fn)
            flat_grads_step, _ = jax.flatten_util.ravel_pytree(train_grads_step)
            grad_norm_step = jnp.linalg.norm(flat_grads_step)

            # estimate accuracy from logits
            train_accuracy_step = train_utils.compute_accuracy(train_logits_step, targets)
        
            result = jnp.array([state.step, config.lr_trgt, train_loss_step, train_accuracy_step, test_loss_step, test_accuracy_step, random_loss_step, random_accuracy_step, grad_norm_step])
            step_results.append(result)
        
        #check for divergence
        if (jnp.isnan(train_loss_step) or jnp.isinf(train_loss_step)): divergence = True; break
        
        print(f'step: {state.step}, train loss: {train_loss_step:0.4f}, train accuracy: {train_accuracy_step:0.4f}, test loss: {test_loss_step:0.4f}, test_accuracy: {test_accuracy_step:0.4f}, random loss: {random_loss_step:0.4f}, random accuracy: {random_accuracy_step:0.4f}, grad norm: {grad_norm_step:0.4f}') 
    step_results = jnp.asarray(step_results)
    step_results = jax.device_get(step_results)
    
    # Create a pandas dataframe
    df_train = pd.DataFrame(step_results, columns = ['step', 'lr', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'random_loss', 'random_accuracy', 'grad_norm'])
    # add some variables to the dataset
    df_train['init'] = config.init_seed; df_train['width'] = config.width; df_train['widening_factor'] = config.widening_factor; df_train['num_params'] = num_params; 
    df_train['lr_init'] = config.lr_init; df_train['lr_trgt'] = config.lr_trgt;df_train['num_steps'] = config.num_steps; 
    df_train['random_steps'] = config.random_steps; df_train['random_examples'] = config.random_examples; df_train['num_params'] = num_params
    # save the data
    path = f'{save_dir}/dynamics_{config.dataset}_m{config.random_examples}_{config.model}_n{config.width}_w{config.widening_factor}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt}_{config.schedule_name}_lr{config.lr_trgt:0.4f}_T{config.num_steps}_k{config.random_steps}_B{config.batch_size}_m{config.momentum}.tab'
    df_train.to_csv(path, sep = '\t')
    return divergence


#models = {'fcn_mup': model_utils.fcn_int, 'fcn_sp': model_utils.fcn_sp, 'myrtle_sp': model_utils.Myrtle, 'myrtle_mup': model_utils.Myrtle_int}
models = {'WideResNet16': model_utils.WideResNet16, 'WideResNet20': model_utils.WideResNet20, 'WideResNet28': model_utils.WideResNet28, 'WideResNet40': model_utils.WideResNet40}
loss_fns = {'mse': train_utils.mse_loss, 'xent': train_utils.cross_entropy_loss}
activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
# Dataset parameters
parser.add_argument('--dataset', type = str, default = 'cifar10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--num_examples', type = int, default = 50000)
parser.add_argument('--random_examples', type = int, default = 128)

# Model parameters
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--width', type = int, default = 16)
parser.add_argument('--widening_factor', type = int, default = 1)
parser.add_argument('--depth', type = int, default = 16)
parser.add_argument('--bias', type = str, default = 'True') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 1)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warm_steps', type = int, default = 512)
parser.add_argument('--num_steps', type = int, default = 10_000)
parser.add_argument('--random_steps', type = int, default = 10)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_trgt', type = float, default = 0.1)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 128)
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--test', type = str, default = 'True')
parser.add_argument('--measure_batches', type = int, default = 10)

config = parser.parse_args()

# Model parameters
config.model = f'WideResNet{config.depth}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False
config.act = activations[config.act_name]

config.loss_fn = loss_fns[config.loss_name]
# Optimization parameters
config.schedule_name = 'constant'


save_dir = 'resnet_results'
data_dir = '/nfshomes/dayal'

(x_train, y_train), (x_test, y_test) = data_utils.load_image_data(data_dir, config.dataset, flatten = False, num_examples = config.num_examples)

config.num_train, config.num_test = x_train.shape[0], x_test.shape[0]

print(config)

# #standardize the inputs
x_train = data_utils._standardize(x_train, abc = config.abc)
x_test = data_utils._standardize(x_test, abc = config.abc)

config.in_dim = int(jnp.prod(jnp.array(x_train.shape[1:])))

#get one hot encoding for the labels
y_train = data_utils._one_hot(y_train, config.out_dim)
y_test = data_utils._one_hot(y_test, config.out_dim)

# create random dataset
config.random_examples = config.batch_size # the random batch is of the same size as the batch size
x_random, y_random = create_random_dataset(config.random_examples, x_train.shape[1:], config.out_dim)
y_random = data_utils._one_hot(y_random, config.out_dim)

config.num_batches = train_utils.estimate_num_batches(config.num_train, config.batch_size)

### TRAIN THE NETWORK AND EVALUATE ####
divergence = train_and_evaluate(config, (x_train, y_train), (x_test, y_test), (x_random, y_random))

