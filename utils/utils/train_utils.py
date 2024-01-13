#Some imports
import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from flax import core
from flax import struct
from jax.numpy.linalg import norm
from jax.experimental import sparse
import numpy as np
import utils.data_utils as data_utils

class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node = False)
    params: core.FrozenDict[str, Any]
    opt: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value."""
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step = self.step + 1, params = new_params, opt_state = new_opt_state, **kwargs,)

    def update_learning_rate(self, *, learning_rate):
        """ Updates the learning rate"""
        self.opt_state.hyperparams['learning_rate'] = learning_rate
        return

    def get_optimizer_hparams(self,):
        return self.opt_state.hyperparams

    @classmethod
    def create(cls, *, apply_fn, params, opt, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = opt.init(params)
        return cls(step = 0, apply_fn = apply_fn, params = params, opt = opt, opt_state = opt_state, **kwargs, )

def cross_entropy_loss(logits, labels):
  return optax.softmax_cross_entropy(logits = logits, labels = labels).mean()

def mse_loss(logits, labels):
    """ MSE loss used while measuring the state"""
    return 0.5 * jnp.mean((logits - labels) ** 2)

@jax.jit
def compute_accuracy(logits, targets):
    """ Accuracy, used while measuring the state"""
    # Get the label of the one-hot encoded target
    target_class = jnp.argmax(targets, axis = 1)
    # Predict the class of the batch of images using
    predicted_class = jnp.argmax(logits, axis = 1)
    return jnp.mean(predicted_class == target_class)

@partial(jax.jit, static_argnums = 2)
def gradients_step(state: TrainState, batch: Tuple, loss_function):
    "Compute gradients for a single batch"
    x, y = batch

    def loss_fn(params):
        "loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    return grads, loss

@partial(jax.jit, static_argnums = 3)
def loss_step(state: TrainState, batch: Tuple, params, loss_function):
    "Compute loss for a single batch"
    x, y = batch
    logits = state.apply_fn({'params': params}, x)
    loss = loss_function(logits, y)
    return loss

@partial(jax.jit, static_argnums = 2)
def train_step(state: TrainState, batch: Tuple, loss_function):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, logits, loss

def compute_hessian(state, loss_function, batches, num_batches = 10, power_iterations = 20):
    top_hessian = 0
    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        top_hessian_batch, _ = hessian_step(state, batch, loss_function, power_iterations)
        top_hessian += top_hessian_batch
    top_hessian = top_hessian / num_batches
    return top_hessian


def compute_metrics(state, loss_function, batches, num_examples, batch_size):
    """
    Description: Estimates the loss and accuracy of a batched data stream

    Input:
        1. state: a Trainstate instance
        2. loss function:
        3. batches: a batched datastream
        4. num_examples: number of examples in the dataset
        5. batch_size: batch size

    """
    total_loss = 0
    total_accuracy = 0

    num_batches = estimate_num_batches(num_examples, batch_size)

    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        #calculate logits
        logits = state.apply_fn({'params': state.params}, x)
        #calculate loss and accuracy
        total_loss += loss_function(logits, y)
        total_accuracy += compute_accuracy(logits, y)

    ds_loss = total_loss / num_batches
    ds_accuracy = total_accuracy / num_batches
    return ds_loss, ds_accuracy

@partial(jax.jit, static_argnums = 2)
def hessian_step(state: TrainState, batch: Tuple, loss_function, power_iterations: int = 20):
    "Compute top eigenvalue of the hessian using power iterations"
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)

    #  i here is only for fori_loop usage
    def fori_hvp(i, v):
        return body_hvp(v / norm(v))

    # Power Iteration
    key = jax.random.PRNGKey(24)
    v = jax.random.normal(key, shape=flat_params.shape)
    v = v / norm(v)
    v = jax.lax.fori_loop(0, power_iterations, fori_hvp, v / norm(v))
    top_eigen_value = jnp.vdot(v / norm(v), hvp(flat_params, v / norm(v)))

    return top_eigen_value, v


@partial(jax.jit, static_argnums = 2)
def hessian_spectrum_step(state: TrainState, batch: Tuple, loss_function, vs, m = 100, tol = 1e-10):
    "Compute top-k eigenvalue and hessian"
    x, y = batch
    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = mse_loss(logits, y)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss
    
    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]
    
    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / jnp.linalg.norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m, tol = tol)
    return eigs, eigvs, n_iter

def data_stream(seed, ds, batch_size, augment = False):
    " Creates a data stream with a predifined batch size."
    train_images, train_labels = ds
    num_train = train_images.shape[0]
    num_batches = estimate_num_batches(num_train, batch_size)
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)

    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1)*batch_size]
            x_batch = train_images[batch_idx]
            y_batch = train_labels[batch_idx]
            if augment:
                key, subkey = jax.random.split(key)
                x_batch, y_batch = data_utils.transform(subkey, (x_batch, y_batch) )
            yield x_batch, y_batch

def estimate_num_batches(num_train, batch_size):
    "Estimates number of batches using dataset and batch size"
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches
