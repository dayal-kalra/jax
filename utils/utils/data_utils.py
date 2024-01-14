import jax.numpy as jnp
import jax
import numpy as np
import pickle as pl

def _one_hot(x, k, dtype = jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def _standardize(x, abc = 'ntp'):
    """Standardization per sample across feature dimension."""
    axes = tuple(range(1, len(x.shape)))  
    mean = jnp.mean(x, axis = axes, keepdims = True)
    std_dev = jnp.std(x, axis = axes, keepdims = True)
    normx = (x - mean) / std_dev
    if abc == 'mup':
        in_dim = jnp.prod(jnp.array(x.shape[1:]))
        normx /= jnp.sqrt(in_dim)
    return normx

def _random_crop(x, pixels, key):
    """x should have shape [batch, img_dim, img_dim, channels]."""
    zero = (0, 0)
    pixpad = (pixels, pixels)
    paddedX = np.pad(x, (zero, pixpad, pixpad, zero), 'reflect')
    corner = jax.random.randint(key, (x.shape[0], 2), 0, 2 * pixels)
    assert x.shape[1] == x.shape[2]
    img_size = x.shape[1]
    slices = [(slice(int(o[0]), int(o[0]) + img_size), slice(int(o[1]), int(o[1]) + img_size), slice(None)) for x, o in zip(paddedX, corner)]
    paddedX = np.concatenate([x[np.newaxis, s[0], s[1], s[2]] for x, s in zip(paddedX, slices)])
    return paddedX

def _random_horizontal_flip(key, x, prob):
    """Perform horizontal flip with probability prob"""
    assert x.shape[1] == x.shape[2] # check wheather its a square image
    flip = jax.random.uniform(key, shape = (len(x), 1, 1, 1))
    flippedX = x[:, :, ::-1, :]
    x = jnp.where(flip < prob, flippedX, x)
    return x

def _random_vertical_flip(key, x, prob):
    """Perform vertical flip along axis with probability prob"""
    assert x.shape[1] == x.shape[2] # check wheather its a square image
    flip = jax.random.uniform(key, shape = (len(x), 1, 1, 1))
    flippedX = x[:, ::-1, :, :]
    x = jnp.where(flip < prob, flippedX, x)
    return x


def crop(key, batch):
    """Random flips and crops."""
    image, label = batch
    pixels = 4 #
    pixpad = (pixels, pixels)
    zero = (0, 0)
    padded_image = jnp.pad(image, (pixpad, pixpad, zero))
    corner = jax.random.randint(key, (2,), 0, 2 * pixels)
    corner = jnp.concatenate((corner, jnp.zeros((1,), jnp.int32)))
    img_size = (32, 32, 3)
    cropped_image = jax.lax.dynamic_slice(padded_image, corner, img_size)
    return cropped_image, label

crop = jax.vmap(crop, 0, 0)

def mixup(key, batch, alpha = 1.0):

    """
    Mixup data augmentation: Mixes two training examples with weight from beta distribution
                            for alpha = 1.0, it draws from uniform distribution

    """

    image, label = batch

    N = image.shape[0]

    weight = jax.random.beta(key, alpha, alpha, (N, 1))
    mixed_label = weight * label + (1.0 - weight) * label[::-1]

    weight = jnp.reshape(weight, (N, 1, 1, 1))
    mixed_image = weight * image + (1.0 - weight) * image[::-1]

    return mixed_image, mixed_label

def transform(key, batch):
    image, label = batch

    key, split = jax.random.split(key)

    image = _random_horizontal_flip(key, image, prob = 0.1)

    subkey, key = jax.random.split(key, )

    N = image.shape[0]

    image = jnp.reshape(image, (N, 32, 32, 3))

    image = jnp.where(jax.random.uniform(split, (N, 1, 1, 1)) < 0.5, image[:, :, ::-1], image)

    key, split = jax.random.split(key)
    batch_split = jax.random.split(split, N)
    image, label = crop(batch_split, (image, label))

    return mixup(key, (image, label), alpha = 1.0)

def load_image_data(dir: str, dataset: str, flatten: bool = True, num_examples: int = 1000):
    """
    Description: loads existing dataset from a directory

    Inputs: 
      1. dir: directory where the data is saved
      2. dataset: dataset name, the existing file should be dataset.dump
      3. num_examples: num_examples required
    """
    path = f'{dir}/datasets/{dataset}.dump'    
    with open(path, 'rb') as fi:
      (x_train, y_train), (x_test, y_test) = pl.load(fi)

    # Flatten the image for FCNs
    if flatten:
      x_train = x_train.reshape((x_train.shape[0], -1))
      x_test = x_test.reshape((x_test.shape[0], -1))
    
    # consider a subset of the existing dataset
    if num_examples < x_train.shape[0]:
      x_train = x_train[:num_examples]
      y_train = y_train[:num_examples]

    if num_examples < x_test.shape[0]:
      x_test = x_test[:num_examples]
      y_test = y_test[:num_examples]
      
    return (x_train, y_train), (x_test, y_test)


