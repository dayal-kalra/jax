from flax import linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax
from typing import Any, Callable, Sequence, Tuple
from functools import partial

activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}

def check_criticality(params, abc = 'sp'):
    if abc == 'sp':
        varws = tree_map(lambda x: x.shape[0]*jnp.var(x), params)
        print(varws)
    else:
        varws = tree_map(lambda x: jnp.var(x), params)
        print(varws)
    return


############################
########## UV model ########
############################

class UV_int(nn.Module):
    """
    Description: UV model that interpolates between NTP and muP 
    Inputs:
        1. width: width of the network
        2. out_dim: output dimension / number of classes
        3. 
    """
    width: int
    out_dim: int = 1
    varw: float = 1.0
    scale: float = 1.0
    
    def setup(self):
        # setup initialization for all but last layer
        kernel_init = jax.nn.initializers.normal(stddev = jnp.sqrt(self.varw / self.width**self.scale) ) 
        # First layer
        self.U = nn.Dense(self.width, use_bias = False, kernel_init = kernel_init) 
        # Last layer
        self.V = nn.Dense(self.out_dim, use_bias = False, kernel_init = kernel_init) 
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1)) # flatten the input
        h = self.U(x)
        f = self.V(h) / jnp.sqrt(self.width**(1-self.scale))
        return f

############################
########## FCNs ############
############################


class fcn_sp(nn.Module):
    """
    Description: A constant width FCN in standard parameterization
    Inputs:
        1. width: width of the network
        2. depth: depth of the network # number of layers
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name
    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    act_name: str
    scale: float = 1.0

    def setup(self):
        # setup initialization for all but last layer
        kernel_init = jax.nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        kernel_lst = jax.nn.initializers.variance_scaling(scale = 1.0, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer = [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = kernel_lst) ]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1)) # flatten the input
        for d, layer in enumerate(self.layers):
            x = layer(x)
            if d+1 != self.depth:
                x = self.act(x)
        return x


class fcn_int(nn.Module):
    """
    Description: A constant width FCN in a parameterization that interpolates between SP-like and muP parameterization
    This requires input to be normalized using the muP option, i.e., ||x|| = 1
    Inputs:
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name
        7. scale: interpolates between sp like (s = 0) and mup (s = 1) parameterization

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    act_name: str
    scale: float 

    def setup(self):
        # effective width
        alpha = self.scale / 2.0
        # setup initialization for all but last layer
        kernel_frst = jax.nn.initializers.normal(stddev = jnp.sqrt( self.varw / self.width**self.scale) ) 
        kernel_init = jax.nn.initializers.normal(stddev = jnp.sqrt(self.varw) / jnp.sqrt(self.width))
        kernel_lst = jax.nn.initializers.normal(stddev = 1.0 / jnp.sqrt(self.width))
        # setup activation
        self.act = activations[self.act_name]
        # First layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_frst)]
        # intermediate layers
        self.layers += tuple([nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-2)])
        # last layer with different initialization constant
        self.layers += tuple([nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = kernel_lst) ])
        return


    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        for d, layer in enumerate(self.layers):
            x = layer(x)
            if d == 0:
                x *= jnp.sqrt(self.width**self.scale)
            if d+1 != self.depth:
                x = self.act(x)
        x /= jnp.sqrt(self.width**self.scale)
        return x



###########################################
############## Myrtle CNNs ################
###########################################


#dictionary of pool list and depth

models = {
    'myrtle5': ([1, 2, 3], 4),
    'myrtle7': ([1, 3, 5], 6),
    'myrtle10': ([2, 5, 8], 9)
    }

ModuleDef = Any

class Block(nn.Module):
  """CNN block."""
  filters: int
  conv: ModuleDef
  act: Callable
  kernel: Tuple[int, int] = (3, 3)

  @nn.compact
  def __call__(self, x,):
    x = self.conv(self.filters, self.kernel)(x)
    x = self.act(x)
    return x
  

class Myrtle(nn.Module):
    """Myrtle CNNs"""
    num_filters: int
    num_layers: int
    pool_list: Sequence[int]
    num_classes: int = 10
    block_cls: ModuleDef = Block
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    use_bias: bool = False
    kernel: Tuple[int, int] = (3, 3)
    varw: float = 2.0
    scale: float = 0.0 # not used; just a dummy parameter

    @nn.compact
    def __call__(self, x):
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        conv = partial(self.conv, use_bias = self.use_bias, padding = 'SAME', strides = 1, kernel_init = kernel_init)
        for i in range(self.num_layers):
            x = self.block_cls(self.num_filters, conv = conv, act = self.act, kernel = self.kernel)(x)
            if i in self.pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))

        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = x.reshape((x.shape[0], -1))

        kernel_init_last =  nn.initializers.variance_scaling(scale = 1.0, distribution = 'normal', mode = 'fan_in')
        x = nn.Dense(self.num_classes, use_bias = self.use_bias, kernel_init = kernel_init_last)(x)
        x = jnp.asarray(x)
        return x
    

class Myrtle_int(nn.Module):
    """Myrtle CNNs"""
    num_filters: int
    num_layers: int
    pool_list: Sequence[int]
    num_classes: int = 10
    #dtype: Any = jnp.float32
    block_cls: ModuleDef = Block
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    use_bias: bool = False
    kernel: Tuple[int, int] = (3, 3)
    varw: float = 2.0
    scale: float = 1.0


    @nn.compact
    def __call__(self, x):
        alpha = self.scale / 2.0
        eff_width = self.num_filters * jnp.prod( jnp.array(self.kernel) )

        # initializations for different layers
        init_frst = jax.nn.initializers.normal(stddev = jnp.sqrt( self.varw/ (eff_width**self.scale) )  )
        init_rest = jax.nn.initializers.normal(stddev = jnp.sqrt(self.varw / eff_width))
        init_last = jax.nn.initializers.normal(stddev = jnp.sqrt(1 / eff_width))

        # First layer
        conv_init = partial(self.conv, use_bias = self.use_bias, padding = 'SAME', strides = 1, kernel_init = init_frst)
        x = self.block_cls(self.num_filters, conv = conv_init, act = self.act, kernel = self.kernel)(x)
        x *= eff_width**alpha

        # conv for rest of the layers
        conv = partial(self.conv, use_bias = self.use_bias, padding = 'SAME', strides = 1, kernel_init = init_rest)

        # Intermediate layers
        for i in range(1, self.num_layers):
            x = self.block_cls(self.num_filters, conv = conv, act = self.act, kernel = self.kernel)(x)
            if i in self.pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        # Pooling layers
        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = x.reshape((x.shape[0], -1))

        # Last layer
        x = nn.Dense(self.num_classes, use_bias = self.use_bias, kernel_init = init_last)(x)
        x = jnp.asarray(x) / eff_width**alpha
        return x


######################################
############  ResNets ################
######################################

ModuleDef = Any

class ResNetBlock(nn.Module):
    """ ResNet block """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)

class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        # Layer 1
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        # Layer 2
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        # Layer 3
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)
        # Projection to match different shapes
        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)
        return self.act(residual + y)
    

class ResNet(nn.Module):
    """ResNetV1.5."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    varw: float = 2.0
    scale: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')

        conv = partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init = kernel_init)
        #norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype, axis_name='batch',)
        norm = nn.LayerNorm

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x) # decreases the image size to half
        x = norm(name='ln_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME') # further decreases the size to half


        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i, strides=strides, conv=conv, norm=norm, act=self.act,)(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype, kernel_init = kernel_init)(x)
        x = jnp.asarray(x, self.dtype)
        return x
    


class ResNet_int(nn.Module):
    """ResNetV1.5."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    varw: float = 2.0
    scale: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        frst_width = self.num_filters * 7 * 7
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')

        conv = partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init = kernel_init)
        norm = nn.LayerNorm
        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x) # decreases the image size to half
        x *= jnp.sqrt(frst_width**self.scale)

        x = norm(name='ln_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME') # further decreases the size to half

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i, strides=strides, conv=conv, norm=norm, act=self.act,)(x)

        x = jnp.mean(x, axis=(1, 2))
        lst_width = x.shape[-1]
        x = nn.Dense(self.num_classes, dtype=self.dtype, kernel_init = kernel_init)(x)
        x = jnp.asarray(x, self.dtype) / jnp.sqrt(lst_width**self.scale)
        return x
    
ResNet18 = partial(ResNet, stage_sizes = [2, 2, 2, 2], block_cls = ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
)
ResNet101 = partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock
)
ResNet152 = partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock
)
ResNet200 = partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock
)


ResNet18Local = partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)

ResNet18_mup = partial(ResNet_int, stage_sizes = [2, 2, 2, 2], block_cls = ResNetBlock)



class WideResNetBlock(nn.Module):
    features: int
    strides: Tuple[int, int] = (1, 1)
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.features, kernel_size=(3, 3), strides=self.strides, padding='SAME')(x)
        y = nn.LayerNorm()(y)
        y = self.act(y)
        y = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(y)
        y = nn.LayerNorm()(y)

        if x.shape[-1] != self.features or self.strides != (1, 1):
            x = nn.Conv(self.features, kernel_size=(1, 1), strides = self.strides, padding = 'SAME')(x)

        return self.act(y + x)

class WideResNet(nn.Module):
    stage_sizes: Sequence[int]
    num_filters: int = 16
    widening_factor: int = 1
    num_classes: int = 10
    act: Callable = nn.relu
    varw: float = 1.0
    scale: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)

        for stage, num_blocks in enumerate(self.stage_sizes):
            for block in range(num_blocks):
                features = self.num_filters * (2 ** stage) * self.widening_factor
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = WideResNetBlock(features, strides = strides, act = self.act)(x)

        x = nn.avg_pool(x, window_shape=(8, 8)) # Adjust window_shape as needed
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_classes)(x)
        return x

WideResNet16 = partial(WideResNet, stage_sizes = [2, 2, 2])
WideResNet28 = partial(WideResNet, stage_sizes = [4, 4, 4])
WideResNet40 = partial(WideResNet, stage_sizes = [6, 6, 6])
