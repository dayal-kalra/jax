import numpy as np
import pandas as pd
import argparse
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
parser.add_argument('--init_seed', type = int, default = 24)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warm_steps', type = int, default = 512)
parser.add_argument('--num_steps', type = int, default = 100_000)
parser.add_argument('--random_steps', type = int, default = 10)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_trgt', type = float, default = 0.03)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--batch_size', type = int, default = 128)

config = parser.parse_args()
# Model parameters
config.model = f'WideResNet{config.depth}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False

# Optimization parameters
config.schedule_name = 'constant'

save_dir = 'resnet_results'

path = f'{save_dir}/dynamics_{config.dataset}_m{config.random_examples}_{config.model}_n{config.width}_w{config.widening_factor}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt}_{config.schedule_name}_lr{config.lr_trgt:0.4f}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}.tab'

df = pd.read_csv(path, sep = '\t')
print(df)

num_steps_per_epoch = config.num_examples / config.batch_size

df_eot = df.loc[df['step'] > config.num_steps - num_steps_per_epoch]

df_mean = df_eot.mean()
print(df_mean)
