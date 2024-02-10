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
parser.add_argument('--init_seed', type = int, default = 1)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warm_steps', type = int, default = 512)
parser.add_argument('--num_steps', type = int, default = 100_000)
parser.add_argument('--random_steps', type = int, default = 10)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_trgt', type = float, default = 0.1)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--measure_batches', type = int, default = 10)

config = parser.parse_args()

# Model parameters
config.model = f'WideResNet{config.depth}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False

# Optimization parameters
config.schedule_name = 'linear'

save_dir = 'resnet_results'
data_dir = '/home/dayal'

fo = open('left.list', 'w')

lr_trgts = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
widths = [1, 2, 4, 8, 12, 16]
ks = [10, 100, 1000, 10000]

for lr_trgt in lr_trgts:
    for w in widths:
        for k in ks:
            config.random_steps = k
            config.lr_trgt = lr_trgt
            config.widening_factor = w
            
            try:
                # save the data
                path = f'{save_dir}/dynamics_{config.dataset}_m{config.random_examples}_{config.model}_n{config.width}_w{config.widening_factor}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt}_{config.schedule_name}_lr{config.lr_trgt:0.4f}_T{config.num_steps}_k{config.random_steps}_B{config.batch_size}_m{config.momentum}_M{config.measure_batches}.tab'
                df = pd.read_csv(path, sep = '\t')
            except FileNotFoundError:
                print(path)
                fo.write(f'srun python3 train_resnet_memorization_warmup_constant.py --dataset cifar10 --loss_name xent --abc sp --width 16 --widening_factor {config.widening_factor} --depth 16 --batch_size 128 --lr_trgt {config.lr_trgt} --momentum 0.9 --num_steps 100_000 --random_steps {config.random_steps} --init_seed {config.init_seed} --augment False\n')
                

