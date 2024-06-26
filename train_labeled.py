# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

import argparse
import datetime
import json
import os
from os.path import join

import torch.utils.data as data

from networks.attgan import AttGAN
from trainer.attgan import Trainer
from helpers import Progressbar
from utils import set_seed
from tensorboardX import SummaryWriter


attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]
attrs_default = ['Male']

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sens', dest='sens', default="Male")
    parser.add_argument('--target', dest='target', default="Attractive")
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'UTKFace', 'FairFace'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/list_attr_celeba.txt')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=5)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1)
    parser.add_argument('--gpu', dest='gpu', type=int)
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    parser.add_argument('--ratio', dest='ratio', type=float, default=0.01)
    parser.add_argument('--seed', dest='seed', type=int, default=1)

    
    return parser.parse_args(args)


def main(args):
    set_seed(args.seed)

    args.lr_base = args.lr
    args.n_attrs = 1
    args.betas = (args.beta1, args.beta2)

    os.makedirs(join('output', args.experiment_name), exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'checkpoint_labeled'), exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'sample_training_labeled'), exist_ok=True)
    with open(join('output', args.experiment_name, 'setting_labeled.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if args.data == 'CelebA':
        from data_handler.data import CelebA
        train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.sens, args.target, args.ratio)
        valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.sens, args.target)
    else:
        pass
    
    args.it_per_epoch = len(train_dataset) // args.batch_size

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True
    )
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
        shuffle=False, drop_last=False
    )
    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

    attgan = AttGAN(args)
    progressbar = Progressbar()
    writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

    trainer = Trainer()
    trainer.set_valid_image(valid_dataloader, args)
    trainer.train_labeled(
        attgan,
        train_dataloader,
        progressbar,
        writer,
        args,
    )


if __name__ == '__main__':
    main(parse())
