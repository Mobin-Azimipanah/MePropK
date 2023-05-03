'''
Train MLPs for MNIST using meProp
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from argparse import ArgumentParser

import torch

from data import get_mnist
from util import TestGroup


def get_args():

    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=512, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=32, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=False)
    return parser.parse_args()


def get_args_unified():

    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=5, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=500, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=50, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=True)
    return parser.parse_args()


def main():
    args = get_args()
    trn, dev, tst = get_mnist()
    print(f"k value: {args.k}")
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    group.run(0, args.n_epoch)

    group.run(args.k, args.n_epoch)
   
def main_unified():
    args = get_args_unified()
    trn, dev, tst = get_mnist()

    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    group.run(0)
   
    group.run()

if __name__ == '__main__':
    # uncomment to run meprop
    # main()
    # run unified meprop
    main()

