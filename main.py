import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU

import yaml
import torch
import argparse
import numpy as np

from Dataset import *
from Loss import *
from Model import *
from Train import *
from Util import *
from Evaluation import *


DATASET_DICT = {
    'CIFAR-10': '/home/root1/yyl/data/Cifar_10',
    'STL-10': '/home/root1/yyl/data/STL_10',
    'ImageNet-10': '/home/root1/yyl/data/imagenet_10',
    'ImageNet-Dogs': '',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''Build Dataset from args'''
    if args.dataset == 'CIFAR-10':
        train_dataloader, test_dataloader = build_dataset_CIFAR10(
            DATASET_DICT['CIFAR-10'], args
        )
    elif args.dataset == 'STL-10':
        train_dataloader, test_dataloader = build_dataset_STL10(
            DATASET_DICT['STL-10'], args
        )
    elif args.dataset == 'ImageNet-10':
        train_dataloader, test_dataloader = build_dataset_ImageNet10(
            DATASET_DICT['ImageNet-10'], args
        )
    else:
        raise ValueError
    
    '''Init work and load checkpoint'''
    net = Network_CC(args.feature_dim, args.class_num).to('cuda')
    if args.model_load_name != '':
        model_load_path = os.path.join(args.model_path, args.model_load_name)
        if os.path.exists(model_load_path) == False:
            raise FileNotFoundError
        else:
            print('load model: {}'.format(model_load_path))
            checkpoint = torch.load(model_load_path)
            net.load_state_dict(checkpoint['net'])

    '''loss function and optimizer'''
    instance_loss = Instance_CC(args.batch_size, temperature=0.5, device='cuda').to('cuda')
    cluster_loss = Cluster_CC(args.class_num, temperature=1.0, device='cuda').to('cuda')
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    '''train loop'''
    best_acc = args.best_acc
    args.log_path = os.path.join(args.model_path, 'log.log')
    args.write_log = write_log
    for epoch in range(args.start_epoch, args.max_epochs):
        print('epoch: {}'.format(epoch))

        net = train_CC(
            net=net,
            optimizer=optimizer,
            instance_loss=instance_loss,
            cluster_loss=cluster_loss,
            train_dataloader=train_dataloader,
            args=args
        )

        acc, nmi, ari = Evaluation(
            net=net,
            test_loader=test_dataloader,
            dataset_size=args.test_size,
            test_batch_size=args.test_batch_size,
        )

        if acc > best_acc:
            best_acc = acc
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            with open(os.path.join(args.model_path, args.model_save_name), 'wb') as out:
                torch.save(state, out) 
        args.write_log(
            args.log_path, 
            "epoch: {} best acc: {} acc: {} nmi: {} ari: {}\n".format(epoch, best_acc, acc, nmi, ari)
        )
        