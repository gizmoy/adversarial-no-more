import argparse
import collections
import random
import torch
import numpy as np

import src.data_loader.data_loaders as module_data
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.model.model as module_arch
import src.trainer as module_trainer
from src.utils.parse_config import ConfigParser
from src.trainer.simple_trainer import Trainer


# fix random seeds for reproducibility
SEED = random.randint(0, 2**30)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = train_data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch).cuda()
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']['type'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer_class = getattr(module_trainer, config['trainer']['type'])
    trainer = trainer_class(model, criterion, metrics, optimizer,
                            config=config,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler, **config['trainer']['args'])



    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Cartoon Interpolation Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    main(config=ConfigParser.from_args(args, options))
