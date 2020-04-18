import argparse

from cleverhans.future.torch.attacks import fast_gradient_method
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

import src.data_loader.data_loaders as module_data
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.model.model as module_arch
from src.utils.parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('test_data_loader', module_data)
    apply_ifgsm = config['test']['iterative_fgsm']['apply']
    n_samples = len(data_loader.sampler)
    if apply_ifgsm:
        n_samples *= config['test']['iterative_fgsm']['iters']

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']['type'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    for i, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if apply_ifgsm:
            data, target = iterative_fgsm(model, data, target,
                                          iters=config['test']['iterative_fgsm']['iters'],
                                          eps=config['test']['iterative_fgsm']['eps'])

        output = model(data)

        #
        # save sample images, or do something with output here
        #

        # computing loss, metrics on test set
        loss = criterion(output, target, **config['loss']['args'])
        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        for j, metric in enumerate(metric_fns):
            total_metrics[j] += metric(output, target) * batch_size

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


def iterative_fgsm(model, data, target, iters, eps):
    all_data = [data]
    for i in range(iters):
        fgsm_data = fast_gradient_method(model, all_data[i], eps, np.inf)
        all_data.append(fgsm_data)
    data = torch.cat(tuple(all_data[1:]), 0)
    target = torch.cat(tuple([target] * iters), 0)
    return data, target


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    main(config=ConfigParser.from_args(args))
