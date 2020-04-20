import argparse

from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

import src.data_loader.data_loaders as module_data
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.model.model as module_arch
from src.data_loader.cutout import cutout
from src.data_loader.random_noise import random_noise as noise
from src.utils.parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('test_data_loader', module_data)

    # fix data groups
    data_groups = []
    if config['test']['adversarial']['add_clean']:
        data_groups.append('clean')
    if config['test']['adversarial']['add_adversarial']:
        data_groups += config['test']['adversarial']['adversarial_methods']

    # initialize iterative fgsm
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
    group_metrics = {group: torch.zeros(len(metric_fns)) for group in data_groups}

    for i, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if apply_ifgsm:
            data, target = iterative_fgsm(model, data, target,
                                          iters=config['test']['iterative_fgsm']['iters'],
                                          eps=config['test']['iterative_fgsm']['eps'])

        elif data_groups:
            data, target = generate_adversaries(data, target, data_groups, device, model,
                                                eps=config['test']['adversarial']['eps'])

        output = model(data)

        # computing loss, metrics on test set
        loss = criterion(output, target, **config['loss']['args'])
        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        for j, metric in enumerate(metric_fns):
            total_metrics[j] += metric(output, target) * batch_size
        for i, group in enumerate(group_metrics):
            bs = min(data_loader.batch_size, output.shape[0] // len(data_groups))
            group_output = output[i * bs: (i + 1) * bs]
            group_target = target[i * bs: (i + 1) * bs]
            for j, metric in enumerate(metric_fns):
                group_metrics[group][j] += metric(group_output, group_target) * bs

    div = n_samples * (len(data_groups) if data_groups else 1)
    log = {'loss': total_loss / div}
    combined_metrics = {f'combined__{met.__name__}': total_metrics[i].item() / div
                        for i, met in enumerate(metric_fns)}
    group_metrics = {f'{group}__{met.__name__}': metrics[i].item() / n_samples
                        for group, metrics in group_metrics.items() for i, met in enumerate(metric_fns) }
    log.update({**combined_metrics, **group_metrics})
    logger.info(log)


def iterative_fgsm(model, data, target, iters, eps):
    all_data = [data]
    for i in range(iters):
        fgsm_data = fast_gradient_method(model, all_data[i], eps, np.inf)
        all_data.append(fgsm_data)
    data = torch.cat(tuple(all_data[1:]), 0)
    target = torch.cat(tuple([target] * iters), 0)
    return data, target


def compute_partial_metrics(metric_ftns, data_groups, output, target, loader):
    res = {}

    for i, group in enumerate(data_groups):
        for met in metric_ftns:
            bs = min(loader.batch_size, output.shape[0] // len(data_groups))
            group_output = output[i * bs: (i + 1) * bs]
            group_target = target[i * bs: (i + 1) * bs]
            res[f'{group}__{met.__name__}'] = met(group_output, group_target)

    return res


def generate_adversaries(clean_data, clean_target, data_groups, device, model, eps):
    data = []
    target = []

    for group in data_groups:
        if group == 'clean':
            data.append(clean_data)
            target.append(clean_target)
        elif group == 'pgd':
            pgd_data = projected_gradient_descent(model, clean_data, eps, 0.01, 40, np.inf)
            data.append(pgd_data)
            target.append(clean_target)
        elif group == 'fgsm':
            fgsm_data = fast_gradient_method(model, clean_data, eps, np.inf)
            data.append(fgsm_data)
            target.append(clean_target)
        elif group == 'cutout':
            cutout_data = cutout(clean_data).to(device)
            data.append(cutout_data)
            target.append(clean_target)
        elif group == 'random_noise':
            random_noise = noise(clean_data).to(device)
            data.append(random_noise)
            noise_class = model.num_classes - 1
            random_target = torch.Tensor(clean_target.shape[0]).fill_(noise_class).long().to(device)
            target.append(random_target)

    data = torch.cat(tuple(data), 0)
    target = torch.cat(tuple(target), 0)

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
