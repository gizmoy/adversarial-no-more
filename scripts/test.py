import os
import argparse
from datetime import datetime

from PIL import Image
import torch
import torch.nn as nn
import numpy as np

import src.data_loader.data_loaders as module_data
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.model.model as module_arch
from src.utils.parse_config import ConfigParser


def save_frame(frame, path):
    frame = frame.astype(np.float32)
    frame = np.moveaxis(frame, 0, 2)
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(frame).save(path)


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instance
    data_loader = config.init_obj('test_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss']['type'])
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

    experiment_name = config['name']
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_folder = os.path.join(config['trainer']['save_dir'], 'test', experiment_name, run_id)

    os.makedirs(save_folder)

    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(data_loader):
            for k, v in batch_sample.items():
                if '_path' not in k:
                    batch_sample[k] = v.to(device)

            output = model(batch_sample['frame0'], batch_sample['frame2'], batch_sample['frame4'])
            target = (batch_sample['frame1'], batch_sample['frame2'], batch_sample['frame3'])
            batch_size = batch_sample['frame0'].shape[0]

            frames1 = output[0].cpu().detach().numpy()
            frames3 = output[2].cpu().detach().numpy()
            true_frames0 = batch_sample['frame0'].cpu().detach().numpy()
            true_frames1 = batch_sample['frame1'].cpu().detach().numpy()
            true_frames2 = batch_sample['frame2'].cpu().detach().numpy()
            true_frames3 = batch_sample['frame3'].cpu().detach().numpy()
            true_frames4 = batch_sample['frame4'].cpu().detach().numpy()

            for i in range(batch_size):
                save_frame(frames1[i], os.path.join(save_folder, batch_sample['frame1_path'][i] + '.pred.png'))
                save_frame(frames3[i], os.path.join(save_folder, batch_sample['frame3_path'][i] + '.pred.png'))
                save_frame(true_frames0[i], os.path.join(save_folder, batch_sample['frame0_path'][i]))
                save_frame(true_frames1[i], os.path.join(save_folder, batch_sample['frame1_path'][i]))
                save_frame(true_frames2[i], os.path.join(save_folder, batch_sample['frame2_path'][i]))
                save_frame(true_frames3[i], os.path.join(save_folder, batch_sample['frame3_path'][i]))
                save_frame(true_frames4[i], os.path.join(save_folder, batch_sample['frame4_path'][i]))

                # computing loss, metrics on test set
                loss = loss_fn(output, target, **config['loss']['args'])
                total_loss += loss.item()
                for j, metric in enumerate(metric_fns):
                    total_metrics[j] += metric(output[0], target[0])
                    total_metrics[j] += metric(output[2], target[2])

        n_samples = len(data_loader.sampler) * 2
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Cartoon interpolation test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    main(config=ConfigParser.from_args(args))
