import numpy as np
import torch
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
from src.data_loader.cutout import cutout
from src.data_loader.random_noise import random_noise as noise
from torchvision.utils import make_grid
from src.trainer.base import BaseTrainer
from src.utils.utils import inf_loop, MetricTracker


class AdversarialTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_data_loader, add_clean, add_adversarial,
                 adversarial_methods, eps=None, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.add_clean = add_clean if adversarial_methods else True
        self.add_adversarial = add_adversarial if adversarial_methods else False
        self.adversarial_methods = adversarial_methods
        self.eps = eps
        self.data_groups = []

        if self.add_clean:
            self.data_groups.append('clean')
        if self.add_adversarial:
            self.data_groups += self.adversarial_methods

        self.config = config
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        metric_names = [f'combined__{m.__name__}' for m in self.metric_ftns]
        for group in self.data_groups:
            metric_names += [f'{group}__{m.__name__}' for m in self.metric_ftns]

        self.train_metrics = MetricTracker('loss', *metric_names, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *metric_names, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (clean_data, target) in enumerate(self.train_data_loader):
            clean_data, target = clean_data.to(self.device), target.to(self.device)
            data, target = self._generate_adversaries(clean_data, target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, **self.config['loss']['args'])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self._compute_metrics(self.train_metrics, loss, output, target, self.train_data_loader)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx),
                                                                           loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        for batch_idx, (clean_data, target) in enumerate(self.valid_data_loader):
            clean_data, target = clean_data.to(self.device), target.to(self.device)
            data, target = self._generate_adversaries(clean_data, target)

            output = self.model(data)
            loss = self.criterion(output, target, **self.config['loss']['args'])

            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self._compute_metrics(self.valid_metrics, loss, output, target, self.valid_data_loader)
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _generate_adversaries(self, clean_data, clean_target):
        data = []
        target = []

        for group in self.data_groups:
            if group == 'clean':
                data.append(clean_data)
                target.append(clean_target)
            elif group == 'pgd':
                pgd_data = projected_gradient_descent(self.model, clean_data, self.eps, 0.01, 40, np.inf)
                data.append(pgd_data)
                target.append(clean_target)
            elif group == 'fgsm':
                fgsm_data = fast_gradient_method(self.model, clean_data, self.eps, np.inf)
                data.append(fgsm_data)
                target.append(clean_target)
            elif group == 'cutout':
                cutout_data = cutout(clean_data).to(self.device)
                data.append(cutout_data)
                target.append(clean_target)
            elif group == 'random_noise':
                random_noise = noise(clean_data).to(self.device)
                data.append(random_noise)
                noise_class = self.model.num_classes - 1
                random_target = torch.Tensor(clean_target.shape[0]).fill_(noise_class).long().to(self.device)
                target.append(random_target)

        data = torch.cat(tuple(data), 0)
        target = torch.cat(tuple(target), 0)

        return data, target

    def _compute_metrics(self, metrics, loss, output, target, loader):
        metrics.update('loss', loss.item())

        for met in self.metric_ftns:
            metrics.update(f'combined__{met.__name__}', met(output, target))

        for i, group in enumerate(self.data_groups):
            for met in self.metric_ftns:
                bs = loader.batch_size

                # the last batch may have size less than batch size
                bs = min(bs, output.shape[0] // len(self.data_groups))

                group_output = output[i * bs: (i + 1) * bs]
                group_target = target[i * bs: (i + 1) * bs]
                metrics.update(f'{group}__{met.__name__}', met(group_output, group_target))
