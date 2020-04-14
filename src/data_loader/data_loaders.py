from src.data_loader.base import BaseDataLoader
from src.dataset import Mnist, Cifar10


class MnistDataLoader(BaseDataLoader):
    """
    Mnist Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, add_random_noise=False):
        self.data_dir = data_dir
        self.dataset = Mnist(root=self.data_dir, train=training, add_random_noise=add_random_noise)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR-10 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, add_random_noise=False):
        self.data_dir = data_dir
        self.dataset = Cifar10(root=self.data_dir, train=training, add_random_noise=add_random_noise)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
