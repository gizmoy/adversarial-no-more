from src.data_loader.base import BaseDataLoader
from src.dataset import Mnist, Cifar10


class MnistDataLoader(BaseDataLoader):
    """
    Mnist Data Loader
    """
    def __init__(self, root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, train=True):
        self.root = root
        self.dataset = Mnist(self.root, train=train)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR-10 Data Loader
    """
    def __init__(self, root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, train=True):
        self.root = root
        self.dataset = Cifar10(self.root, train=train)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
