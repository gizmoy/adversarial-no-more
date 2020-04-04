from torch.utils import data
import torchvision


class Mnist(data.Dataset):

    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.dataset = torchvision.datasets.MNIST(
            root, train, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
             ]))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class Cifar10(data.Dataset):

    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.dataset = torchvision.datasets.CIFAR10(
            root, train, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
