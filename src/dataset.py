from torch.utils import data
import torchvision


class Mnist(data.Dataset):

    def __init__(self, root, train, add_random_noise=False):
        self.root = root
        self.train = train
        self.add_random_noise = add_random_noise
        self.dataset = torchvision.datasets.MNIST(
            root, train, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
             ]))

        if self.add_random_noise:
            # add random noise as an extra class to predict
            self.dataset.classes.append('10 - random noise')

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class Cifar10(data.Dataset):

    def __init__(self, root, train, add_random_noise=False):
        self.root = root
        self.train = train
        self.add_random_noise = add_random_noise
        self.dataset = torchvision.datasets.CIFAR10(
            root, train, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))

        if self.add_random_noise:
            # add random noise as an extra class to predict
            self.dataset.classes.append('10 - random noise')

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
