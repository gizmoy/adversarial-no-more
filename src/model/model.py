import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseModel(nn.Module):

    def __init__(self, add_random_noise, num_classes):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.add_random_noise = add_random_noise
        if self.add_random_noise:
            self.num_classes += 1


class ShallowMnistModel(BaseModel):

    def __init__(self, add_random_noise, num_classes=10):
        super(ShallowMnistModel, self).__init__(add_random_noise, num_classes)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class DeepMnistModel(BaseModel):

    def __init__(self, add_random_noise, num_classes=10):
        super(DeepMnistModel, self).__init__(add_random_noise, num_classes)

        self.net =  nn.Sequential(
            self._conv_block(1, 32),
            self._conv_block(32, 64),
            Flatten(),
            self._fc(25600, 128),
        )

    def forward(self, x):
        return self.net(x)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            self._conv(in_channels,  out_channels, kernel=(3, 3), stride=(1, 1), pad=(0, 0)),
            self._conv(out_channels, out_channels, kernel=(3, 3), stride=(1, 1), pad=(0, 0)),
            self._conv(out_channels, out_channels, kernel=(5, 5), stride=(1, 1), pad=(2, 2), dropout=0.4)
        )

    def _conv(self, in_channels, out_channels, kernel, stride, pad, dropout=None):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, pad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        return nn.Sequential(*layers)

    def _fc(self, in_channels, hid_channels):
        return nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.4),
            nn.Linear(hid_channels, self.num_classes),
        )
